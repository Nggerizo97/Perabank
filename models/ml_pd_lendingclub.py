"""Modelo de probabilidad de default (PD) de originación sobre LendingClub.

Predice, con lo que se sabe del solicitante al momento de originar, la probabilidad
de que el préstamo termine castigado o en default.

Decisiones de diseño:

- Población: solo préstamos madurados con desenlace conocido (ver
  etl/gold/lendingclub.py). Incluir cosechas inmaduras sesga la tasa de default.
- Validación fuera de tiempo: se entrena con cosechas viejas, se calibra con la
  siguiente y se evalúa con la última. Un split aleatorio mezclaría años y
  sobrestimaría el desempeño frente a cosechas futuras.
- Features en lista blanca (FEATURES), no en lista negra: una columna nueva en gold
  no entra al modelo por accidente. EXCLUIDAS documenta por qué queda fuera cada
  columna disponible que podría parecer útil.
- Benchmark: el sub_grade que asignó LendingClub. Es la salida de su propio modelo
  de riesgo, así que no es feature; sirve para saber si este modelo aporta algo.
"""
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from etl.common import warehouse  # noqa: E402  (requiere REPO_ROOT en sys.path)
from etl.gold.lendingclub import COLUMNAS_POST_ORIGINACION, TABLE_NAME  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "pd_lendingclub_v1.joblib"
RANDOM_STATE = 42

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("perabank.pd")

FEATURES_NUMERICAS = [
    "loan_amnt", "term", "emp_length", "annual_inc", "monto_sobre_ingreso", "dti", "delinq_2yrs",
    "fico_promedio", "meses_historial_credito", "inq_last_6mths", "mths_since_last_delinq",
    "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "mort_acc", "pub_rec_bankruptcies", "acc_open_past_24mths", "bc_util", "num_actv_rev_tl",
    "tot_cur_bal", "total_rev_hi_lim",
]
FEATURES_CATEGORICAS = ["home_ownership", "verification_status", "purpose", "application_type"]
FEATURES = FEATURES_NUMERICAS + FEATURES_CATEGORICAS

EXCLUIDAS = {
    "grade": "salida del modelo de riesgo de LendingClub; se usa como benchmark",
    "sub_grade": "salida del modelo de riesgo de LendingClub; se usa como benchmark",
    "int_rate": "precio fijado por LendingClub a partir de su sub_grade",
    "installment": "derivada de int_rate y del plazo",
    "funded_amnt": "lo decide LendingClub tras evaluar; loan_amnt es lo que pide el solicitante",
    "initial_list_status": "decisión operativa de LendingClub sobre cómo listar el préstamo",
    "addr_state": "geografía: proxy potencial de atributos protegidos (riesgo de redlining)",
    "zip_code": "geografía: proxy potencial de atributos protegidos (riesgo de redlining)",
    "fico_range_low": "reemplazada por fico_promedio",
    "fico_range_high": "reemplazada por fico_promedio",
}

# Disponibilidad de variables de buró: mort_acc, bc_util, tot_cur_bal y similares no
# existen antes de 2012 y están completas desde 2013. Los préstamos a 60 meses solo
# maduran (plazo + gracia) hasta 2015, así que calibración y prueba son solo 36 meses.
VENTANAS = {
    "entrenamiento": ("2013-01-01", "2015-12-31"),
    "calibracion": ("2016-01-01", "2016-12-31"),
    "prueba": ("2017-01-01", "2017-12-31"),
}

SUBGRADOS = [f"{g}{n}" for g in "ABCDEFG" for n in range(1, 6)]

assert not set(FEATURES) & set(COLUMNAS_POST_ORIGINACION), "feature con información posterior a originar"
assert not set(FEATURES) & set(EXCLUIDAS), "feature marcada como excluida"


def cargar_dataset() -> pd.DataFrame:
    """Préstamos madurados con desenlace, en orden determinístico por id."""
    columnas = [c for c in FEATURES if c != "monto_sobre_ingreso"]
    return warehouse.query(f"""
        SELECT id_prestamo, issue_d, sub_grade, {', '.join(columnas)},
               loan_amnt / NULLIF(annual_inc, 0) AS monto_sobre_ingreso,
               es_default
        FROM {TABLE_NAME}
        WHERE madurado AND es_default IS NOT NULL
        ORDER BY id_prestamo
    """)


def dividir_por_cosecha(df: pd.DataFrame) -> dict:
    fecha = pd.to_datetime(df["issue_d"])
    return {
        nombre: df[fecha.between(inicio, fin)].reset_index(drop=True)
        for nombre, (inicio, fin) in VENTANAS.items()
    }


def matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Features listas para el modelo. Las categóricas van como dtype category: el
    boosting fija su codificación con las del entrenamiento y trata una categoría
    nueva como desconocida en vez de fallar."""
    X = df[FEATURES].copy()
    for col in FEATURES_CATEGORICAS:
        X[col] = X[col].astype("category")
    return X


def ks(y: np.ndarray, proba: np.ndarray) -> float:
    """Kolmogorov-Smirnov: máxima separación entre las distribuciones acumuladas de
    score de buenos y malos. Métrica estándar en scoring de crédito."""
    fpr, tpr, _ = roc_curve(y, proba)
    return float(np.max(tpr - fpr))


def entrenar(train: pd.DataFrame, calibracion: pd.DataFrame, **params) -> CalibratedClassifierCV:
    config = dict(
        categorical_features="from_dtype",
        learning_rate=0.05,
        max_iter=2000,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=RANDOM_STATE,
    ) | params
    modelo = HistGradientBoostingClassifier(**config).fit(matriz(train), train["es_default"].astype(int))
    logger.info("Boosting: %s de %s iteraciones (early stopping)", modelo.n_iter_, config["max_iter"])

    # Isotónica sobre una cosecha posterior: ajusta el nivel de PD a la tasa de
    # default más reciente sin reentrenar el ordenamiento del boosting.
    calibrado = CalibratedClassifierCV(FrozenEstimator(modelo), method="isotonic")
    return calibrado.fit(matriz(calibracion), calibracion["es_default"].astype(int))


def evaluar(modelo: CalibratedClassifierCV, df: pd.DataFrame) -> dict:
    y = df["es_default"].astype(int).to_numpy()
    proba = modelo.predict_proba(matriz(df))[:, 1]
    auc = roc_auc_score(y, proba)

    # Benchmark: el orden de sub_grade (A1 mejor ... G5 peor) como score de riesgo.
    rango_subgrado = df["sub_grade"].map({s: i for i, s in enumerate(SUBGRADOS)})
    con_subgrado = rango_subgrado.notna().to_numpy()
    auc_subgrado = roc_auc_score(y[con_subgrado], rango_subgrado[con_subgrado])

    deciles = pd.qcut(proba, 10, labels=False, duplicates="drop")
    calibracion = (pd.DataFrame({"decil": deciles, "pd_predicha": proba, "default_observado": y})
                   .groupby("decil").agg(prestamos=("pd_predicha", "size"),
                                         pd_predicha=("pd_predicha", "mean"),
                                         default_observado=("default_observado", "mean")))
    return {
        "n": int(len(y)),
        "tasa_default": float(y.mean()),
        "pd_media": float(proba.mean()),
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks(y, proba),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "benchmark_subgrado_auc": float(auc_subgrado),
        "benchmark_subgrado_gini": float(2 * auc_subgrado - 1),
        "calibracion_deciles": calibracion.round(4).to_dict("index"),
    }


def main():
    if TABLE_NAME not in warehouse.tables():
        logger.error("No existe %s en gold. Corre primero el ETL de LendingClub.", TABLE_NAME)
        return

    partes = dividir_por_cosecha(cargar_dataset())
    for nombre, parte in partes.items():
        logger.info("%-13s %9s préstamos | default %.2f%%", nombre, f"{len(parte):,}",
                    100 * parte["es_default"].mean())

    modelo = entrenar(partes["entrenamiento"], partes["calibracion"])
    metricas = {nombre: evaluar(modelo, partes[nombre]) for nombre in ("calibracion", "prueba")}

    m = metricas["prueba"]
    logger.info("PRUEBA (fuera de tiempo, %s): AUC %.4f | Gini %.4f | KS %.4f | Brier %.4f",
                VENTANAS["prueba"][0][:4], m["auc"], m["gini"], m["ks"], m["brier"])
    logger.info("Benchmark sub_grade de LendingClub: AUC %.4f | Gini %.4f",
                m["benchmark_subgrado_auc"], m["benchmark_subgrado_gini"])
    logger.info("PD media %.2f%% vs default observado %.2f%%", 100 * m["pd_media"], 100 * m["tasa_default"])
    logger.info("Calibración por decil:\n%s", pd.DataFrame(m["calibracion_deciles"]).T.to_string())

    artefacto = {
        "modelo": modelo,
        "features": {"numericas": FEATURES_NUMERICAS, "categoricas": FEATURES_CATEGORICAS},
        "excluidas": EXCLUIDAS,
        "ventanas": VENTANAS,
        "metricas": metricas,
        "metadata": {
            "entrenado_en": datetime.now(timezone.utc).isoformat(),
            "fuente": TABLE_NAME,
            "poblacion": "préstamos madurados (plazo + gracia) con desenlace conocido",
        },
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, MODEL_PATH)
    logger.info("Artefacto guardado: %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
