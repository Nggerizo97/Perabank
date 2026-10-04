"""Pérdida esperada (EL = PD × LGD × EAD) sobre LendingClub, validada contra la
pérdida que de verdad ocurrió.

- PD: el modelo calibrado de ml_pd_lendingclub.py (se carga su artefacto).
- LGD y EAD: promedios por plazo (36/60 meses) sobre los préstamos castigados de la
  cosecha madurada más reciente que tenga suficientes castigados de ese plazo, igual
  que la PD se calibra con la cosecha más reciente. La razón es una deriva medida: en
  36 meses, los defaults de 2016-2017 llegan antes (16.3 vs 17.3 meses hasta el último
  pago en 2013) y con más capital pendiente (EAD 0.61 vs 0.58). Con parámetros de
  2013-2015, la pérdida esperada de 2017 quedaba 5% por debajo de la realizada.

¿Por qué promedios y no modelos? Se probó modelar LGD y EAD con boosting sobre las
mismas features de originación y, fuera de tiempo, no superan al promedio (R² ≈ 0).
Tiene sentido: la severidad la decide la cobranza después del castigo, y la exposición
depende de cuándo llega el default; nada de eso se conoce al originar. Publicar un
modelo que no aporta solo añade riesgo. evaluar_predictibilidad() repite esa
comparación en cada corrida, así que si los datos cambian la decisión se revisa.
"""
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from etl.common import warehouse  # noqa: E402  (requiere REPO_ROOT en sys.path)
from etl.gold.lendingclub import TABLE_NAME  # noqa: E402
from models import ml_pd_lendingclub as pd_model  # noqa: E402

ARTIFACT_PATH = REPO_ROOT / "models" / "perdida_esperada_v1.joblib"

# Mínimo de castigados para estimar LGD/EAD de un plazo en una ventana; con menos,
# se usa la ventana anterior.
MIN_CASTIGADOS = 1000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("perabank.el")


def cargar_cartera() -> pd.DataFrame:
    """Préstamos madurados cuya pérdida realizada se puede medir: pagados (pérdida 0)
    o castigados. Excluye los que están en Default sin castigar: aún no registran
    recuperaciones, así que su pérdida real se desconoce."""
    columnas = [c for c in pd_model.FEATURES if c != "monto_sobre_ingreso"]
    return warehouse.query(f"""
        SELECT id_prestamo, issue_d, sub_grade, funded_amnt, {', '.join(columnas)},
               loan_amnt / NULLIF(annual_inc, 0)  AS monto_sobre_ingreso,
               es_default, lgd_realizada,
               ead_al_default / funded_amnt      AS ead_ratio_realizado,
               perdida_realizada
        FROM {TABLE_NAME}
        WHERE madurado AND perdida_realizada IS NOT NULL
        ORDER BY id_prestamo
    """)


def parametros_lgd_ead(ventanas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """LGD y EAD (como fracción del monto desembolsado) promedio por plazo, sobre los
    castigados con exposición positiva. `ventanas` va de la más reciente a la más
    antigua; cada plazo toma la primera con al menos MIN_CASTIGADOS."""
    parametros = {}
    for nombre, df in ventanas.items():
        castigados = df[df["lgd_realizada"].notna()]
        resumen = castigados.groupby("term").agg(
            lgd=("lgd_realizada", "mean"),
            ead_ratio=("ead_ratio_realizado", "mean"),
            castigados=("lgd_realizada", "size"),
        )
        for term, fila in resumen.iterrows():
            if term not in parametros and fila["castigados"] >= MIN_CASTIGADOS:
                parametros[term] = {**fila.to_dict(), "ventana": nombre}
    return pd.DataFrame.from_dict(parametros, orient="index").rename_axis("term").sort_index()


def perdida_esperada(df: pd.DataFrame, pd_predicha: np.ndarray, parametros: pd.DataFrame) -> pd.Series:
    faltantes = set(df["term"]) - set(parametros.index)
    if faltantes:
        raise ValueError(f"Sin LGD/EAD estimadas para los plazos {sorted(faltantes)}")
    lgd = df["term"].map(parametros["lgd"])
    ead = df["term"].map(parametros["ead_ratio"]) * df["funded_amnt"]
    return pd.Series(pd_predicha, index=df.index) * lgd * ead


def validar(df: pd.DataFrame, el: pd.Series, pd_predicha: np.ndarray) -> dict:
    """Pérdida esperada vs realizada, a nivel cartera y por decil de PD."""
    real = df["perdida_realizada"]
    deciles = pd.qcut(pd_predicha, 10, labels=False, duplicates="drop")
    por_decil = (pd.DataFrame({"decil": deciles, "el": el.to_numpy(), "real": real.to_numpy(),
                               "monto": df["funded_amnt"].to_numpy()})
                 .groupby("decil")
                 .agg(prestamos=("el", "size"), perdida_esperada=("el", "sum"),
                      perdida_realizada=("real", "sum"), monto=("monto", "sum")))
    por_decil["tasa_esperada"] = por_decil["perdida_esperada"] / por_decil["monto"]
    por_decil["tasa_realizada"] = por_decil["perdida_realizada"] / por_decil["monto"]
    return {
        "prestamos": int(len(df)),
        "monto_desembolsado": float(df["funded_amnt"].sum()),
        "perdida_esperada": float(el.sum()),
        "perdida_realizada": float(real.sum()),
        "ratio_esperada_realizada": float(el.sum() / real.sum()),
        "tasa_perdida_esperada": float(el.sum() / df["funded_amnt"].sum()),
        "tasa_perdida_realizada": float(real.sum() / df["funded_amnt"].sum()),
        "por_decil_pd": por_decil.round(4).to_dict("index"),
    }


def evaluar_predictibilidad(train: pd.DataFrame, test: pd.DataFrame, objetivo: str) -> dict:
    """¿Supera un modelo de originación al promedio para este objetivo, fuera de tiempo?"""
    train = train[train[objetivo].notna()]
    test = test[test[objetivo].notna()]
    modelo = HistGradientBoostingRegressor(
        categorical_features="from_dtype", learning_rate=0.05, max_iter=300,
        min_samples_leaf=200, random_state=pd_model.RANDOM_STATE,
    ).fit(pd_model.matriz(train), train[objetivo])
    pred = np.clip(modelo.predict(pd_model.matriz(test)), 0, 1)
    media = np.full(len(test), train[objetivo].mean())
    return {
        "n_entrenamiento": int(len(train)),
        "n_prueba": int(len(test)),
        "mae_modelo": float(mean_absolute_error(test[objetivo], pred)),
        "mae_promedio": float(mean_absolute_error(test[objetivo], media)),
        "r2_modelo": float(r2_score(test[objetivo], pred)),
    }


def main():
    if not pd_model.MODEL_PATH.exists():
        logger.error("Falta %s. Entrena primero: python models/ml_pd_lendingclub.py", pd_model.MODEL_PATH)
        return
    artefacto_pd = joblib.load(pd_model.MODEL_PATH)

    partes = pd_model.dividir_por_cosecha(cargar_cartera())
    parametros = parametros_lgd_ead({"calibracion": partes["calibracion"],
                                     "entrenamiento": partes["entrenamiento"]})
    logger.info("LGD y EAD por plazo (cosecha madurada más reciente con datos):\n%s",
                parametros.round(4).to_string())

    predictibilidad = {
        objetivo: evaluar_predictibilidad(partes["entrenamiento"], partes["prueba"], objetivo)
        for objetivo in ("lgd_realizada", "ead_ratio_realizado")
    }
    for objetivo, r in predictibilidad.items():
        logger.info("%s: MAE modelo %.4f vs promedio %.4f | R² modelo %.4f",
                    objetivo, r["mae_modelo"], r["mae_promedio"], r["r2_modelo"])

    # Solo la prueba es fuera de muestra: los parámetros de 36 meses salen de calibración.
    validacion = {}
    for nombre in ("calibracion", "prueba"):
        df = partes[nombre]
        pd_predicha = artefacto_pd["modelo"].predict_proba(pd_model.matriz(df))[:, 1]
        validacion[nombre] = v = validar(df, perdida_esperada(df, pd_predicha, parametros), pd_predicha)
        logger.info("%-11s %s préstamos | EL %.2f%% vs realizada %.2f%% del monto | EL/realizada %.3f",
                    nombre, f"{v['prestamos']:,}", 100 * v["tasa_perdida_esperada"],
                    100 * v["tasa_perdida_realizada"], v["ratio_esperada_realizada"])
    logger.info("Prueba por decil de PD:\n%s", pd.DataFrame(validacion["prueba"]["por_decil_pd"]).T
                [["prestamos", "tasa_esperada", "tasa_realizada"]].to_string())

    joblib.dump({
        "parametros_lgd_ead": parametros,
        "predictibilidad_lgd_ead": predictibilidad,
        "validacion": validacion,
        "metadata": {
            "entrenado_en": datetime.now(timezone.utc).isoformat(),
            "modelo_pd": str(pd_model.MODEL_PATH.name),
            "modelo_pd_entrenado_en": artefacto_pd["metadata"]["entrenado_en"],
            "formula": "EL = PD × LGD[plazo] × EAD_ratio[plazo] × funded_amnt",
        },
    }, ARTIFACT_PATH)
    logger.info("Artefacto guardado: %s", ARTIFACT_PATH)


if __name__ == "__main__":
    main()
