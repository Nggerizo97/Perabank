"""Entrenamiento de los modelos de riesgo de PeraBank contra el warehouse Gold (Parquet vía DuckDB).

Se entrenan TRES modelos, uno por grano real de datos. No se unen entre sí porque las
fuentes no comparten identidad de cliente: unir un proveedor del SECOP con un cliente
de campaña de depósitos sería inventar una relación que no existe en los datos.

  1. credit    -> riesgo de mora minorista   (dim_cliente + fact_campana_marcado + macro)
  2. fraud     -> fraude transaccional tarjeta (fact_fraude_tarjeta, componentes PCA)
  3. factoring -> cumplimiento de pago estatal (dim_proveedor_estatal + fact_contrato_estatal)

Los tres se guardan en un único artefacto joblib como diccionario, junto con el esquema
de features que la app de Streamlit usa para construir sus formularios.
"""
from datetime import datetime, timezone
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from etl.common import warehouse  # noqa: E402  (requiere REPO_ROOT en sys.path)
from etl.common.config import GOLD_DIR  # noqa: E402

MODEL_PATH = REPO_ROOT / "perabank_risk_pipeline_v1.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("perabank.ml")

load_dotenv(REPO_ROOT / ".env")
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Construcción de datasets, uno por grano
# ---------------------------------------------------------------------------

def build_credit_dataset() -> tuple:
    """Riesgo de mora minorista. Join real vía sk_cliente y sk_fecha.

    Se EXCLUYEN a propósito spread_tasa_credito y tasa_oferta_estimada: la capa gold
    las deriva de tiene_mora, así que usarlas como features sería fuga del target
    (el modelo leería la respuesta en la pregunta y daría un AUC irreal de ~1.0).
    """
    df = warehouse.query("""
        SELECT c.edad, c.ocupacion, c.estado_civil, c.nivel_educativo,
               f.balance_eur, f.tiene_hipoteca, f.tiene_prestamo_personal,
               f.duracion_contacto_seg, f.resultado_previo,
               f.tasa_ibr_referencia, f.suscrito_deposito,
               f.tiene_mora
        FROM fact_campana_marcado f
        JOIN dim_cliente c ON c.sk_cliente = f.sk_cliente
        ORDER BY f.id_campana_contacto
    """)

    # Contexto macro real del warehouse: promedio de los benchmarks de tesorería.
    macro = warehouse.query("""
        SELECT tipo_tasa, AVG(valor_tasa) AS valor
        FROM fact_tasas_mercado
        WHERE tipo_tasa IN ('TREASURY_10Y', 'TBILL_3M')
        GROUP BY tipo_tasa
    """).set_index("tipo_tasa")["valor"]
    df["tasa_treasury_10y"] = macro.get("TREASURY_10Y", np.nan)
    df["tasa_tbill_3m"] = macro.get("TBILL_3M", np.nan)

    y = df.pop("tiene_mora").astype(int)
    categoricas = ["ocupacion", "estado_civil", "nivel_educativo", "resultado_previo"]
    numericas = [c for c in df.columns if c not in categoricas]
    return df, y, numericas, categoricas


def build_fraud_dataset() -> tuple:
    """Fraude con tarjeta. Los componentes PCA V1..V28 son la señal predictiva real."""
    df = warehouse.query("SELECT * FROM fact_fraude_tarjeta")
    df = df.drop(columns=["id_evento_tarjeta", "source_system"], errors="ignore")

    y = df.pop("es_fraude").astype(int)
    categoricas = ["banda_riesgo_macro"]
    numericas = [c for c in df.columns if c not in categoricas]
    return df, y, numericas, categoricas


def build_factoring_dataset() -> tuple:
    """Cumplimiento de pago en contratación estatal, para calificar factoring.

    Target: el contrato superó el 50% de ejecución de pago.
    Se EXCLUYEN valor_pagado y valor_pendiente: ambos definen el target aritméticamente.
    """
    df = warehouse.query("""
        SELECT f.id_contrato, f.sk_proveedor, f.valor_del_contrato, f.valor_pagado,
               f.departamento, f.estado_contrato, f.tipo_de_contrato,
               f.modalidad_de_contratacion, p.es_pyme
        FROM fact_contrato_estatal f
        JOIN dim_proveedor_estatal p ON p.sk_proveedor = f.sk_proveedor
        WHERE f.valor_del_contrato > 0
        ORDER BY f.id_contrato
    """)

    # Volumen de contratación por proveedor: señal legítima de trayectoria.
    volumen = df.groupby("sk_proveedor")["id_contrato"].transform("count")
    df["contratos_del_proveedor"] = volumen

    ratio = (df["valor_pagado"] / df["valor_del_contrato"]).clip(0, 1)
    y = (ratio >= 0.5).astype(int)

    df = df.drop(columns=["id_contrato", "sk_proveedor", "valor_pagado"])
    categoricas = ["departamento", "estado_contrato", "tipo_de_contrato", "modalidad_de_contratacion"]
    numericas = [c for c in df.columns if c not in categoricas]
    return df, y, numericas, categoricas


# ---------------------------------------------------------------------------
# Entrenamiento y evaluación
# ---------------------------------------------------------------------------

def train_model(nombre: str, X: pd.DataFrame, y: pd.Series,
                numericas: list, categoricas: list) -> dict:
    logger.info("=" * 70)
    logger.info("Modelo '%s': %s filas, %s features (%s num / %s cat)",
                nombre, len(X), X.shape[1], len(numericas), len(categoricas))
    logger.info("Distribución del target: %s", dict(y.value_counts()))

    if y.nunique() < 2:
        logger.error("Modelo '%s' omitido: el target tiene una sola clase.", nombre)
        return {}

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numericas),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categoricas),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_train, y_train)

    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = pipeline.predict(X_test)

    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)
    matriz = confusion_matrix(y_test, pred)

    logger.info("ROC-AUC: %.4f | PR-AUC: %.4f (baseline PR = %.4f)",
                roc_auc, pr_auc, y_test.mean())
    logger.info("Matriz de confusión:\n%s", matriz)
    logger.info("Reporte:\n%s", classification_report(y_test, pred, zero_division=0))

    importancias = _feature_importances(pipeline)
    logger.info("Top 10 features:\n%s", importancias.head(10).to_string())

    return {
        "pipeline": pipeline,
        "features": {"numericas": numericas, "categoricas": categoricas},
        "categorias": {c: sorted(X[c].dropna().astype(str).unique().tolist()) for c in categoricas},
        "defaults": _defaults(X, numericas),
        "metricas": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "baseline_pr": float(y_test.mean()),
            "confusion_matrix": matriz.tolist(),
            "n_filas": int(len(X)),
            "positivos": int(y.sum()),
        },
        "importancias": importancias.to_dict(),
    }


def _feature_importances(pipeline: Pipeline) -> pd.Series:
    nombres = pipeline.named_steps["preprocessor"].get_feature_names_out()
    valores = pipeline.named_steps["classifier"].feature_importances_
    return pd.Series(valores, index=nombres).sort_values(ascending=False)


def _defaults(X: pd.DataFrame, numericas: list) -> dict:
    """Valores medianos por feature numérica, para prellenar el formulario de scoring."""
    return {c: float(X[c].median()) for c in numericas if pd.notna(X[c].median())}


def upload_to_s3(path: Path) -> None:
    """Sube el artefacto a S3 solo si las credenciales están configuradas."""
    bucket = os.getenv("AWS_S3_BUCKET_ML")
    if not (bucket and os.getenv("AWS_ACCESS_KEY_ID")):
        logger.info("Variables AWS no configuradas: el modelo queda solo en local (%s).", path)
        return
    try:
        import boto3
        cliente = boto3.client(
            "s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
        cliente.upload_file(str(path), bucket, f"models/{path.name}")
        logger.info("Modelo subido a s3://%s/models/%s", bucket, path.name)
    except Exception as e:
        logger.warning("No se pudo subir a S3 (%s). El modelo local sigue siendo válido.", e)


def main():
    if not warehouse.tables():
        logger.error("No hay tablas gold en %s. Corre primero: python -m etl.run_pipeline", GOLD_DIR)
        return

    artefacto = {
        "credit": train_model("credit", *build_credit_dataset()),
        "fraud": train_model("fraud", *build_fraud_dataset()),
        "factoring": train_model("factoring", *build_factoring_dataset()),
        "metadata": {
            "entrenado_en": datetime.now(timezone.utc).isoformat(),
            "fuente": str(GOLD_DIR),
            "nota_granos": (
                "Tres modelos independientes. Las fuentes no comparten identidad de "
                "cliente, así que no se unen features entre granos."
            ),
        },
    }

    joblib.dump(artefacto, MODEL_PATH)
    logger.info("Artefacto guardado: %s (%.1f MB)", MODEL_PATH, MODEL_PATH.stat().st_size / 1e6)
    upload_to_s3(MODEL_PATH)


if __name__ == "__main__":
    main()
