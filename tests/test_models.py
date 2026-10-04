"""Guardas del modelo de crédito: sin fuga del target y entrenamiento reproducible.

Se arma un warehouse gold mínimo a partir del silver sintético y se apunta el módulo
de entrenamiento a él, para probar las consultas reales y no una copia de ellas.
"""
import numpy as np
import pandas as pd
import pytest

from etl.common import warehouse
from etl.gold.dimensions import build_dim_cliente
from etl.gold.facts import build_fact_campana_marcado
from models import ml_perabanck_official as ml

# Columnas que la capa gold deriva del target tiene_mora (ver build_fact_campana_marcado).
CREDIT_LEAKAGE = {"tiene_mora", "spread_tasa_credito", "tasa_oferta_estimada"}


@pytest.fixture
def gold_db(fake_silver, gold_dir):
    warehouse.write_table(build_dim_cliente(), "dim_cliente")
    warehouse.write_table(build_fact_campana_marcado(), "fact_campana_marcado")
    warehouse.write_table(pd.DataFrame({
        "tipo_tasa": ["TREASURY_10Y", "TBILL_3M", "IBR"],
        "valor_tasa": [4.2, 5.1, 9.0],
    }), "fact_tasas_mercado")
    return gold_dir


def test_credit_dataset_excludes_target_derived_columns(gold_db):
    X, y, numericas, categoricas = ml.build_credit_dataset()

    assert CREDIT_LEAKAGE.isdisjoint(X.columns)
    assert set(numericas) | set(categoricas) == set(X.columns)
    assert len(X) == len(y) > 0


def test_no_credit_feature_is_a_copy_of_the_target(gold_db):
    """Centinela genérico: si alguien agrega una columna que replica el target,
    este test falla aunque no esté en la lista explícita de fuga."""
    X, y, numericas, _ = ml.build_credit_dataset()
    for col in numericas:
        values = pd.to_numeric(X[col], errors="coerce")
        assert not values.equals(y.astype(values.dtype)), f"'{col}' replica el target"


def test_credit_model_trains_reproducibly(gold_db):
    X, y, numericas, categoricas = ml.build_credit_dataset()

    first = ml.train_model("credit", X, y, numericas, categoricas)
    second = ml.train_model("credit", X, y, numericas, categoricas)

    assert {"pipeline", "features", "categorias", "defaults", "metricas"} <= first.keys()
    proba = first["pipeline"].predict_proba(X)[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    # n_jobs=-1 promedia los árboles en orden no determinístico: igualdad salvo ruido de redondeo.
    np.testing.assert_allclose(proba, second["pipeline"].predict_proba(X)[:, 1], rtol=0, atol=1e-12)


def test_train_model_skips_single_class_target():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert ml.train_model("x", X, pd.Series([0, 0, 0]), ["a"], []) == {}
