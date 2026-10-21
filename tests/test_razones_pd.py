import numpy as np
import pandas as pd
import pytest

from models import razones_pd


def predecir_lineal(df: pd.DataFrame) -> np.ndarray:
    """Modelo de juguete con efectos conocidos: FICO bajo, DTI alto, renta y mora
    reciente suben la PD; sin mora (NaN) no suma nada."""
    mora = df["mths_since_last_delinq"].astype(float)
    return (
        0.10
        + 0.002 * (700 - df["fico_promedio"].astype(float))
        + 0.003 * (df["dti"].astype(float) - 15)
        + 0.05 * (df["home_ownership"] == "RENT")
        + np.where(mora.isna(), 0.0, 0.04)
    ).to_numpy()


REFS = {"fico_promedio": 700.0, "dti": 15.0, "mths_since_last_delinq": float("nan"), "home_ownership": "MORTGAGE"}


def test_references_use_paid_loans_and_missing_majority_means_never():
    train = pd.DataFrame({
        "es_default": [False, False, False, True],
        "fico_promedio": [700, 720, 710, 600],
        "mths_since_last_delinq": [np.nan, np.nan, 30, 2],
        "home_ownership": ["MORTGAGE", "MORTGAGE", "RENT", "RENT"],
    })
    refs = razones_pd.referencias(train, ["fico_promedio", "mths_since_last_delinq"], ["home_ownership"])

    assert refs["fico_promedio"] == 710                       # mediana de los pagados, ignora el default
    assert np.isnan(refs["mths_since_last_delinq"])          # la mayoría de los pagados nunca tuvo mora
    assert refs["home_ownership"] == "MORTGAGE"


def test_contributions_recover_known_effects():
    solicitantes = pd.DataFrame({
        "fico_promedio": [650.0, 700.0], "dti": [25.0, 15.0],
        "mths_since_last_delinq": [6.0, np.nan], "home_ownership": ["RENT", "MORTGAGE"],
    })
    contrib = razones_pd.contribuciones(predecir_lineal, solicitantes, REFS)

    assert contrib.loc[0].to_dict() == pytest.approx(
        {"fico_promedio": 0.10, "dti": 0.03, "mths_since_last_delinq": 0.04, "home_ownership": 0.05})
    assert contrib.loc[1].abs().max() == pytest.approx(0)    # ya es el perfil de referencia


def test_reasons_are_ordered_worded_and_limited_to_adverse_factors():
    fila = pd.Series({"fico_promedio": 650.0, "dti": 10.0, "mths_since_last_delinq": 6.0, "home_ownership": "RENT"})
    contrib = pd.Series({"fico_promedio": 0.10, "dti": -0.015, "mths_since_last_delinq": 0.04, "home_ownership": 0.05})

    salida = razones_pd.razones(fila, contrib, REFS, top=4)

    assert [r["razon"] for r in salida] == [
        "Puntaje FICO bajo",
        "Tenencia de vivienda: RENT",
        "Meses desde la última mora: 6",
    ]                                                          # el DTI baja la PD: no es razón adversa
    assert salida[0]["aumento_pd"] == pytest.approx(0.10)
    assert salida[2]["referencia"] is None                    # referencia "nunca tuvo mora"


def test_missing_bureau_data_is_worded_as_missing():
    fila = pd.Series({"fico_promedio": np.nan})
    salida = razones_pd.razones(fila, pd.Series({"fico_promedio": 0.02}), {"fico_promedio": 700.0})
    assert salida[0]["razon"] == "Sin información de puntaje fico"


def test_reason_frequency_over_a_population():
    contrib = pd.DataFrame({"a": [0.3, 0.1, -0.1], "b": [0.2, 0.2, 0.0], "c": [0.1, 0.3, 0.05]})
    freq = razones_pd.frecuencia_razones(contrib, top=1)
    assert freq.to_dict() == {"c": 2 / 3, "a": 1 / 3, "b": 0.0}


def test_every_model_feature_has_a_description():
    from models.ml_pd_lendingclub import FEATURES
    assert set(FEATURES) <= set(razones_pd.DESCRIPCIONES)
