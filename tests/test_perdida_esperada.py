import numpy as np
import pandas as pd
import pytest

from models import ml_pd_lendingclub as pd_model
from models import ml_perdida_esperada as el


def _castigados(term, lgd, ead_ratio, n):
    return pd.DataFrame({"term": term, "lgd_realizada": lgd, "ead_ratio_realizado": ead_ratio}, index=range(n))


def test_parameters_come_from_most_recent_window_with_enough_charge_offs(monkeypatch):
    monkeypatch.setattr(el, "MIN_CASTIGADOS", 3)
    reciente = pd.concat([
        _castigados(36, 0.9, 0.6, 3),
        _castigados(60, 0.5, 0.5, 2),                              # muy pocos: no alcanza
        pd.DataFrame({"term": [36], "lgd_realizada": [np.nan], "ead_ratio_realizado": [np.nan]}),  # pagado
    ])
    antigua = pd.concat([_castigados(36, 0.1, 0.1, 5), _castigados(60, 0.8, 0.7, 4)])

    params = el.parametros_lgd_ead({"calibracion": reciente, "entrenamiento": antigua})

    assert params.loc[36, ["lgd", "ead_ratio", "castigados", "ventana"]].tolist() == [0.9, 0.6, 3, "calibracion"]
    assert params.loc[60, ["lgd", "ead_ratio", "castigados", "ventana"]].tolist() == [0.8, 0.7, 4, "entrenamiento"]


def test_expected_loss_formula():
    params = pd.DataFrame({"lgd": [0.9, 0.8], "ead_ratio": [0.5, 0.6]}, index=pd.Index([36, 60], name="term"))
    df = pd.DataFrame({"term": [36, 60], "funded_amnt": [1000, 2000]})

    el_ = el.perdida_esperada(df, np.array([0.1, 0.2]), params)

    assert el_.tolist() == pytest.approx([0.1 * 0.9 * 0.5 * 1000, 0.2 * 0.8 * 0.6 * 2000])


def test_expected_loss_rejects_terms_without_parameters():
    params = pd.DataFrame({"lgd": [0.9], "ead_ratio": [0.5]}, index=pd.Index([36], name="term"))
    with pytest.raises(ValueError, match="60"):
        el.perdida_esperada(pd.DataFrame({"term": [60], "funded_amnt": [1000]}), np.array([0.1]), params)


def test_validation_compares_portfolio_and_deciles():
    n = 100
    df = pd.DataFrame({"funded_amnt": np.full(n, 1000.0), "perdida_realizada": np.where(np.arange(n) < 10, 500.0, 0.0)})
    pd_predicha = np.linspace(0.01, 0.5, n)
    esperada = pd.Series(np.full(n, 50.0))

    v = el.validar(df, esperada, pd_predicha)

    assert v["perdida_esperada"] == 5000 and v["perdida_realizada"] == 5000
    assert v["ratio_esperada_realizada"] == 1
    assert v["tasa_perdida_esperada"] == pytest.approx(0.05)
    assert sum(d["prestamos"] for d in v["por_decil_pd"].values()) == n


def test_predictability_check_reports_no_lift_on_noise():
    rng = np.random.default_rng(0)

    def ventana(n):
        df = pd.DataFrame({c: rng.normal(size=n) for c in pd_model.FEATURES_NUMERICAS})
        for c in pd_model.FEATURES_CATEGORICAS:
            df[c] = rng.choice(["a", "b"], n)
        df["lgd_realizada"] = rng.uniform(0.7, 1.0, n)            # ruido: nada que aprender
        return df

    r = el.evaluar_predictibilidad(ventana(2000), ventana(500), "lgd_realizada")

    assert r["n_entrenamiento"] == 2000 and r["n_prueba"] == 500
    assert r["r2_modelo"] < 0.05
    assert r["mae_modelo"] >= r["mae_promedio"] * 0.95
