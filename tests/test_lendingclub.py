"""LendingClub de punta a punta: CSV crudo -> bronze -> silver -> gold -> dataset de PD.

Se usa un CSV sintético con el mismo formato del archivo real (columna índice sin
nombre, tasas como " 10.65%", plazos como " 36 months", fechas como "Dec-2011").
"""
import numpy as np
import pandas as pd
import pytest

from etl.bronze.ingest_lendingclub import ingest
from etl.common import warehouse
from etl.gold.dimensions import build_dim_fecha
from etl.gold.lendingclub import COLUMNAS_POST_ORIGINACION, build_fact_prestamo_minorista
from etl.silver.lendingclub import transform
from models import ml_pd_lendingclub as pd_model
from schemas.gold_schemas import GOLD_FACT_PRESTAMO_MINORISTA_CONTRACT

FILA_BASE = {
    "id": "1", "loan_amnt": "10000", "funded_amnt": "10000", "term": " 36 months", "int_rate": " 10.65%",
    "installment": "325.5", "grade": "B", "sub_grade": "B2", "emp_length": "10+ years",
    "home_ownership": "RENT", "annual_inc": "50000", "verification_status": "Verified",
    "issue_d": "Dec-2011", "loan_status": "Fully Paid", "purpose": "debt_consolidation",
    "addr_state": "CA", "zip_code": "940xx", "dti": "15.0", "delinq_2yrs": "0",
    "earliest_cr_line": "Jan-2000", "fico_range_low": "700", "fico_range_high": "704",
    "inq_last_6mths": "1", "mths_since_last_delinq": "", "mths_since_last_record": "",
    "open_acc": "10", "pub_rec": "0", "revol_bal": "5000", "revol_util": "50.5%", "total_acc": "20",
    "initial_list_status": "w", "application_type": "Individual", "mort_acc": "1",
    "pub_rec_bankruptcies": "0", "acc_open_past_24mths": "3", "bc_util": "40.0",
    "num_actv_rev_tl": "4", "tot_cur_bal": "20000", "total_rev_hi_lim": "15000",
    "total_pymnt": "11000", "total_rec_prncp": "10000", "recoveries": "0",
    "collection_recovery_fee": "0", "last_pymnt_d": "Jan-2015",
}


def fila(**cambios) -> dict:
    return FILA_BASE | cambios


@pytest.fixture
def capas(tmp_path, gold_dir):
    dirs = {"bronze": tmp_path / "bronze", "silver": tmp_path / "silver", "gold": gold_dir}
    gold_dir.mkdir(parents=True)
    build_dim_fecha("2007-01-01", "2021-12-31").to_parquet(gold_dir / "dim_fecha.parquet", index=False)
    return dirs


def correr_etl(capas, tmp_path, filas: list[dict]):
    raw = tmp_path / "loans.csv"
    # index=True reproduce la columna índice sin nombre del archivo real.
    pd.DataFrame(filas).to_csv(raw, index=True)
    ingest(raw, capas["bronze"])
    transform(capas["bronze"], capas["silver"])
    build_fact_prestamo_minorista(capas["silver"], capas["gold"])
    return warehouse.query("SELECT * FROM fact_prestamo_minorista ORDER BY id_prestamo")


def test_bronze_rejects_file_missing_contract_columns(tmp_path):
    raw = tmp_path / "loans.csv"
    pd.DataFrame([{k: v for k, v in FILA_BASE.items() if k != "loan_status"}]).to_csv(raw, index=True)
    with pytest.raises(ValueError, match="loan_status"):
        ingest(raw, tmp_path / "bronze")


def test_silver_parses_formatted_text_fields(capas, tmp_path):
    gold = correr_etl(capas, tmp_path, [
        fila(id="1", emp_length="10+ years"),
        fila(id="2", emp_length="< 1 year", term=" 60 months", int_rate=" 7.5%", revol_util=""),
        fila(id="3", emp_length="n/a"),
    ])
    silver = pd.read_parquet(capas["silver"] / "lendingclub.parquet").set_index("id")

    assert silver.loc[1, "int_rate"] == pytest.approx(10.65)
    assert silver.loc[2, "int_rate"] == pytest.approx(7.5)
    assert silver.loc[1, "revol_util"] == pytest.approx(50.5)
    assert pd.isna(silver.loc[2, "revol_util"])
    assert silver["term"].tolist() == [36, 60, 36]
    assert silver.loc[1, "emp_length"] == 10 and silver.loc[2, "emp_length"] == 0
    assert pd.isna(silver.loc[3, "emp_length"])
    assert pd.Timestamp(silver.loc[1, "issue_d"]) == pd.Timestamp("2011-12-01")
    assert gold["meses_historial_credito"].tolist() == [143, 143, 143]  # Jan-2000 -> Dec-2011
    assert gold["fico_promedio"].tolist() == [702.0] * 3


def test_silver_drops_non_loans_and_duplicates(capas, tmp_path):
    gold = correr_etl(capas, tmp_path, [
        fila(id="1"),
        fila(id="1"),                                       # duplicado
        fila(id="Total amount funded in policy code 1"),    # fila de resumen
        fila(id="4", loan_status=""),                       # sin estado
        fila(id="5", issue_d="sin fecha"),
    ])
    assert gold["id_prestamo"].tolist() == [1]


def test_gold_outcome_and_maturity_rules(capas, tmp_path):
    # El corte es el último issue_d presente: Jan-2016.
    gold = correr_etl(capas, tmp_path, [
        fila(id="1", issue_d="Jan-2012", loan_status="Charged Off"),
        fila(id="2", issue_d="Jan-2012", loan_status="Default"),
        fila(id="3", issue_d="Jan-2012", loan_status="Does not meet the credit policy. Status:Fully Paid"),
        fila(id="4", issue_d="Jul-2012", loan_status="Current"),         # 36 + 6 meses = justo el corte
        fila(id="5", issue_d="Aug-2012", loan_status="Late (31-120 days)"),
        fila(id="6", issue_d="Jan-2016", loan_status="Fully Paid"),
    ]).set_index("id_prestamo")

    assert gold["es_default"].tolist()[:3] == [True, True, False]
    assert gold.loc[[4, 5], "es_default"].isna().all()               # sin desenlace no es "buen pagador"
    assert gold["madurado"].tolist() == [True, True, True, True, False, False]
    assert gold.loc[1, "sk_fecha"] == 20120101
    assert (gold["fecha_corte"] == pd.Timestamp("2016-01-01")).all()


def test_gold_fails_on_dates_outside_dim_fecha(capas, tmp_path):
    build_dim_fecha("2015-01-01", "2015-12-31").to_parquet(capas["gold"] / "dim_fecha.parquet", index=False)
    with pytest.raises(ValueError, match="sk_fecha -> dim_fecha"):
        correr_etl(capas, tmp_path, [fila(id="1", issue_d="Dec-2011")])


def test_pd_dataset_only_has_matured_loans_with_outcome(capas, tmp_path):
    correr_etl(capas, tmp_path, [
        fila(id="2", issue_d="Jan-2012", loan_status="Charged Off", loan_amnt="5000", annual_inc="20000"),
        fila(id="1", issue_d="Jan-2012", loan_status="Fully Paid"),
        fila(id="3", issue_d="Jan-2012", loan_status="Current"),
        fila(id="4", issue_d="Jan-2016", loan_status="Charged Off"),
    ])
    df = pd_model.cargar_dataset()

    assert df["id_prestamo"].tolist() == [1, 2]
    assert df["es_default"].tolist() == [False, True]
    assert df.loc[1, "monto_sobre_ingreso"] == pytest.approx(0.25)
    assert set(pd_model.FEATURES) <= set(df.columns)


def test_pd_features_are_origination_only_and_exist_in_gold():
    gold_cols = set(GOLD_FACT_PRESTAMO_MINORISTA_CONTRACT.get_column_names())
    assert set(pd_model.FEATURES) - {"monto_sobre_ingreso"} <= gold_cols
    assert not set(pd_model.FEATURES) & set(COLUMNAS_POST_ORIGINACION)
    assert not set(pd_model.FEATURES) & set(pd_model.EXCLUIDAS)


def _dataset_sintetico(n_por_anio=600, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    partes = []
    for anio in range(2013, 2018):
        n = n_por_anio
        df = pd.DataFrame({c: rng.normal(size=n) for c in pd_model.FEATURES_NUMERICAS})
        df["fico_promedio"] = rng.normal(700, 30, n)
        df["home_ownership"] = rng.choice(["RENT", "OWN", "MORTGAGE"], n)
        df["verification_status"] = rng.choice(["Verified", "Not Verified"], n)
        df["purpose"] = rng.choice(["credit_card", "car"] if anio < 2017 else ["credit_card", "proposito_nuevo"], n)
        df["application_type"] = "Individual"
        riesgo = 1 / (1 + np.exp((df["fico_promedio"] - 690) / 15))
        df["es_default"] = rng.random(n) < riesgo
        df["issue_d"] = pd.Timestamp(f"{anio}-06-01")
        df["sub_grade"] = pd.cut(riesgo, 5, labels=["A1", "B1", "C1", "D1", "E1"]).astype(str)
        partes.append(df)
    return pd.concat(partes, ignore_index=True)


def test_split_is_out_of_time_without_overlap():
    partes = pd_model.dividir_por_cosecha(_dataset_sintetico())
    anios = {k: set(pd.to_datetime(v["issue_d"]).dt.year) for k, v in partes.items()}
    assert anios == {"entrenamiento": {2013, 2014, 2015}, "calibracion": {2016}, "prueba": {2017}}


def test_pd_model_trains_calibrates_and_handles_unseen_categories():
    partes = pd_model.dividir_por_cosecha(_dataset_sintetico())
    modelo = pd_model.entrenar(partes["entrenamiento"], partes["calibracion"],
                               max_iter=40, min_samples_leaf=20, early_stopping=False)

    # La prueba trae un purpose que no existía al entrenar.
    m = pd_model.evaluar(modelo, partes["prueba"])

    assert m["auc"] > 0.75
    assert m["gini"] == pytest.approx(2 * m["auc"] - 1)
    assert 0 < m["ks"] <= 1
    assert abs(m["pd_media"] - m["tasa_default"]) < 0.05
    # La isotónica da PD escalonadas: con empates, qcut puede fusionar deciles.
    grupos = m["calibracion_deciles"].values()
    assert 5 <= len(grupos) <= 10
    assert sum(g["prestamos"] for g in grupos) == m["n"]


def test_ks_matches_definition():
    y = np.array([0, 0, 1, 1])
    assert pd_model.ks(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert pd_model.ks(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.0
