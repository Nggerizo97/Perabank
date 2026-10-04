"""Fixtures compartidas: un silver sintético mínimo que reemplaza la lectura de Parquet.

Los builders de gold leen silver vía load_silver(name). Aquí se intercepta esa función
en los módulos que la importan, de modo que dimensiones y hechos se construyen sobre
los mismos DataFrames en memoria sin tocar data/ ni el datalake.
"""
import numpy as np
import pandas as pd
import pytest

N_MARKETING = 40
N_BANK_TX = 30
N_PAYSIM = 25


def _bank_marketing(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "age": rng.integers(18, 80, N_MARKETING),
        "job": rng.choice(["admin.", "technician", None], N_MARKETING),
        "marital": rng.choice(["married", "single"], N_MARKETING),
        "education": rng.choice(["primary", "tertiary", None], N_MARKETING),
        "default": rng.random(N_MARKETING) < 0.2,
        "balance": rng.normal(1500, 800, N_MARKETING),
        "housing": rng.random(N_MARKETING) < 0.5,
        "loan": rng.random(N_MARKETING) < 0.3,
        "contact": "cellular",
        "day": rng.integers(1, 31, N_MARKETING),
        "month": "may",
        "duration": rng.integers(10, 900, N_MARKETING),
        "campaign": rng.integers(1, 5, N_MARKETING),
        "pdays": -1,
        "previous": 0,
        "poutcome": rng.choice(["success", "failure", None], N_MARKETING),
        "deposit": rng.random(N_MARKETING) < 0.4,
    })


def _bank_transactions(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "TransactionID": [f"T{i}" for i in range(N_BANK_TX)],
        # Varias transacciones por cliente: dim_cliente debe deduplicar.
        "CustomerID": [f"C{i % 10}" for i in range(N_BANK_TX)],
        "CustGender": rng.choice(["M", "F", None], N_BANK_TX),
        "CustLocation": rng.choice(["MUMBAI", None], N_BANK_TX),
        "CustAccountBalance": rng.normal(20000, 5000, N_BANK_TX),
        "TransactionDate": pd.to_datetime("2016-08-01") + pd.to_timedelta(rng.integers(0, 60, N_BANK_TX), unit="D"),
        "TransactionAmount (INR)": rng.uniform(10, 5000, N_BANK_TX),
    })


def _paysim(rng: np.random.Generator) -> pd.DataFrame:
    return pd.DataFrame({
        "step": rng.integers(1, 700, N_PAYSIM),
        "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], N_PAYSIM),
        "amount": rng.uniform(1, 10000, N_PAYSIM),
        "nameOrig": [f"O{i}" for i in range(N_PAYSIM)],
        "oldbalanceOrg": rng.uniform(0, 20000, N_PAYSIM),
        "newbalanceOrig": rng.uniform(0, 20000, N_PAYSIM),
        "nameDest": [f"D{i % 7}" for i in range(N_PAYSIM)],
        "oldbalanceDest": rng.uniform(0, 20000, N_PAYSIM),
        "newbalanceDest": rng.uniform(0, 20000, N_PAYSIM),
        "isFraud": rng.random(N_PAYSIM) < 0.1,
        "isFlaggedFraud": False,
    })


@pytest.fixture
def silver_frames() -> dict:
    rng = np.random.default_rng(42)
    return {
        "bank_marketing": _bank_marketing(rng),
        "bank_transactions": _bank_transactions(rng),
        "paysim": _paysim(rng),
    }


@pytest.fixture
def fake_silver(monkeypatch, silver_frames) -> dict:
    """Redirige load_silver a silver_frames. Fuentes no declaradas llegan vacías,
    igual que cuando el Parquet no existe en una corrida real."""
    def _load(name: str) -> pd.DataFrame:
        return silver_frames.get(name, pd.DataFrame()).reset_index(drop=True)

    monkeypatch.setattr("etl.gold.dimensions.load_silver", _load)
    monkeypatch.setattr("etl.gold.facts.load_silver", _load)
    return silver_frames


@pytest.fixture
def gold_dir(tmp_path, monkeypatch):
    """Warehouse gold vacío y aislado: todo lo que lee o escribe etl.common.warehouse
    durante el test ocurre aquí, nunca en data/gold."""
    path = tmp_path / "gold"
    monkeypatch.setattr("etl.common.warehouse.GOLD_DIR", path)
    return path
