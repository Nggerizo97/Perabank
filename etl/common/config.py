"""Rutas y configuración compartida entre bronze, silver y gold."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # PeraBank/Perabank
PROJECT_ROOT = REPO_ROOT.parent                    # PeraBank

DATALAKE_DIR = PROJECT_ROOT / "datalake"

DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

RAW_SOURCES = {
    "paysim": DATALAKE_DIR / "PS_20174392719_1491204439457_log.csv",
    "bank_marketing": DATALAKE_DIR / "bank.csv",
    "bank_transactions": DATALAKE_DIR / "bank_transactions.csv",
    "creditcard": DATALAKE_DIR / "creditcard.csv",
}

# Histórico LendingClub 2007-2020Q3 (~2.9M préstamos, 1.77 GB). Va aparte de
# RAW_SOURCES porque no cabe en pandas: se procesa con DuckDB en las tres capas.
LENDINGCLUB_RAW = DATALAKE_DIR / "Loan_status_2007-2020Q3.gzip"

