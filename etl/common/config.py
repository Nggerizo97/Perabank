"""Rutas y configuración compartida entre bronze, silver y gold."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]   # PeraBank/Perabank
PROJECT_ROOT = REPO_ROOT.parent                    # PeraBank

DATALAKE_DIR = PROJECT_ROOT / "datalake"

DATA_DIR = REPO_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

SQLITE_DB_PATH = DATA_DIR / "perabank.db"

RAW_SOURCES = {
    "paysim": DATALAKE_DIR / "PS_20174392719_1491204439457_log.csv",
    "bank_marketing": DATALAKE_DIR / "bank.csv",
    "bank_transactions": DATALAKE_DIR / "bank_transactions.csv",
    "creditcard": DATALAKE_DIR / "creditcard.csv",
}

