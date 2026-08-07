"""Aterriza cada CSV crudo del datalake en la capa bronze, sin transformar datos."""
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from etl.common.config import BRONZE_DIR, RAW_SOURCES
from etl.common.logging_utils import get_logger
from schemas.bronze_schemas import (
    BRONZE_BANK_MARKETING_CONTRACT,
    BRONZE_BANK_TRANSACTIONS_CONTRACT,
    BRONZE_CREDITCARD_CONTRACT,
    BRONZE_PAYSIM_CONTRACT,
)

logger = get_logger(__name__)

CONTRACTS = {
    "paysim": BRONZE_PAYSIM_CONTRACT,
    "bank_marketing": BRONZE_BANK_MARKETING_CONTRACT,
    "bank_transactions": BRONZE_BANK_TRANSACTIONS_CONTRACT,
    "creditcard": BRONZE_CREDITCARD_CONTRACT,
}


def ingest(source_name: str) -> Path:
    raw_path = RAW_SOURCES[source_name]
    df = pd.read_csv(raw_path)
    df["source_system"] = source_name
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    CONTRACTS[source_name].validate(df)

    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DIR / f"{source_name}.parquet"
    df.to_parquet(out_path, index=False)

    logger.info("bronze/%s: %s filas -> %s", source_name, len(df), out_path)
    return out_path


def main():
    for source_name in RAW_SOURCES:
        ingest(source_name)


if __name__ == "__main__":
    main()
