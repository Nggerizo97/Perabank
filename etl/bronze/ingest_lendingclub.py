"""Aterriza el histórico de préstamos de LendingClub (2007-2020Q3) en bronze.

El archivo pesa 1.77 GB y tiene ~2.9M filas: no se carga en pandas. DuckDB lo lee
en streaming y escribe Parquet directamente, con memoria acotada. Pese a la extensión
.gzip, el archivo es un CSV plano sin comprimir.

Bronze no transforma valores: los tipos los infiere DuckDB recorriendo el archivo
completo, y campos como int_rate (" 10.65%") llegan como texto hasta silver.
"""
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from etl.common.config import BRONZE_DIR, LENDINGCLUB_RAW
from etl.common.logging_utils import get_logger
from schemas.bronze_schemas import BRONZE_LENDINGCLUB_CONTRACT

logger = get_logger(__name__)

SOURCE_NAME = "lendingclub"


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def ingest(raw_path: Path = LENDINGCLUB_RAW, out_dir: Path = BRONZE_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{SOURCE_NAME}.parquet"
    ingested_at = datetime.now(timezone.utc).isoformat()

    with duckdb.connect() as con:
        con.execute(f"""
            CREATE VIEW raw AS
            SELECT *, '{SOURCE_NAME}' AS source_system, '{ingested_at}' AS ingested_at
            FROM read_csv('{_sql_path(raw_path)}', compression = 'none', header = true, sample_size = -1)
        """)
        BRONZE_LENDINGCLUB_CONTRACT.validate(con.execute("SELECT * FROM raw LIMIT 0").df())
        con.execute(f"COPY raw TO '{_sql_path(out_path)}' (FORMAT parquet, COMPRESSION zstd)")
        filas = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_sql_path(out_path)}')").fetchone()[0]

    logger.info("bronze/%s: %s filas -> %s", SOURCE_NAME, filas, out_path)
    return out_path


def main():
    ingest()


if __name__ == "__main__":
    main()
