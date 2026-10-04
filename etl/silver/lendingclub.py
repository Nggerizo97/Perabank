"""Silver de LendingClub: tipado y limpieza en SQL (DuckDB), sin pasar por pandas.

Una fila por préstamo. Se descartan registros sin id numérico, sin fecha de
originación o sin estado (las exportaciones de LendingClub pueden traer filas de
resumen; en el archivo 2007-2020Q3 solo hay una fila así, sin loan_status). No se
decide aquí qué es "default": eso es regla de negocio y vive en gold.
"""
from pathlib import Path

import duckdb

from etl.common.config import BRONZE_DIR, SILVER_DIR
from etl.common.logging_utils import get_logger
from schemas.silver_schemas import SILVER_LENDINGCLUB_CONTRACT

logger = get_logger(__name__)

SOURCE_NAME = "lendingclub"

# Campos que en el CSV vienen como texto con formato ("Dec-2011", " 36 months",
# " 10.65%", "10+ years") se convierten a su tipo; el resto se conserva tal cual.
SILVER_SQL = """
SELECT
    TRY_CAST(id AS BIGINT)                                         AS id,
    TRY_STRPTIME(issue_d, '%b-%Y')::DATE                           AS issue_d,
    TRY_CAST(regexp_extract(term, '(\\d+)', 1) AS INTEGER)         AS term,
    TRY_CAST(replace(trim(int_rate), '%', '') AS DOUBLE)           AS int_rate,
    installment, grade, sub_grade,
    loan_amnt, funded_amnt,
    CASE trim(emp_length)
        WHEN '< 1 year'  THEN 0
        WHEN '10+ years' THEN 10
        ELSE TRY_CAST(regexp_extract(emp_length, '(\\d+)', 1) AS INTEGER)
    END                                                            AS emp_length,
    home_ownership, annual_inc, verification_status, purpose, addr_state, zip_code,
    dti, delinq_2yrs,
    TRY_STRPTIME(earliest_cr_line, '%b-%Y')::DATE                  AS earliest_cr_line,
    fico_range_low, fico_range_high, inq_last_6mths,
    mths_since_last_delinq, mths_since_last_record,
    open_acc, pub_rec, revol_bal,
    TRY_CAST(replace(trim(revol_util), '%', '') AS DOUBLE)         AS revol_util,
    total_acc, initial_list_status, application_type,
    mort_acc, pub_rec_bankruptcies, acc_open_past_24mths, bc_util,
    num_actv_rev_tl, tot_cur_bal, total_rev_hi_lim,
    loan_status,
    total_pymnt, total_rec_prncp, recoveries, collection_recovery_fee,
    TRY_STRPTIME(last_pymnt_d, '%b-%Y')::DATE                      AS last_pymnt_d,
    source_system, ingested_at
FROM bronze
WHERE TRY_CAST(id AS BIGINT) IS NOT NULL
  AND TRY_STRPTIME(issue_d, '%b-%Y') IS NOT NULL
  AND loan_status IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY TRY_CAST(id AS BIGINT) ORDER BY ingested_at DESC) = 1
ORDER BY id
"""


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def transform(bronze_dir: Path = BRONZE_DIR, silver_dir: Path = SILVER_DIR) -> Path:
    silver_dir.mkdir(parents=True, exist_ok=True)
    in_path = bronze_dir / f"{SOURCE_NAME}.parquet"
    out_path = silver_dir / f"{SOURCE_NAME}.parquet"

    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW bronze AS SELECT * FROM read_parquet('{_sql_path(in_path)}')")
        con.execute(f"CREATE VIEW silver AS {SILVER_SQL}")
        SILVER_LENDINGCLUB_CONTRACT.validate(con.execute("SELECT * FROM silver LIMIT 0").df())
        con.execute(f"COPY silver TO '{_sql_path(out_path)}' (FORMAT parquet, COMPRESSION zstd)")
        entrada = con.execute("SELECT COUNT(*) FROM bronze").fetchone()[0]
        salida = con.execute(f"SELECT COUNT(*) FROM read_parquet('{_sql_path(out_path)}')").fetchone()[0]

    logger.info("silver/%s: %s -> %s filas (%s descartadas) -> %s",
                SOURCE_NAME, entrada, salida, entrada - salida, out_path)
    return out_path


def main():
    transform()


if __name__ == "__main__":
    main()
