"""fact_prestamo_minorista: un préstamo de LendingClub por fila, con su desenlace.

Las columnas que vienen de LendingClub conservan su nombre original (documentadas en
datalake/LCDataDictionary.xlsx); las derivadas aquí van en español.

Dos reglas de negocio viven en esta tabla y en ningún otro lugar:

- es_default: TRUE si el préstamo terminó castigado (Charged Off) o en Default; FALSE
  si se pagó por completo; NULL si al corte aún no tiene desenlace (Current, Late,
  In Grace Period, Issued). Un préstamo sin desenlace no es un "buen pagador".
- madurado: el plazo completo más MESES_GRACIA transcurrió antes del corte. En
  cosechas recientes solo se conocen los desenlaces tempranos (defaults rápidos y
  prepagos), así que modelar con ellas sesga la tasa de default; el modelo de PD usa
  solo préstamos madurados. La gracia existe porque el cierre tarda: en el dataset,
  ~30% de los préstamos vencidos hace 0-3 meses sigue figurando como Current, y la
  fracción sin desenlace cae por debajo del 1% a partir del mes 6.
"""
from pathlib import Path

import duckdb

from etl.common.config import GOLD_DIR, SILVER_DIR
from etl.common.logging_utils import get_logger
from schemas.gold_schemas import GOLD_FACT_PRESTAMO_MINORISTA_CONTRACT

logger = get_logger(__name__)

TABLE_NAME = "fact_prestamo_minorista"

MESES_GRACIA = 6

ESTADOS_DEFAULT = ("Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off")
ESTADOS_PAGADO = ("Fully Paid", "Does not meet the credit policy. Status:Fully Paid")

# Conocidas solo después de originar el préstamo. Nunca pueden ser features de un
# modelo de originación: describen cómo terminó el crédito, no al solicitante.
COLUMNAS_POST_ORIGINACION = (
    "estado_final", "es_default", "madurado", "fecha_corte", "last_pymnt_d",
    "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
)


def _sql_list(values: tuple) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


GOLD_SQL = f"""
WITH corte AS (SELECT MAX(issue_d) AS fecha_corte FROM silver)
SELECT
    s.id                                                     AS id_prestamo,
    CAST(strftime(s.issue_d, '%Y%m%d') AS BIGINT)            AS sk_fecha,
    s.issue_d, s.term,
    s.loan_amnt, s.funded_amnt, s.int_rate, s.installment, s.grade, s.sub_grade,
    s.emp_length, s.home_ownership, s.annual_inc, s.verification_status, s.purpose,
    s.addr_state, s.zip_code, s.dti, s.delinq_2yrs,
    s.fico_range_low, s.fico_range_high,
    (s.fico_range_low + s.fico_range_high) / 2.0             AS fico_promedio,
    date_diff('month', s.earliest_cr_line, s.issue_d)        AS meses_historial_credito,
    s.inq_last_6mths, s.mths_since_last_delinq, s.mths_since_last_record,
    s.open_acc, s.pub_rec, s.revol_bal, s.revol_util, s.total_acc,
    s.initial_list_status, s.application_type, s.mort_acc, s.pub_rec_bankruptcies,
    s.acc_open_past_24mths, s.bc_util, s.num_actv_rev_tl, s.tot_cur_bal, s.total_rev_hi_lim,
    s.loan_status                                            AS estado_final,
    CASE
        WHEN s.loan_status IN ({_sql_list(ESTADOS_DEFAULT)}) THEN TRUE
        WHEN s.loan_status IN ({_sql_list(ESTADOS_PAGADO)})  THEN FALSE
    END                                                      AS es_default,
    s.issue_d + to_months(s.term + {MESES_GRACIA}) <= c.fecha_corte AS madurado,
    c.fecha_corte,
    s.last_pymnt_d, s.total_pymnt, s.total_rec_prncp, s.recoveries, s.collection_recovery_fee
FROM silver s CROSS JOIN corte c
ORDER BY s.id
"""


def _sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def _assert_quality(con: duckdb.DuckDBPyConnection, gold_dir: Path) -> None:
    """Mismas garantías que assert_quality (PK única, no nulos, FK), pero en SQL:
    la tabla tiene millones de filas y no se materializa en pandas para validarla."""
    dup = con.execute(f"SELECT COUNT(*) - COUNT(DISTINCT id_prestamo) FROM {TABLE_NAME}").fetchone()[0]
    if dup:
        raise ValueError(f"[gold/{TABLE_NAME}] ERROR DQ: {dup} filas duplicadas en id_prestamo.")

    for col in ("id_prestamo", "sk_fecha", "issue_d", "term", "estado_final", "madurado"):
        nulos = con.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE {col} IS NULL").fetchone()[0]
        if nulos:
            raise ValueError(f"[gold/{TABLE_NAME}] ERROR DQ: columna obligatoria '{col}' con {nulos} nulos.")

    dim_fecha = gold_dir / "dim_fecha.parquet"
    if not dim_fecha.exists():
        raise ValueError(f"[gold/{TABLE_NAME}] falta dim_fecha en {gold_dir}: constrúyela antes de este hecho.")
    huerfanos = con.execute(f"""
        SELECT COUNT(*) FROM {TABLE_NAME} f
        ANTI JOIN read_parquet('{_sql_path(dim_fecha)}') d ON d.sk_fecha = f.sk_fecha
    """).fetchone()[0]
    if huerfanos:
        raise ValueError(f"[gold/{TABLE_NAME}] ERROR DQ: {huerfanos} filas violan la FK sk_fecha -> dim_fecha.")


def build_fact_prestamo_minorista(silver_dir: Path = SILVER_DIR, gold_dir: Path = GOLD_DIR) -> Path:
    gold_dir.mkdir(parents=True, exist_ok=True)
    in_path = silver_dir / "lendingclub.parquet"
    out_path = gold_dir / f"{TABLE_NAME}.parquet"

    with duckdb.connect() as con:
        con.execute(f"CREATE VIEW silver AS SELECT * FROM read_parquet('{_sql_path(in_path)}')")
        con.execute(f"CREATE TABLE {TABLE_NAME} AS {GOLD_SQL}")
        GOLD_FACT_PRESTAMO_MINORISTA_CONTRACT.validate(con.execute(f"SELECT * FROM {TABLE_NAME} LIMIT 0").df())
        _assert_quality(con, gold_dir)
        con.execute(f"COPY {TABLE_NAME} TO '{_sql_path(out_path)}' (FORMAT parquet, COMPRESSION zstd)")
        resumen = con.execute(f"""
            SELECT COUNT(*), COUNT(es_default), COUNT(*) FILTER (WHERE madurado AND es_default IS NOT NULL),
                   AVG(es_default::INT) FILTER (WHERE madurado)
            FROM {TABLE_NAME}
        """).fetchone()

    logger.info("gold/%s: %s préstamos, %s con desenlace, %s madurados con desenlace "
                "(tasa de default madurada %.2f%%) -> %s",
                TABLE_NAME, resumen[0], resumen[1], resumen[2], 100 * (resumen[3] or 0), out_path)
    return out_path
