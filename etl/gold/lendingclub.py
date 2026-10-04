"""fact_prestamo_minorista: un préstamo de LendingClub por fila, con su desenlace.

Las columnas que vienen de LendingClub conservan su nombre original (documentadas en
datalake/LCDataDictionary.xlsx); las derivadas aquí van en español.

Las reglas de negocio sobre el desenlace viven en esta tabla y en ningún otro lugar:

- es_default: TRUE si el préstamo terminó castigado (Charged Off) o en Default; FALSE
  si se pagó por completo; NULL si al corte aún no tiene desenlace (Current, Late,
  In Grace Period, Issued). Un préstamo sin desenlace no es un "buen pagador".
- madurado: el plazo completo más MESES_GRACIA transcurrió antes del corte. En
  cosechas recientes solo se conocen los desenlaces tempranos (defaults rápidos y
  prepagos), así que modelar con ellas sesga la tasa de default; el modelo de PD usa
  solo préstamos madurados. La gracia existe porque el cierre tarda: en el dataset,
  ~30% de los préstamos vencidos hace 0-3 meses sigue figurando como Current, y la
  fracción sin desenlace cae por debajo del 1% a partir del mes 6.
- Pérdida realizada, solo para préstamos castigados (Charged Off). Los que están en
  "Default" sin castigar aún no registran recuperaciones y darían una LGD falsa de 1.
    ead_al_default    = capital pendiente al castigo: funded_amnt - total_rec_prncp
    recuperacion_neta = recoveries - collection_recovery_fee (lo que de verdad volvió)
    lgd_realizada     = 1 - recuperacion_neta / ead_al_default, acotada a [0, 1]
    perdida_realizada = ead_al_default - recuperacion_neta; 0 si se pagó completo
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
ESTADOS_CASTIGO = ("Charged Off", "Does not meet the credit policy. Status:Charged Off")

# Conocidas solo después de originar el préstamo. Nunca pueden ser features de un
# modelo de originación: describen cómo terminó el crédito, no al solicitante.
COLUMNAS_POST_ORIGINACION = (
    "estado_final", "es_default", "madurado", "fecha_corte", "last_pymnt_d",
    "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
    "ead_al_default", "recuperacion_neta", "lgd_realizada", "perdida_realizada",
)


def _sql_list(values: tuple) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


GOLD_SQL = f"""
WITH corte AS (SELECT MAX(issue_d) AS fecha_corte FROM silver),
base AS (
    SELECT s.*, c.fecha_corte,
           s.loan_status IN ({_sql_list(ESTADOS_CASTIGO)})       AS castigado,
           GREATEST(s.funded_amnt - s.total_rec_prncp, 0)         AS ead,
           GREATEST(s.recoveries - s.collection_recovery_fee, 0)  AS recuperado
    FROM silver s CROSS JOIN corte c
)
SELECT
    b.id                                                     AS id_prestamo,
    CAST(strftime(b.issue_d, '%Y%m%d') AS BIGINT)            AS sk_fecha,
    b.issue_d, b.term,
    b.loan_amnt, b.funded_amnt, b.int_rate, b.installment, b.grade, b.sub_grade,
    b.emp_length, b.home_ownership, b.annual_inc, b.verification_status, b.purpose,
    b.addr_state, b.zip_code, b.dti, b.delinq_2yrs,
    b.fico_range_low, b.fico_range_high,
    (b.fico_range_low + b.fico_range_high) / 2.0             AS fico_promedio,
    date_diff('month', b.earliest_cr_line, b.issue_d)        AS meses_historial_credito,
    b.inq_last_6mths, b.mths_since_last_delinq, b.mths_since_last_record,
    b.open_acc, b.pub_rec, b.revol_bal, b.revol_util, b.total_acc,
    b.initial_list_status, b.application_type, b.mort_acc, b.pub_rec_bankruptcies,
    b.acc_open_past_24mths, b.bc_util, b.num_actv_rev_tl, b.tot_cur_bal, b.total_rev_hi_lim,
    b.loan_status                                            AS estado_final,
    CASE
        WHEN b.loan_status IN ({_sql_list(ESTADOS_DEFAULT)}) THEN TRUE
        WHEN b.loan_status IN ({_sql_list(ESTADOS_PAGADO)})  THEN FALSE
    END                                                      AS es_default,
    b.issue_d + to_months(b.term + {MESES_GRACIA}) <= b.fecha_corte AS madurado,
    b.fecha_corte,
    b.last_pymnt_d, b.total_pymnt, b.total_rec_prncp, b.recoveries, b.collection_recovery_fee,
    CASE WHEN b.castigado THEN b.ead END                     AS ead_al_default,
    CASE WHEN b.castigado THEN b.recuperado END              AS recuperacion_neta,
    CASE WHEN b.castigado AND b.ead > 0
         THEN LEAST(GREATEST(1 - b.recuperado / b.ead, 0), 1) END AS lgd_realizada,
    CASE
        WHEN b.castigado THEN GREATEST(b.ead - b.recuperado, 0)
        WHEN b.loan_status IN ({_sql_list(ESTADOS_PAGADO)}) THEN 0
    END                                                      AS perdida_realizada
FROM base b
ORDER BY b.id
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
