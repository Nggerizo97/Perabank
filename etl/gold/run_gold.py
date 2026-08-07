"""Orquestador de la capa Gold de PeraBank (modelo copo de nieve).

Orden de construcción: sub-dimensiones nivel 2 -> dimensiones nivel 1 -> hechos.
Cada tabla pasa por assert_quality (PK única, no nulos, integridad referencial)
antes de persistirse en Parquet y SQLite, con índices B-Tree sobre PK y FK.
"""
import sqlite3

import pandas as pd

from etl.common.config import GOLD_DIR, SQLITE_DB_PATH
from etl.common.logging_utils import get_logger
from etl.common.quality_checks import assert_quality
from etl.gold.dimensions import (
    build_dim_cliente,
    build_dim_entidad_financiera,
    build_dim_fecha,
    build_dim_moneda,
    build_dim_proveedor_estatal,
    build_dim_tipo_credito,
)
from etl.gold.facts import (
    build_fact_campana_marcado,
    build_fact_contrato_estatal,
    build_fact_fraude_tarjeta,
    build_fact_tasas_mercado,
    build_fact_transaccion,
)

logger = get_logger(__name__)

# Índices B-Tree por tabla: PK primero, luego cada FK usada en joins.
SQLITE_INDEXES = {
    "dim_entidad_financiera": ["sk_entidad"],
    "dim_proveedor_estatal": ["sk_proveedor"],
    "dim_tipo_credito": ["sk_tipo_credito"],
    "dim_cliente": ["sk_cliente", "sk_entidad"],
    "dim_fecha": ["sk_fecha"],
    "dim_moneda": ["sk_moneda"],
    "fact_transaccion": ["id_transaccion", "sk_cliente", "sk_fecha", "sk_moneda"],
    "fact_campana_marcado": ["id_campana_contacto", "sk_cliente", "sk_fecha", "sk_tipo_credito"],
    "fact_fraude_tarjeta": ["id_evento_tarjeta"],
    "fact_contrato_estatal": ["id_contrato", "sk_proveedor", "sk_fecha"],
    "fact_tasas_mercado": ["id_observacion_tasa", "sk_fecha", "sk_entidad", "sk_tipo_credito"],
}


def export_to_sqlite(gold_tables: dict) -> None:
    """Exporta las tablas Gold a SQLite y crea los índices sobre PK y FK."""
    logger.info("Exportando tablas Gold a SQLite (%s)...", SQLITE_DB_PATH)
    SQLITE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(SQLITE_DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        for table_name, df in gold_tables.items():
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            logger.info("Tabla SQLite '%s' actualizada (%s filas).", table_name, len(df))

        total_indexes = 0
        for table_name, columns in SQLITE_INDEXES.items():
            if table_name not in gold_tables:
                continue
            for column in columns:
                index_name = f"idx_{table_name}_{column}"
                conn.execute(f'CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({column});')
                total_indexes += 1
        conn.commit()
        logger.info("Índices B-Tree creados/verificados: %s.", total_indexes)

    # if_exists="replace" deja páginas libres en el archivo: sin VACUUM el .db sigue
    # ocupando el tamaño de la corrida más grande que haya existido.
    with sqlite3.connect(SQLITE_DB_PATH, isolation_level=None) as conn:
        conn.execute("VACUUM;")
    logger.info("SQLite compactado (VACUUM): %.1f MB.", SQLITE_DB_PATH.stat().st_size / 1e6)


def _persist(df: pd.DataFrame, name: str) -> None:
    df.to_parquet(GOLD_DIR / f"{name}.parquet", index=False)


def main():
    logger.info("--- INICIANDO CONSTRUCCIÓN DE LA CAPA GOLD (COPO DE NIEVE) ---")
    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Sub-dimensiones nivel 2 (normalizadas)
    dim_entidad = build_dim_entidad_financiera()
    assert_quality(dim_entidad, "dim_entidad_financiera", "gold", primary_keys=["sk_entidad"])
    _persist(dim_entidad, "dim_entidad_financiera")

    dim_proveedor = build_dim_proveedor_estatal()
    assert_quality(dim_proveedor, "dim_proveedor_estatal", "gold", primary_keys=["sk_proveedor"])
    _persist(dim_proveedor, "dim_proveedor_estatal")

    dim_tipo_credito = build_dim_tipo_credito()
    assert_quality(dim_tipo_credito, "dim_tipo_credito", "gold", primary_keys=["sk_tipo_credito"])
    _persist(dim_tipo_credito, "dim_tipo_credito")

    # 2. Dimensiones nivel 1 (conformadas)
    dim_cliente = build_dim_cliente()
    assert_quality(
        dim_cliente, "dim_cliente", "gold",
        primary_keys=["sk_cliente"],
        ref_checks={"sk_entidad": dim_entidad["sk_entidad"]},
    )
    _persist(dim_cliente, "dim_cliente")

    dim_fecha = build_dim_fecha()
    assert_quality(dim_fecha, "dim_fecha", "gold", primary_keys=["sk_fecha"])
    _persist(dim_fecha, "dim_fecha")

    dim_moneda = build_dim_moneda()
    assert_quality(dim_moneda, "dim_moneda", "gold", primary_keys=["sk_moneda"])
    _persist(dim_moneda, "dim_moneda")

    # 3. Hechos
    fact_transaccion = build_fact_transaccion()
    assert_quality(
        fact_transaccion, "fact_transaccion", "gold",
        primary_keys=["id_transaccion"],
        ref_checks={
            "sk_cliente": dim_cliente["sk_cliente"],
            "sk_fecha": dim_fecha["sk_fecha"],
            "sk_moneda": dim_moneda["sk_moneda"],
        },
    )
    _persist(fact_transaccion, "fact_transaccion")

    fact_campana = build_fact_campana_marcado()
    assert_quality(
        fact_campana, "fact_campana_marcado", "gold",
        primary_keys=["id_campana_contacto"],
        ref_checks={
            "sk_cliente": dim_cliente["sk_cliente"],
            "sk_fecha": dim_fecha["sk_fecha"],
            "sk_tipo_credito": dim_tipo_credito["sk_tipo_credito"],
        },
    )
    _persist(fact_campana, "fact_campana_marcado")

    fact_fraude = build_fact_fraude_tarjeta()
    assert_quality(fact_fraude, "fact_fraude_tarjeta", "gold", primary_keys=["id_evento_tarjeta"])
    _persist(fact_fraude, "fact_fraude_tarjeta")

    fact_contrato = build_fact_contrato_estatal(dim_proveedor)
    assert_quality(
        fact_contrato, "fact_contrato_estatal", "gold",
        primary_keys=["id_contrato"],
        ref_checks={
            "sk_proveedor": dim_proveedor["sk_proveedor"],
            "sk_fecha": dim_fecha["sk_fecha"],
        },
    )
    _persist(fact_contrato, "fact_contrato_estatal")

    fact_tasas = build_fact_tasas_mercado(dim_entidad, dim_tipo_credito)
    assert_quality(
        fact_tasas, "fact_tasas_mercado", "gold",
        primary_keys=["id_observacion_tasa"],
        ref_checks={
            "sk_fecha": dim_fecha["sk_fecha"],
            "sk_entidad": dim_entidad["sk_entidad"],
            "sk_tipo_credito": dim_tipo_credito["sk_tipo_credito"],
        },
    )
    _persist(fact_tasas, "fact_tasas_mercado")

    # 4. Persistencia en SQLite con índices
    export_to_sqlite({
        "dim_entidad_financiera": dim_entidad,
        "dim_proveedor_estatal": dim_proveedor,
        "dim_tipo_credito": dim_tipo_credito,
        "dim_cliente": dim_cliente,
        "dim_fecha": dim_fecha,
        "dim_moneda": dim_moneda,
        "fact_transaccion": fact_transaccion,
        "fact_campana_marcado": fact_campana,
        "fact_fraude_tarjeta": fact_fraude,
        "fact_contrato_estatal": fact_contrato,
        "fact_tasas_mercado": fact_tasas,
    })

    logger.info("--- CAPA GOLD COMPLETADA EXITOSAMENTE ---")


if __name__ == "__main__":
    main()
