"""Orquestador de la capa Gold de PeraBank (modelo copo de nieve).

Orden de construcción: sub-dimensiones nivel 2 -> dimensiones nivel 1 -> hechos.
Cada tabla pasa por assert_quality (PK única, no nulos, integridad referencial)
antes de persistirse en Parquet, que es el único almacenamiento de gold: DuckDB lo
consulta en sitio (etl/common/warehouse.py).
"""
import pandas as pd

from etl.common.config import GOLD_DIR
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


    logger.info("--- CAPA GOLD COMPLETADA EXITOSAMENTE ---")


if __name__ == "__main__":
    main()
