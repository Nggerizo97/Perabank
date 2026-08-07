"""Dimensiones conformadas y sub-dimensiones normalizadas (modelo copo de nieve).

Nivel 1 (conformadas): dim_fecha, dim_moneda, dim_cliente.
Nivel 2 (normalizadas): dim_entidad_financiera, dim_proveedor_estatal, dim_tipo_credito.

Todo es vectorizado: sin .iterrows() ni .apply() fila por fila.
"""
from datetime import datetime, timezone

import pandas as pd

from etl.common.logging_utils import get_logger
from etl.gold.sources import (
    DIM_FECHA_FIN,
    DIM_FECHA_INICIO,
    SK_FECHA_DESCONOCIDA,
    UNKNOWN_LABEL,
    generate_surrogate_key,
    load_silver,
    map_surrogate_keys,
    unknown_key,
)
from schemas.gold_schemas import (
    GOLD_DIM_CLIENTE_CONTRACT,
    GOLD_DIM_ENTIDAD_FINANCIERA_CONTRACT,
    GOLD_DIM_FECHA_CONTRACT,
    GOLD_DIM_MONEDA_CONTRACT,
    GOLD_DIM_PROVEEDOR_ESTATAL_CONTRACT,
    GOLD_DIM_TIPO_CREDITO_CONTRACT,
)

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------
# Nivel 2: sub-dimensiones normalizadas
# --------------------------------------------------------------------------

def build_dim_entidad_financiera() -> pd.DataFrame:
    """Normaliza las entidades vigiladas por Superfinanciera, referenciadas tanto por
    dim_cliente como por los hechos de tasas."""
    logger.info("Construyendo dim_entidad_financiera...")
    now_iso = _now_iso()
    parts = []

    df_activas = load_silver("enrichment_tasas_activas")
    if not df_activas.empty:
        sub = df_activas[["codigo_entidad", "nombre_entidad", "tipo_entidad", "nombre_tipo_entidad"]].copy()
        sub = sub.dropna(subset=["codigo_entidad"]).drop_duplicates(subset=["codigo_entidad"])
        parts.append(pd.DataFrame({
            "sk_entidad": map_surrogate_keys(sub["codigo_entidad"].astype(str), "superfinanciera"),
            "codigo_entidad": sub["codigo_entidad"].astype(str),
            "nombre_entidad": sub["nombre_entidad"].astype(str),
            "tipo_entidad": sub["tipo_entidad"].astype(str),
            "nombre_tipo_entidad": sub["nombre_tipo_entidad"].astype(str),
            "source_system": "enrichment_tasas_activas",
            "ingested_at": now_iso,
        }))

    df_captacion = load_silver("enrichment_tasas_captacion")
    if not df_captacion.empty:
        sub = df_captacion[["codigoentidad", "nombreentidad", "tipoentidad"]].copy()
        sub = sub.dropna(subset=["codigoentidad"]).drop_duplicates(subset=["codigoentidad"])
        parts.append(pd.DataFrame({
            "sk_entidad": map_surrogate_keys(sub["codigoentidad"].astype(str), "superfinanciera"),
            "codigo_entidad": sub["codigoentidad"].astype(str),
            "nombre_entidad": sub["nombreentidad"].astype(str),
            "tipo_entidad": sub["tipoentidad"].astype(str),
            "nombre_tipo_entidad": UNKNOWN_LABEL,
            "source_system": "enrichment_tasas_captacion",
            "ingested_at": now_iso,
        }))

    # Miembro DESCONOCIDO: destino de las FK cuya fuente no reporta entidad financiera.
    parts.append(pd.DataFrame([{
        "sk_entidad": unknown_key("dim_entidad_financiera"),
        "codigo_entidad": UNKNOWN_LABEL,
        "nombre_entidad": UNKNOWN_LABEL,
        "tipo_entidad": UNKNOWN_LABEL,
        "nombre_tipo_entidad": UNKNOWN_LABEL,
        "source_system": "perabank_etl",
        "ingested_at": now_iso,
    }]))

    df_dim = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sk_entidad"]).reset_index(drop=True)
    GOLD_DIM_ENTIDAD_FINANCIERA_CONTRACT.validate(df_dim)
    logger.info("dim_entidad_financiera finalizada con %s entidades.", len(df_dim))
    return df_dim


def build_dim_proveedor_estatal() -> pd.DataFrame:
    """Normaliza los proveedores/contratistas del Estado desde SECOP II."""
    logger.info("Construyendo dim_proveedor_estatal...")
    now_iso = _now_iso()
    df_secop = load_silver("enrichment_secop_contratos")

    parts = []
    if not df_secop.empty:
        sub = df_secop[["documento_proveedor", "proveedor_adjudicado", "es_pyme"]].copy()
        sub = sub.dropna(subset=["documento_proveedor"]).drop_duplicates(subset=["documento_proveedor"])
        parts.append(pd.DataFrame({
            "sk_proveedor": map_surrogate_keys(sub["documento_proveedor"].astype(str), "secop"),
            "documento_proveedor": sub["documento_proveedor"].astype(str),
            "proveedor_adjudicado": sub["proveedor_adjudicado"].astype(str),
            "es_pyme": sub["es_pyme"].astype(str).str.strip().str.lower().eq("si"),
            "source_system": "enrichment_secop_contratos",
            "ingested_at": now_iso,
        }))

    parts.append(pd.DataFrame([{
        "sk_proveedor": unknown_key("dim_proveedor_estatal"),
        "documento_proveedor": UNKNOWN_LABEL,
        "proveedor_adjudicado": UNKNOWN_LABEL,
        "es_pyme": False,
        "source_system": "perabank_etl",
        "ingested_at": now_iso,
    }]))

    df_dim = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sk_proveedor"]).reset_index(drop=True)
    GOLD_DIM_PROVEEDOR_ESTATAL_CONTRACT.validate(df_dim)
    logger.info("dim_proveedor_estatal finalizada con %s proveedores.", len(df_dim))
    return df_dim


def build_dim_tipo_credito() -> pd.DataFrame:
    """Normaliza modalidades de colocación (crédito) e instrumentos de captación (fondeo)."""
    logger.info("Construyendo dim_tipo_credito...")
    now_iso = _now_iso()
    parts = []

    df_activas = load_silver("enrichment_tasas_activas")
    if not df_activas.empty:
        sub = df_activas[["tipo_de_cr_dito", "producto_de_cr_dito"]].copy()
        sub = sub.dropna(subset=["tipo_de_cr_dito"]).drop_duplicates()
        nombre = sub["tipo_de_cr_dito"].astype(str)
        producto = sub["producto_de_cr_dito"].astype(str)
        parts.append(pd.DataFrame({
            "sk_tipo_credito": map_surrogate_keys("COLOCACION::" + nombre + "::" + producto, "superfinanciera"),
            "categoria": "COLOCACION",
            "nombre_tipo": nombre,
            "producto": producto,
            "source_system": "enrichment_tasas_activas",
            "ingested_at": now_iso,
        }))

    df_captacion = load_silver("enrichment_tasas_captacion")
    if not df_captacion.empty:
        sub = df_captacion[["descripcion", "nombre_unidad_de_captura"]].copy()
        sub = sub.dropna(subset=["descripcion"]).drop_duplicates()
        nombre = sub["descripcion"].astype(str)
        producto = sub["nombre_unidad_de_captura"].astype(str)
        parts.append(pd.DataFrame({
            "sk_tipo_credito": map_surrogate_keys("CAPTACION::" + nombre + "::" + producto, "superfinanciera"),
            "categoria": "CAPTACION",
            "nombre_tipo": nombre,
            "producto": producto,
            "source_system": "enrichment_tasas_captacion",
            "ingested_at": now_iso,
        }))

    # Modalidad usada por la campaña de depósito a término (bank_marketing).
    parts.append(pd.DataFrame([{
        "sk_tipo_credito": generate_surrogate_key("perabank_etl", "CAPTACION::DEPOSITO_A_TERMINO"),
        "categoria": "CAPTACION",
        "nombre_tipo": "DEPOSITO_A_TERMINO",
        "producto": "Campaña de depósito a término",
        "source_system": "perabank_etl",
        "ingested_at": now_iso,
    }, {
        "sk_tipo_credito": unknown_key("dim_tipo_credito"),
        "categoria": UNKNOWN_LABEL,
        "nombre_tipo": UNKNOWN_LABEL,
        "producto": UNKNOWN_LABEL,
        "source_system": "perabank_etl",
        "ingested_at": now_iso,
    }]))

    df_dim = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sk_tipo_credito"]).reset_index(drop=True)
    GOLD_DIM_TIPO_CREDITO_CONTRACT.validate(df_dim)
    logger.info("dim_tipo_credito finalizada con %s modalidades.", len(df_dim))
    return df_dim


# --------------------------------------------------------------------------
# Nivel 1: dimensiones conformadas
# --------------------------------------------------------------------------

def build_dim_cliente() -> pd.DataFrame:
    """Construye dim_cliente unificando perfiles de clientes de fuentes sin asumir claves compartidas.

    Ninguna de las fuentes transaccionales reporta la entidad financiera del cliente,
    así que sk_entidad apunta al miembro DESCONOCIDO en vez de asignar un banco
    inventado. La FK queda disponible para cuando exista una fuente que sí la traiga.
    """
    logger.info("Construyendo dim_cliente...")
    now_iso = _now_iso()
    sk_entidad_desconocida = unknown_key("dim_entidad_financiera")
    parts = []

    df_bt = load_silver("bank_transactions")
    if not df_bt.empty:
        df_cust = df_bt[["CustomerID", "CustGender", "CustLocation"]].drop_duplicates(subset=["CustomerID"])
        natural_id = df_cust["CustomerID"].astype(str)
        parts.append(pd.DataFrame({
            "sk_cliente": map_surrogate_keys(natural_id, "bank_transactions"),
            "sk_entidad": sk_entidad_desconocida,
            "source_system": "bank_transactions",
            "natural_id": natural_id,
            "genero": df_cust["CustGender"].fillna(UNKNOWN_LABEL),
            "ubicacion": df_cust["CustLocation"].astype(str).where(df_cust["CustLocation"].notnull(), UNKNOWN_LABEL),
            "edad": pd.NA,
            "ocupacion": pd.NA,
            "estado_civil": pd.NA,
            "nivel_educativo": pd.NA,
            "ingested_at": now_iso,
        }))

    df_bm = load_silver("bank_marketing")
    if not df_bm.empty:
        natural_id = pd.Series("bm_" + df_bm.index.astype(str), index=df_bm.index)
        parts.append(pd.DataFrame({
            "sk_cliente": map_surrogate_keys(natural_id, "bank_marketing"),
            "sk_entidad": sk_entidad_desconocida,
            "source_system": "bank_marketing",
            "natural_id": natural_id,
            "genero": UNKNOWN_LABEL,
            "ubicacion": UNKNOWN_LABEL,
            "edad": df_bm["age"].astype("int64"),
            "ocupacion": df_bm["job"].fillna(UNKNOWN_LABEL),
            "estado_civil": df_bm["marital"].fillna(UNKNOWN_LABEL),
            "nivel_educativo": df_bm["education"].fillna(UNKNOWN_LABEL),
            "ingested_at": now_iso,
        }))

    df_ps = load_silver("paysim")
    if not df_ps.empty:
        all_ps_ids = pd.concat([df_ps["nameOrig"], df_ps["nameDest"]]).drop_duplicates().astype(str)
        parts.append(pd.DataFrame({
            "sk_cliente": map_surrogate_keys(all_ps_ids, "paysim"),
            "sk_entidad": sk_entidad_desconocida,
            "source_system": "paysim",
            "natural_id": all_ps_ids,
            "genero": UNKNOWN_LABEL,
            "ubicacion": UNKNOWN_LABEL,
            "edad": pd.NA,
            "ocupacion": pd.NA,
            "estado_civil": pd.NA,
            "nivel_educativo": pd.NA,
            "ingested_at": now_iso,
        }))

    df_dim = pd.concat(parts, ignore_index=True).drop_duplicates(subset=["sk_cliente"]).reset_index(drop=True)
    GOLD_DIM_CLIENTE_CONTRACT.validate(df_dim)
    logger.info("dim_cliente finalizada con %s registros únicos.", len(df_dim))
    return df_dim


def build_dim_fecha(start_date_str: str = DIM_FECHA_INICIO, end_date_str: str = DIM_FECHA_FIN) -> pd.DataFrame:
    """Genera la dimensión conformada de fechas.

    El rango arranca en 1991 porque la TRM oficial se publica desde 1991-12-02, y
    llega a 2056 porque SECOP II reporta contratos con fin hasta 2055. Cubrir todo
    el rango es lo que permite validar integridad referencial real sobre sk_fecha.
    """
    logger.info("Construyendo dim_fecha (%s a %s)...", start_date_str, end_date_str)
    fechas = pd.date_range(start=start_date_str, end=end_date_str, freq="D")

    df_dim = pd.DataFrame({
        "sk_fecha": fechas.strftime("%Y%m%d").astype("int64"),
        "fecha": fechas.strftime("%Y-%m-%d"),
        "anio": fechas.year.astype("int64"),
        "mes": fechas.month.astype("int64"),
        "dia": fechas.day.astype("int64"),
        "trimestre": fechas.quarter.astype("int64"),
        "dia_semana": fechas.day_name(),
        "es_fin_de_semana": fechas.dayofweek >= 5,
    })

    # Miembro DESCONOCIDO para hechos cuya fuente no reporta fecha utilizable.
    desconocida = pd.DataFrame([{
        "sk_fecha": SK_FECHA_DESCONOCIDA,
        "fecha": UNKNOWN_LABEL,
        "anio": -1,
        "mes": -1,
        "dia": -1,
        "trimestre": -1,
        "dia_semana": UNKNOWN_LABEL,
        "es_fin_de_semana": False,
    }])

    df_dim = pd.concat([desconocida, df_dim], ignore_index=True)
    GOLD_DIM_FECHA_CONTRACT.validate(df_dim)
    logger.info("dim_fecha finalizada con %s registros.", len(df_dim))
    return df_dim


def build_dim_moneda() -> pd.DataFrame:
    """Construye la dimensión conformada de divisas (dim_moneda)."""
    logger.info("Construyendo dim_moneda...")
    currencies = [
        {"sk_moneda": "USD", "codigo_iso": "USD", "nombre_moneda": "US Dollar", "simbolo": "$"},
        {"sk_moneda": "EUR", "codigo_iso": "EUR", "nombre_moneda": "Euro", "simbolo": "€"},
        {"sk_moneda": "INR", "codigo_iso": "INR", "nombre_moneda": "Indian Rupee", "simbolo": "₹"},
        {"sk_moneda": "COP", "codigo_iso": "COP", "nombre_moneda": "Colombian Peso", "simbolo": "$"},
    ]
    df_dim = pd.DataFrame(currencies)
    GOLD_DIM_MONEDA_CONTRACT.validate(df_dim)
    logger.info("dim_moneda finalizada con %s registros.", len(df_dim))
    return df_dim
