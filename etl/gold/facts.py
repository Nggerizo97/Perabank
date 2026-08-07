"""Tablas de hechos del modelo copo de nieve.

Todas las uniones con datos de mercado usan pd.merge (vectorizado) y las surrogate
keys se resuelven con Series.map sobre valores únicos. Sin .iterrows() ni .apply()
fila por fila en ningún punto.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from etl.common.logging_utils import get_logger
from etl.gold.sources import (
    DIM_FECHA_FIN,
    DIM_FECHA_INICIO,
    SK_FECHA_DESCONOCIDA,
    generate_surrogate_key,
    load_silver,
    map_surrogate_keys,
    unknown_key,
)
from schemas.gold_schemas import (
    GOLD_FACT_CAMPANA_MARCADO_CONTRACT,
    GOLD_FACT_CONTRATO_ESTATAL_CONTRACT,
    GOLD_FACT_FRAUDE_TARJETA_CONTRACT,
    GOLD_FACT_TASAS_MERCADO_CONTRACT,
    GOLD_FACT_TRANSACCION_CONTRACT,
)

logger = get_logger(__name__)

FX_FALLBACK_INR = 83.5
FX_FALLBACK_COP = 4100.0
FX_FALLBACK_EUR = 0.92
IBR_FALLBACK = 6.25


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_sk_fecha(fechas: pd.Series) -> pd.Series:
    """Convierte una serie de fechas a sk_fecha (YYYYMMDD), enrutando al miembro
    DESCONOCIDO todo lo que no sea parseable o caiga fuera del rango de dim_fecha."""
    dt = pd.to_datetime(fechas, errors="coerce")
    dentro_de_rango = dt.between(pd.Timestamp(DIM_FECHA_INICIO), pd.Timestamp(DIM_FECHA_FIN))
    sk = dt.dt.strftime("%Y%m%d")
    sk = pd.to_numeric(sk, errors="coerce").where(dentro_de_rango)
    return sk.fillna(SK_FECHA_DESCONOCIDA).astype("int64")


def _load_market() -> pd.DataFrame:
    df = load_silver("enrichment_market")
    if df.empty:
        return pd.DataFrame(columns=["fecha", "tasa_usd_inr", "tasa_usd_eur", "tasa_usd_cop",
                                     "tasa_ibr_overnight", "volatilidad_fx_30d"])
    return df


def _load_trm() -> pd.DataFrame:
    """TRM oficial de Superfinanciera, normalizada a (fecha, tasa_trm_oficial)."""
    df = load_silver("enrichment_trm_gov")
    if df.empty:
        return pd.DataFrame(columns=["fecha", "tasa_trm_oficial"])
    out = pd.DataFrame({
        "fecha": pd.to_datetime(df["vigenciadesde"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "tasa_trm_oficial": pd.to_numeric(df["valor"], errors="coerce"),
    })
    return out.dropna(subset=["fecha"]).drop_duplicates(subset=["fecha"])


def _load_ecb() -> pd.DataFrame:
    df = load_silver("enrichment_ecb")
    if df.empty:
        return pd.DataFrame(columns=["fecha", "tasa_eur_usd"])
    return df[["fecha", "tasa_eur_usd"]].drop_duplicates(subset=["fecha"])


def build_fact_transaccion() -> pd.DataFrame:
    """Hechos de transacción conformados, homologando moneda con tres fuentes FX reales:
    Yahoo Finance (intradía), TRM oficial Superfinanciera y tasa de referencia BCE."""
    logger.info("Construyendo fact_transaccion...")
    now_iso = _now_iso()
    df_market = _load_market()
    df_trm = _load_trm()
    df_ecb = _load_ecb()
    parts = []

    df_bt = load_silver("bank_transactions")
    if not df_bt.empty:
        fecha = df_bt["TransactionDate"].dt.strftime("%Y-%m-%d").fillna("2023-01-01")
        merged = (
            df_bt.assign(fecha=fecha)
            .merge(df_market[["fecha", "tasa_usd_inr", "tasa_usd_cop"]], on="fecha", how="left")
            .merge(df_trm, on="fecha", how="left")
            .merge(df_ecb, on="fecha", how="left")
        )
        rate_inr = merged["tasa_usd_inr"].fillna(FX_FALLBACK_INR)
        rate_cop = merged["tasa_usd_cop"].fillna(FX_FALLBACK_COP)
        trm = merged["tasa_trm_oficial"]
        monto_inr = merged["TransactionAmount (INR)"].astype("float64")
        monto_usd = (monto_inr / rate_inr.where(rate_inr > 0, FX_FALLBACK_INR)).round(2)

        parts.append(pd.DataFrame({
            "id_transaccion": merged["TransactionID"].astype(str),
            "sk_cliente": map_surrogate_keys(merged["CustomerID"].astype(str), "bank_transactions"),
            "sk_fecha": _to_sk_fecha(merged["fecha"]),
            "sk_moneda": "INR",
            "tipo_transaccion": "BANK_TRANSFER",
            "monto_original": monto_inr,
            "tasa_cambio_usd": rate_inr.astype("float64"),
            "monto_usd": monto_usd,
            "monto_cop": (monto_usd * rate_cop).round(2),
            "monto_cop_trm": (monto_usd * trm).round(2),
            "tasa_trm_oficial": trm,
            "tasa_ecb_eur_usd": merged["tasa_eur_usd"],
            "saldo_previo": merged["CustAccountBalance"].astype("float64"),
            "saldo_nuevo": np.nan,
            "es_fraude": False,
            "source_system": "bank_transactions",
            "ingested_at": now_iso,
        }))

    df_ps = load_silver("paysim")
    if not df_ps.empty:
        # PaySim solo trae 'step' (horas desde el inicio de la simulación), no fecha real.
        # Se ancla a una fecha base declarada para poder unir con dim_fecha y con mercado.
        base_date = datetime(2023, 1, 1)
        tx_dt = base_date + pd.to_timedelta(df_ps["step"].astype(int), unit="h")
        fecha = tx_dt.dt.strftime("%Y-%m-%d")
        merged = (
            df_ps.assign(fecha=fecha)
            .merge(df_market[["fecha", "tasa_usd_cop"]], on="fecha", how="left")
            .merge(df_trm, on="fecha", how="left")
            .merge(df_ecb, on="fecha", how="left")
        )
        rate_cop = merged["tasa_usd_cop"].fillna(FX_FALLBACK_COP)
        trm = merged["tasa_trm_oficial"]
        monto_usd = merged["amount"].astype("float64")

        parts.append(pd.DataFrame({
            "id_transaccion": "ps_" + merged.index.astype(str),
            "sk_cliente": map_surrogate_keys(merged["nameOrig"].astype(str), "paysim"),
            "sk_fecha": _to_sk_fecha(merged["fecha"]),
            "sk_moneda": "USD",
            "tipo_transaccion": merged["type"].astype(str),
            "monto_original": monto_usd,
            "tasa_cambio_usd": 1.0,
            "monto_usd": monto_usd,
            "monto_cop": (monto_usd * rate_cop).round(2),
            "monto_cop_trm": (monto_usd * trm).round(2),
            "tasa_trm_oficial": trm,
            "tasa_ecb_eur_usd": merged["tasa_eur_usd"],
            "saldo_previo": merged["oldbalanceOrg"].astype("float64"),
            "saldo_nuevo": merged["newbalanceOrig"].astype("float64"),
            "es_fraude": merged["isFraud"].astype(bool),
            "source_system": "paysim",
            "ingested_at": now_iso,
        }))

    df_fact = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    GOLD_FACT_TRANSACCION_CONTRACT.validate(df_fact)
    logger.info("fact_transaccion finalizada con %s registros.", len(df_fact))
    return df_fact


def build_fact_campana_marcado() -> pd.DataFrame:
    """Hechos de campaña de captación, con pricing sobre IBR y contexto de tasas
    reales del mercado colombiano (activas y de captación)."""
    logger.info("Construyendo fact_campana_marcado...")
    now_iso = _now_iso()
    df_bm = load_silver("bank_marketing")
    if df_bm.empty:
        return pd.DataFrame()

    df_market = _load_market()

    base_date = datetime(2023, 6, 1)
    tx_dt = base_date + pd.to_timedelta(df_bm["day"].astype(int) % 30, unit="D")
    fecha = tx_dt.dt.strftime("%Y-%m-%d")
    merged = df_bm.assign(fecha=fecha).merge(
        df_market[["fecha", "tasa_usd_eur", "tasa_ibr_overnight"]], on="fecha", how="left"
    )
    rate_eur = merged["tasa_usd_eur"].fillna(FX_FALLBACK_EUR)
    tasa_ibr = merged["tasa_ibr_overnight"].fillna(IBR_FALLBACK)

    # Referencia de mercado: promedio observado en el corte más reciente publicado por
    # Superfinanciera. Es contexto macro del mercado, no una tasa por cliente.
    df_activas = load_silver("enrichment_tasas_activas")
    tasa_activa_mercado = (
        pd.to_numeric(df_activas["tasa_efectiva_promedio"], errors="coerce").mean()
        if not df_activas.empty else np.nan
    )
    df_captacion = load_silver("enrichment_tasas_captacion")
    tasa_captacion_mercado = (
        pd.to_numeric(df_captacion["tasa"], errors="coerce").replace(0, np.nan).mean()
        if not df_captacion.empty else np.nan
    )

    balance_eur = merged["balance"].astype("float64")
    balance_usd = (balance_eur / rate_eur.where(rate_eur > 0, FX_FALLBACK_EUR)).round(2)

    has_housing = merged["housing"].astype(bool)
    has_loan = merged["loan"].astype(bool)
    has_default = merged["default"].astype(bool)

    # Regla de negocio: spread de riesgo sobre IBR
    spread = 1.0 + np.where(has_housing | has_loan, 2.5, 0.0) + np.where(has_default, 1.5, 0.0)
    natural_id = pd.Series("bm_" + merged.index.astype(str), index=merged.index)

    df_fact = pd.DataFrame({
        "id_campana_contacto": "contact_" + merged.index.astype(str),
        "sk_cliente": map_surrogate_keys(natural_id, "bank_marketing"),
        "sk_fecha": _to_sk_fecha(merged["fecha"]),
        "sk_tipo_credito": generate_surrogate_key("perabank_etl", "CAPTACION::DEPOSITO_A_TERMINO"),
        "balance_eur": balance_eur,
        "balance_usd": balance_usd,
        "tiene_hipoteca": has_housing,
        "tiene_prestamo_personal": has_loan,
        "tiene_mora": has_default,
        "tasa_ibr_referencia": tasa_ibr.astype("float64"),
        "tasa_activa_mercado": tasa_activa_mercado,
        "tasa_captacion_mercado": tasa_captacion_mercado,
        "spread_tasa_credito": np.round(spread, 2),
        "tasa_oferta_estimada": (tasa_ibr + spread).round(2),
        "duracion_contacto_seg": merged["duration"].astype("int64"),
        "resultado_previo": merged["poutcome"].fillna("DESCONOCIDO"),
        "suscrito_deposito": merged["deposit"].astype(bool),
        "source_system": "bank_marketing",
    })
    GOLD_FACT_CAMPANA_MARCADO_CONTRACT.validate(df_fact)
    logger.info("fact_campana_marcado finalizada con %s registros.", len(df_fact))
    return df_fact


def build_fact_fraude_tarjeta() -> pd.DataFrame:
    """Hechos de fraude con tarjeta, enriquecidos con volatilidad FX y banda de riesgo macro."""
    logger.info("Construyendo fact_fraude_tarjeta...")
    df_cc = load_silver("creditcard")
    if df_cc.empty:
        return pd.DataFrame()

    df_market = _load_market()
    vol_30d = 12.5  # Volatilidad baseline
    if "volatilidad_fx_30d" in df_market.columns and not df_market.empty:
        vol_30d = float(df_market["volatilidad_fx_30d"].mean())

    monto_usd = df_cc["Amount"].astype("float64")

    # Regla de negocio: banda de riesgo macro basada en monto y volatilidad FX
    banda = np.select(
        [monto_usd > 500.0, (monto_usd > 100.0) & (vol_30d > 10.0), monto_usd > 100.0],
        ["ALTO", "ALTO", "MEDIO"],
        default="BAJO",
    )

    df_fact = pd.DataFrame({
        "id_evento_tarjeta": "cc_" + df_cc.index.astype(str),
        "segundos_transcurridos": df_cc["Time"].astype("float64"),
        "monto_usd": monto_usd,
        "volatilidad_fx_30d": round(vol_30d, 4),
        "banda_riesgo_macro": banda,
        "es_fraude": df_cc["Class"].astype(bool),
        "source_system": "creditcard",
    })

    # Componentes PCA anonimizados: se arrastran al hecho porque son la señal
    # predictiva del dataset. Sin ellos el modelo de fraude solo vería monto y tiempo.
    componentes = [c for c in df_cc.columns if c.startswith("V") and c[1:].isdigit()]
    for columna in componentes:
        df_fact[columna] = df_cc[columna].astype("float64")

    GOLD_FACT_FRAUDE_TARJETA_CONTRACT.validate(df_fact)
    logger.info("fact_fraude_tarjeta finalizada con %s registros.", len(df_fact))
    return df_fact


def build_fact_contrato_estatal(df_dim_proveedor: pd.DataFrame) -> pd.DataFrame:
    """Hechos de contratación pública (SECOP II), base para riesgo de crédito
    corporativo y factoring estatal sobre el valor pendiente de pago."""
    logger.info("Construyendo fact_contrato_estatal...")
    now_iso = _now_iso()
    df_secop = load_silver("enrichment_secop_contratos")
    if df_secop.empty:
        return pd.DataFrame()

    valor_contrato = pd.to_numeric(df_secop["valor_del_contrato"], errors="coerce")
    valor_pagado = pd.to_numeric(df_secop["valor_pagado"], errors="coerce")

    sk_proveedor = map_surrogate_keys(df_secop["documento_proveedor"].astype(str), "secop")
    validos = set(df_dim_proveedor["sk_proveedor"])
    sk_proveedor = sk_proveedor.where(sk_proveedor.isin(validos), unknown_key("dim_proveedor_estatal"))

    df_fact = pd.DataFrame({
        "id_contrato": df_secop["id_contrato"].astype(str),
        "sk_proveedor": sk_proveedor,
        "sk_fecha": _to_sk_fecha(df_secop["fecha_de_fin_del_contrato"]),
        "nombre_entidad_contratante": df_secop["nombre_entidad"].astype(str),
        "nit_entidad": df_secop["nit_entidad"].astype(str),
        "departamento": df_secop["departamento"].astype(str),
        "ciudad": df_secop["ciudad"].astype(str),
        "estado_contrato": df_secop["estado_contrato"].astype(str),
        "tipo_de_contrato": df_secop["tipo_de_contrato"].astype(str),
        "modalidad_de_contratacion": df_secop["modalidad_de_contratacion"].astype(str),
        "valor_del_contrato": valor_contrato,
        "valor_pagado": valor_pagado,
        "valor_pendiente": (valor_contrato.fillna(0) - valor_pagado.fillna(0)).clip(lower=0),
        "source_system": "enrichment_secop_contratos",
        "ingested_at": now_iso,
    })
    GOLD_FACT_CONTRATO_ESTATAL_CONTRACT.validate(df_fact)
    logger.info("fact_contrato_estatal finalizada con %s registros.", len(df_fact))
    return df_fact


def build_fact_tasas_mercado(df_dim_entidad: pd.DataFrame, df_dim_tipo_credito: pd.DataFrame) -> pd.DataFrame:
    """Hechos de tasas de mercado. Grano: una observación de tasa.

    Unifica en formato largo benchmarks macro (IBR, Treasuries) con tasas por entidad
    (activas de colocación y pasivas de captación), que tienen granos distintos en
    origen pero comparten la misma pregunta de negocio: qué tasa rigió, cuándo y para quién.
    """
    logger.info("Construyendo fact_tasas_mercado...")
    now_iso = _now_iso()
    sk_entidad_desconocida = unknown_key("dim_entidad_financiera")
    sk_tipo_desconocido = unknown_key("dim_tipo_credito")
    entidades_validas = set(df_dim_entidad["sk_entidad"])
    tipos_validos = set(df_dim_tipo_credito["sk_tipo_credito"])
    parts = []

    # 1. Benchmarks macro y precios FX diarios (Yahoo Finance + snapshot IBR)
    df_market = _load_market()
    if not df_market.empty:
        macro_cols = {
            "tasa_ibr_overnight": "IBR_OVERNIGHT",
            "tasa_treasury_10y": "TREASURY_10Y",
            "tasa_tbill_3m": "TBILL_3M",
            # Los cierres FX se guardan como observaciones de mercado para que el
            # warehouse sea autosuficiente: sin ellos, un consumidor que quiera
            # comparar TRM oficial contra mercado tendría que salir a los parquet.
            "tasa_usd_cop": "FX_USDCOP",
            "tasa_usd_eur": "FX_USDEUR",
            "tasa_usd_inr": "FX_USDINR",
        }
        presentes = [c for c in macro_cols if c in df_market.columns]
        largo = df_market.melt(
            id_vars=["fecha"], value_vars=presentes,
            var_name="columna", value_name="valor_tasa",
        ).dropna(subset=["valor_tasa"])
        largo["tipo_tasa"] = largo["columna"].map(macro_cols)

        parts.append(pd.DataFrame({
            "id_observacion_tasa": map_surrogate_keys(
                largo["tipo_tasa"] + "::" + largo["fecha"].astype(str), "enrichment_market"),
            "sk_fecha": _to_sk_fecha(largo["fecha"]),
            "sk_entidad": sk_entidad_desconocida,
            "sk_tipo_credito": sk_tipo_desconocido,
            "tipo_tasa": largo["tipo_tasa"],
            "valor_tasa": largo["valor_tasa"].astype("float64"),
            "monto_asociado": np.nan,
            "source_system": "enrichment_market",
            "ingested_at": now_iso,
        }))

    # 2. TRM oficial (Superfinanciera) y tasa de referencia BCE, serie histórica completa
    for df_serie, columna, tipo, sistema in (
        (_load_trm(), "tasa_trm_oficial", "TRM_OFICIAL", "enrichment_trm_gov"),
        (_load_ecb(), "tasa_eur_usd", "ECB_EURUSD", "enrichment_ecb"),
    ):
        if df_serie.empty:
            continue
        serie = df_serie.dropna(subset=[columna])
        parts.append(pd.DataFrame({
            "id_observacion_tasa": map_surrogate_keys(
                tipo + "::" + serie["fecha"].astype(str), sistema),
            "sk_fecha": _to_sk_fecha(serie["fecha"]),
            "sk_entidad": sk_entidad_desconocida,
            "sk_tipo_credito": sk_tipo_desconocido,
            "tipo_tasa": tipo,
            "valor_tasa": serie[columna].astype("float64"),
            "monto_asociado": np.nan,
            "source_system": sistema,
            "ingested_at": now_iso,
        }))

    # 3. Tasas activas de colocación por entidad
    df_activas = load_silver("enrichment_tasas_activas")
    if not df_activas.empty:
        sk_entidad = map_surrogate_keys(df_activas["codigo_entidad"].astype(str), "superfinanciera")
        sk_entidad = sk_entidad.where(sk_entidad.isin(entidades_validas), sk_entidad_desconocida)

        clave_tipo = ("COLOCACION::" + df_activas["tipo_de_cr_dito"].astype(str)
                      + "::" + df_activas["producto_de_cr_dito"].astype(str))
        sk_tipo = map_surrogate_keys(clave_tipo, "superfinanciera")
        sk_tipo = sk_tipo.where(sk_tipo.isin(tipos_validos), sk_tipo_desconocido)

        parts.append(pd.DataFrame({
            "id_observacion_tasa": map_surrogate_keys(
                "TASA_ACTIVA::" + df_activas.index.astype(str), "enrichment_tasas_activas"),
            "sk_fecha": _to_sk_fecha(df_activas["fecha_corte"]),
            "sk_entidad": sk_entidad,
            "sk_tipo_credito": sk_tipo,
            "tipo_tasa": "TASA_ACTIVA",
            "valor_tasa": pd.to_numeric(df_activas["tasa_efectiva_promedio"], errors="coerce"),
            "monto_asociado": pd.to_numeric(df_activas["montos_desembolsados"], errors="coerce"),
            "source_system": "enrichment_tasas_activas",
            "ingested_at": now_iso,
        }))

    # 4. Tasas pasivas de captación por entidad
    df_captacion = load_silver("enrichment_tasas_captacion")
    if not df_captacion.empty:
        sk_entidad = map_surrogate_keys(df_captacion["codigoentidad"].astype(str), "superfinanciera")
        sk_entidad = sk_entidad.where(sk_entidad.isin(entidades_validas), sk_entidad_desconocida)

        clave_tipo = ("CAPTACION::" + df_captacion["descripcion"].astype(str)
                      + "::" + df_captacion["nombre_unidad_de_captura"].astype(str))
        sk_tipo = map_surrogate_keys(clave_tipo, "superfinanciera")
        sk_tipo = sk_tipo.where(sk_tipo.isin(tipos_validos), sk_tipo_desconocido)

        parts.append(pd.DataFrame({
            "id_observacion_tasa": map_surrogate_keys(
                "TASA_CAPTACION::" + df_captacion.index.astype(str), "enrichment_tasas_captacion"),
            "sk_fecha": _to_sk_fecha(df_captacion["fechacorte"]),
            "sk_entidad": sk_entidad,
            "sk_tipo_credito": sk_tipo,
            "tipo_tasa": "TASA_CAPTACION",
            "valor_tasa": pd.to_numeric(df_captacion["tasa"], errors="coerce"),
            "monto_asociado": pd.to_numeric(df_captacion["monto"], errors="coerce"),
            "source_system": "enrichment_tasas_captacion",
            "ingested_at": now_iso,
        }))

    df_fact = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df_fact = df_fact.dropna(subset=["valor_tasa"]).reset_index(drop=True)
    GOLD_FACT_TASAS_MERCADO_CONTRACT.validate(df_fact)
    logger.info("fact_tasas_mercado finalizada con %s observaciones.", len(df_fact))
    return df_fact
