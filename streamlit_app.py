"""PeraBank — Panel de riesgo y mercado sobre el warehouse Gold (copo de nieve).

Lee exclusivamente de data/gold/*.parquet a través de DuckDB (etl/common/warehouse.py),
que consulta los Parquet en sitio sin copiarlos a otra base. Toda consulta va cacheada.
"""
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

from etl.common import warehouse
from etl.common.config import GOLD_DIR

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH = REPO_ROOT / "perabank_risk_pipeline_v1.joblib"
CLUSTERING_PATH = REPO_ROOT / "models" / "perabank_clustering_models_v1.joblib"

st.set_page_config(layout="wide", page_title="PeraBank Risk & Market Intelligence", page_icon="🏦")
load_dotenv(REPO_ROOT / ".env")

PLOTLY_TEMPLATE = "plotly_dark"

# Inferencia local con Ollama: sin API keys, sin costo por token y sin enviar
# perfiles de clientes a un servicio externo. qwen2.5:3b responde en ~10s y no
# inventa cifras de mercado, a diferencia de modelos más grandes probados aquí.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")


# ---------------------------------------------------------------------------
# Acceso a datos
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Ejecuta una consulta contra el warehouse. Devuelve DataFrame vacío si la
    tabla no existe todavía, para que la UI degrade con un aviso y no una excepción."""
    try:
        return warehouse.query(sql, params)
    except Exception as exc:  # tabla ausente o esquema desactualizado
        st.session_state.setdefault("errores_sql", []).append(str(exc))
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def tablas_disponibles() -> set:
    return warehouse.tables()


@st.cache_resource(show_spinner=False)
def cargar_clustering() -> dict:
    if not CLUSTERING_PATH.exists():
        return {}
    try:
        return joblib.load(CLUSTERING_PATH)
    except Exception as exc:
        st.warning(f"No se pudo cargar el artefacto de segmentación: {exc}")
        return {}


@st.cache_resource(show_spinner=False)
def cargar_modelos() -> dict:
    if not MODEL_PATH.exists():
        return {}
    try:
        return joblib.load(MODEL_PATH)
    except Exception as exc:
        st.warning(f"No se pudo cargar el artefacto de modelos: {exc}")
        return {}


def exigir_warehouse() -> bool:
    if not warehouse.tables():
        st.error(f"No se encontraron tablas gold en `{GOLD_DIR}`.")
        st.info("Genéralo con:  `python -m etl.run_pipeline`")
        return False
    faltantes = {"dim_cliente", "fact_transaccion"} - tablas_disponibles()
    if faltantes:
        st.error(f"Faltan tablas en el warehouse: {', '.join(sorted(faltantes))}")
        st.info("Reconstruye la capa gold con:  `python -m etl.gold.run_gold`")
        return False
    return True


def moneda(valor: float, simbolo: str = "$") -> str:
    if pd.isna(valor):
        return "n/d"
    for umbral, sufijo in ((1e12, "B"), (1e9, "MM"), (1e6, "M"), (1e3, "K")):
        if abs(valor) >= umbral:
            return f"{simbolo}{valor / umbral:,.2f}{sufijo}"
    return f"{simbolo}{valor:,.2f}"


# ---------------------------------------------------------------------------
# Tab 1 — Portafolio ejecutivo
# ---------------------------------------------------------------------------

def tab_portafolio():
    st.subheader("Resumen ejecutivo del portafolio")

    kpis = run_query("""
        SELECT (SELECT COUNT(*) FROM dim_cliente)                     AS clientes,
               (SELECT COUNT(*) FROM fact_transaccion)                AS transacciones,
               (SELECT SUM(monto_usd) FROM fact_transaccion)          AS volumen_usd,
               (SELECT SUM(monto_cop) FROM fact_transaccion)          AS volumen_cop,
               (SELECT COUNT(*) FROM fact_contrato_estatal)           AS contratos,
               (SELECT SUM(valor_pendiente) FROM fact_contrato_estatal) AS pendiente_cop
    """)
    if kpis.empty:
        st.warning("No hay datos de portafolio disponibles.")
        return
    k = kpis.iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Clientes en el warehouse", f"{int(k.clientes):,}")
    c2.metric("Volumen transaccional (USD)", moneda(k.volumen_usd),
              delta=f"{int(k.transacciones):,} transacciones")
    c3.metric("Volumen equivalente (COP)", moneda(k.volumen_cop, "COL$"))
    c4.metric("Contratos estatales", f"{int(k.contratos):,}",
              delta=f"{moneda(k.pendiente_cop, 'COL$')} por ejecutar")

    st.divider()
    izq, der = st.columns(2)

    with izq:
        st.markdown("**Distribución de riesgo macro (eventos de tarjeta)**")
        riesgo = run_query("""
            SELECT banda_riesgo_macro AS banda, COUNT(*) AS eventos,
                   SUM(CASE WHEN es_fraude=1 THEN 1 ELSE 0 END) AS fraudes
            FROM fact_fraude_tarjeta GROUP BY banda ORDER BY eventos DESC
        """)
        if riesgo.empty:
            st.info("`fact_fraude_tarjeta` no disponible.")
        else:
            fig = px.pie(riesgo, names="banda", values="eventos", hole=0.55,
                         template=PLOTLY_TEMPLATE,
                         color_discrete_sequence=["#ef4444", "#f59e0b", "#22c55e"])
            fig.update_layout(margin=dict(t=10, b=10, l=0, r=0), height=300)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Tasa de fraude observada: {riesgo.fraudes.sum() / riesgo.eventos.sum():.3%}"
            )

    with der:
        st.markdown("**Clientes por sistema fuente**")
        fuentes = run_query("""
            SELECT source_system AS fuente, COUNT(*) AS clientes
            FROM dim_cliente GROUP BY fuente ORDER BY clientes DESC
        """)
        if not fuentes.empty:
            fig = px.bar(fuentes, x="clientes", y="fuente", orientation="h",
                         template=PLOTLY_TEMPLATE, text="clientes",
                         color_discrete_sequence=["#38bdf8"])
            fig.update_layout(margin=dict(t=10, b=10), height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Cada fuente es un sistema distinto: no comparten identidad de cliente, "
                "por eso el modelo las mantiene separadas por `source_system`."
            )

    st.divider()
    st.markdown("### Filtros demográficos de clientes")
    st.caption(
        "El perfil demográfico solo lo reporta la fuente de campañas (`bank_marketing`); "
        "las demás fuentes traen el cliente sin atributos y aparecen como DESCONOCIDO."
    )

    opciones = run_query("""
        SELECT DISTINCT ocupacion, estado_civil, nivel_educativo, ubicacion
        FROM dim_cliente WHERE ocupacion IS NOT NULL
    """)
    if opciones.empty:
        st.info("No hay atributos demográficos disponibles.")
        return

    f1, f2, f3, f4 = st.columns(4)
    edad = f1.slider("Rango de edad", 18, 95, (25, 60))
    ocupaciones = f2.multiselect("Ocupación", sorted(opciones.ocupacion.dropna().unique()))
    civiles = f3.multiselect("Estado civil", sorted(opciones.estado_civil.dropna().unique()))
    educaciones = f4.multiselect("Nivel educativo", sorted(opciones.nivel_educativo.dropna().unique()))

    condiciones = ["edad BETWEEN ? AND ?"]
    params = [edad[0], edad[1]]
    for columna, seleccion in (("ocupacion", ocupaciones), ("estado_civil", civiles),
                               ("nivel_educativo", educaciones)):
        if seleccion:
            condiciones.append(f"{columna} IN ({','.join('?' * len(seleccion))})")
            params.extend(seleccion)

    filtrados = run_query(
        f"""SELECT edad, ocupacion, estado_civil, nivel_educativo, genero, ubicacion
            FROM dim_cliente WHERE {' AND '.join(condiciones)} LIMIT 5000""",
        tuple(params),
    )
    st.metric("Clientes que cumplen el filtro (máx. 5.000 mostrados)", f"{len(filtrados):,}")
    if not filtrados.empty:
        g1, g2 = st.columns([2, 3])
        with g1:
            fig = px.histogram(filtrados, x="edad", nbins=30, template=PLOTLY_TEMPLATE,
                               color_discrete_sequence=["#a78bfa"])
            fig.update_layout(height=280, margin=dict(t=30, b=10), title="Distribución etaria")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            top = filtrados.ocupacion.value_counts().head(10).reset_index()
            top.columns = ["ocupacion", "clientes"]
            fig = px.bar(top, x="clientes", y="ocupacion", orientation="h",
                         template=PLOTLY_TEMPLATE, color_discrete_sequence=["#34d399"])
            fig.update_layout(height=280, margin=dict(t=30, b=10), title="Top ocupaciones")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(filtrados.head(200), use_container_width=True, height=240)


# ---------------------------------------------------------------------------
# Tab 2 — Tesorería y mercado
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def serie_mercado(tipos: tuple) -> pd.DataFrame:
    """Series de mercado por fecha desde fact_tasas_mercado (formato largo)."""
    marcadores = ",".join("?" * len(tipos))
    return run_query(f"""
        SELECT d.fecha, t.tipo_tasa, AVG(t.valor_tasa) AS valor
        FROM fact_tasas_mercado t
        JOIN dim_fecha d ON d.sk_fecha = t.sk_fecha
        WHERE t.tipo_tasa IN ({marcadores})
        GROUP BY d.fecha, t.tipo_tasa
        ORDER BY d.fecha
    """, tipos)


def tab_mercado():
    st.subheader("Tesorería, macro y mercado multi-divisa")

    fx = serie_mercado(("TRM_OFICIAL", "FX_USDCOP", "ECB_EURUSD"))
    if fx.empty:
        st.warning("No hay series FX en `fact_tasas_mercado`.")
    else:
        pivote = fx.pivot(index="fecha", columns="tipo_tasa", values="valor").reset_index()
        solapamiento = pivote.dropna(subset=[c for c in ("TRM_OFICIAL", "FX_USDCOP")
                                             if c in pivote.columns])

        st.markdown("**TRM oficial (Superfinanciera) vs. cierre de mercado (Yahoo Finance)**")
        ventana = st.radio(
            "Ventana", ["Solo periodo con ambas fuentes", "Histórico completo de TRM"],
            horizontal=True, index=0,
        )
        datos = solapamiento if ventana.startswith("Solo") else pivote

        fig = go.Figure()
        if "TRM_OFICIAL" in datos:
            fig.add_trace(go.Scatter(x=datos.fecha, y=datos.TRM_OFICIAL,
                                     name="TRM oficial (Superfinanciera)",
                                     line=dict(color="#fbbf24", width=2)))
        if "FX_USDCOP" in datos:
            fig.add_trace(go.Scatter(x=datos.fecha, y=datos.FX_USDCOP,
                                     name="USD/COP (Yahoo Finance)",
                                     line=dict(color="#38bdf8", width=2, dash="dot")))
        fig.update_layout(template=PLOTLY_TEMPLATE, height=360, hovermode="x unified",
                          margin=dict(t=20, b=10), yaxis_title="COP por USD",
                          legend=dict(orientation="h", y=1.12))
        st.plotly_chart(fig, use_container_width=True)

        if not solapamiento.empty and {"TRM_OFICIAL", "FX_USDCOP"} <= set(solapamiento.columns):
            brecha = (solapamiento.FX_USDCOP - solapamiento.TRM_OFICIAL).abs().mean()
            b1, b2 = st.columns(2)
            b1.metric("Brecha media TRM vs. mercado", f"{brecha:,.2f} COP",
                      delta=f"{len(solapamiento)} días comparables")
            if "ECB_EURUSD" in pivote:
                ultimo = pivote.dropna(subset=["ECB_EURUSD"]).tail(1)
                if not ultimo.empty:
                    b2.metric("EUR/USD de referencia BCE",
                              f"{ultimo.ECB_EURUSD.iloc[0]:.4f}",
                              delta=f"al {ultimo.fecha.iloc[0]}")
        st.caption(
            "La TRM tiene histórico desde 1991 y los cierres de Yahoo solo del último año "
            "descargado; por defecto se grafica el periodo donde ambas fuentes existen para "
            "no sugerir una comparación donde solo hay una serie."
        )

    st.divider()
    izq, der = st.columns(2)

    tasas = run_query("""
        SELECT d.fecha, t.tipo_tasa, AVG(t.valor_tasa) AS valor
        FROM fact_tasas_mercado t
        JOIN dim_fecha d ON d.sk_fecha = t.sk_fecha
        WHERE t.tipo_tasa IN ('TREASURY_10Y','TBILL_3M','IBR_OVERNIGHT')
        GROUP BY d.fecha, t.tipo_tasa ORDER BY d.fecha
    """)

    with izq:
        st.markdown("**Curva de rendimientos del Tesoro de EE.UU.**")
        curva = tasas[tasas.tipo_tasa.isin(["TREASURY_10Y", "TBILL_3M"])]
        if curva.empty:
            st.info("Sin datos de Treasuries.")
        else:
            fig = px.line(curva, x="fecha", y="valor", color="tipo_tasa",
                          template=PLOTLY_TEMPLATE,
                          color_discrete_map={"TREASURY_10Y": "#f472b6", "TBILL_3M": "#4ade80"})
            fig.update_layout(height=320, hovermode="x unified", yaxis_title="Rendimiento (%)",
                              margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.15))
            st.plotly_chart(fig, use_container_width=True)

            pivot = curva.pivot(index="fecha", columns="tipo_tasa", values="valor").dropna()
            if not pivot.empty:
                spread = (pivot["TREASURY_10Y"] - pivot["TBILL_3M"]).iloc[-1]
                st.metric("Spread 10Y − 3M (último dato)", f"{spread:+.2f} pp",
                          delta="curva normal" if spread > 0 else "curva invertida",
                          delta_color="normal" if spread > 0 else "inverse")

    with der:
        st.markdown("**IBR overnight (Banco de la República)**")
        ibr = tasas[tasas.tipo_tasa == "IBR_OVERNIGHT"]
        if ibr.empty:
            st.info("Sin datos de IBR.")
        else:
            fig = px.line(ibr, x="fecha", y="valor", template=PLOTLY_TEMPLATE,
                          color_discrete_sequence=["#fb923c"])
            fig.update_layout(height=320, yaxis_title="Tasa (%)", margin=dict(t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
            if ibr.valor.nunique() == 1:
                st.warning(
                    f"El IBR aparece constante en {ibr.valor.iloc[0]:.2f}%: datos.gov.co "
                    "bloquea la descarga automatizada (HTTP 403) y el ETL usa el último "
                    "valor verificado en vez de inventar una serie."
                )

    st.divider()
    st.markdown("### Conversor de divisas con tasas históricas reales")

    cotizaciones = serie_mercado(("TRM_OFICIAL", "FX_USDCOP", "FX_USDEUR", "FX_USDINR"))
    if cotizaciones.empty:
        st.info("Conversor no disponible sin series de mercado.")
        return

    etiquetas = {
        "TRM_OFICIAL": "COP — TRM oficial (Superfinanciera)",
        "FX_USDCOP": "COP — cierre de mercado (Yahoo Finance)",
        "FX_USDEUR": "EUR — cierre de mercado (Yahoo Finance)",
        "FX_USDINR": "INR — cierre de mercado (Yahoo Finance)",
    }
    presentes = [t for t in etiquetas if t in set(cotizaciones.tipo_tasa)]

    c1, c2, c3 = st.columns(3)
    monto = c1.number_input("Monto en USD", min_value=0.0, value=1000.0, step=100.0)
    destino = c2.selectbox("Convertir a", presentes, format_func=lambda t: etiquetas[t])
    fechas = cotizaciones[cotizaciones.tipo_tasa == destino].fecha.tolist()[::-1]
    fecha_sel = c3.selectbox("Fecha de la cotización", fechas)

    fila = cotizaciones[(cotizaciones.tipo_tasa == destino) & (cotizaciones.fecha == fecha_sel)]
    if fila.empty or pd.isna(fila.valor.iloc[0]):
        st.warning(f"No hay cotización de {etiquetas[destino]} para {fecha_sel}.")
        return

    tasa = float(fila.valor.iloc[0])
    divisa = etiquetas[destino].split(" — ")[0]
    r1, r2 = st.columns(2)
    r1.metric(f"Equivalente en {divisa}", f"{monto * tasa:,.2f} {divisa}")
    r2.metric(f"Cotización del {fecha_sel}", f"{tasa:,.4f} {divisa}/USD")
    st.caption("Cada conversión usa la cotización real observada ese día, no un promedio.")


# ---------------------------------------------------------------------------
# Tab 3 — Tasas bancarias
# ---------------------------------------------------------------------------

def tab_tasas_bancarias():
    st.subheader("Benchmarking de tasas bancarias (Superfinanciera)")

    activas = run_query("""
        SELECT e.nombre_entidad, tc.nombre_tipo AS tipo_credito,
               AVG(t.valor_tasa) AS tasa_promedio,
               SUM(t.monto_asociado) AS monto_desembolsado,
               COUNT(*) AS observaciones
        FROM fact_tasas_mercado t
        JOIN dim_entidad_financiera e ON e.sk_entidad = t.sk_entidad
        JOIN dim_tipo_credito tc      ON tc.sk_tipo_credito = t.sk_tipo_credito
        WHERE t.tipo_tasa = 'TASA_ACTIVA' AND t.valor_tasa > 0
        GROUP BY e.nombre_entidad, tc.nombre_tipo
        HAVING observaciones >= 3
        ORDER BY tasa_promedio DESC
    """)
    if activas.empty:
        st.warning("No hay tasas activas disponibles en `fact_tasas_mercado`.")
        return

    tipos = sorted(activas.tipo_credito.unique())
    seleccion = st.multiselect("Modalidad de crédito", tipos, default=tipos[:3])
    filtrado = activas[activas.tipo_credito.isin(seleccion)] if seleccion else activas

    c1, c2, c3 = st.columns(3)
    c1.metric("Tasa activa promedio", f"{filtrado.tasa_promedio.mean():.2f}%")
    c2.metric("Entidades comparadas", f"{filtrado.nombre_entidad.nunique()}")
    c3.metric("Desembolsos observados", moneda(filtrado.monto_desembolsado.sum(), "COL$"))

    top = filtrado.nlargest(20, "tasa_promedio")
    fig = px.bar(top, x="tasa_promedio", y="nombre_entidad", color="tipo_credito",
                 orientation="h", template=PLOTLY_TEMPLATE,
                 labels={"tasa_promedio": "Tasa efectiva promedio (%)", "nombre_entidad": ""})
    fig.update_layout(height=520, margin=dict(t=20, b=10), legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Costo de fondeo: tasas de captación vs. benchmark")

    captacion = run_query("""
        SELECT e.nombre_entidad, tc.nombre_tipo AS instrumento,
               AVG(t.valor_tasa) AS tasa_captacion, SUM(t.monto_asociado) AS monto_captado
        FROM fact_tasas_mercado t
        JOIN dim_entidad_financiera e ON e.sk_entidad = t.sk_entidad
        JOIN dim_tipo_credito tc      ON tc.sk_tipo_credito = t.sk_tipo_credito
        WHERE t.tipo_tasa = 'TASA_CAPTACION' AND t.valor_tasa > 0
        GROUP BY e.nombre_entidad, tc.nombre_tipo
        ORDER BY tasa_captacion DESC LIMIT 25
    """)
    ibr = run_query("""
        SELECT AVG(valor_tasa) AS ibr FROM fact_tasas_mercado WHERE tipo_tasa='IBR_OVERNIGHT'
    """)
    if captacion.empty:
        st.info("Sin datos de captación.")
        return

    tasa_libre = float(ibr.ibr.iloc[0]) if not ibr.empty and pd.notna(ibr.ibr.iloc[0]) else None
    captacion["spread_sobre_ibr"] = captacion.tasa_captacion - (tasa_libre or 0)

    fig = px.scatter(captacion, x="tasa_captacion", y="nombre_entidad", size="monto_captado",
                     color="spread_sobre_ibr", template=PLOTLY_TEMPLATE,
                     color_continuous_scale="RdYlGn_r",
                     labels={"tasa_captacion": "Tasa de captación (%)", "nombre_entidad": "",
                             "spread_sobre_ibr": "Spread vs IBR"})
    fig.update_layout(height=520, margin=dict(t=20, b=10))
    if tasa_libre:
        fig.add_vline(x=tasa_libre, line_dash="dash", line_color="#fbbf24",
                      annotation_text=f"IBR {tasa_libre:.2f}%")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(captacion.round(2), use_container_width=True, height=240)


# ---------------------------------------------------------------------------
# Tab 4 — SECOP II / factoring
# ---------------------------------------------------------------------------

def tab_factoring():
    st.subheader("Riesgo corporativo y factoring estatal (SECOP II)")

    filtros = run_query("""
        SELECT DISTINCT departamento, estado_contrato FROM fact_contrato_estatal
    """)
    if filtros.empty:
        st.warning("`fact_contrato_estatal` no disponible.")
        return

    c1, c2, c3 = st.columns(3)
    deptos = c1.multiselect("Departamento", sorted(filtros.departamento.dropna().unique()))
    estados = c2.multiselect("Estado del contrato", sorted(filtros.estado_contrato.dropna().unique()))
    pyme = c3.radio("Tipo de proveedor", ["Todos", "Solo PYME", "Solo gran empresa"], horizontal=True)

    condiciones, params = ["f.valor_del_contrato > 0"], []
    if deptos:
        condiciones.append(f"f.departamento IN ({','.join('?' * len(deptos))})")
        params += deptos
    if estados:
        condiciones.append(f"f.estado_contrato IN ({','.join('?' * len(estados))})")
        params += estados
    if pyme == "Solo PYME":
        condiciones.append("p.es_pyme = 1")
    elif pyme == "Solo gran empresa":
        condiciones.append("p.es_pyme = 0")

    contratos = run_query(f"""
        SELECT p.proveedor_adjudicado AS proveedor, p.documento_proveedor, p.es_pyme,
               f.departamento, f.estado_contrato, f.tipo_de_contrato,
               f.valor_del_contrato, f.valor_pagado, f.valor_pendiente
        FROM fact_contrato_estatal f
        JOIN dim_proveedor_estatal p ON p.sk_proveedor = f.sk_proveedor
        WHERE {' AND '.join(condiciones)}
    """, tuple(params))

    if contratos.empty:
        st.info("Ningún contrato cumple los filtros seleccionados.")
        return

    contratos["ratio_pago"] = (contratos.valor_pagado / contratos.valor_del_contrato).clip(0, 1)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Contratos", f"{len(contratos):,}")
    k2.metric("Valor contratado", moneda(contratos.valor_del_contrato.sum(), "COL$"))
    k3.metric("Ejecución de pago media", f"{contratos.ratio_pago.mean():.1%}")
    k4.metric("Expuesto a factoring", moneda(contratos.valor_pendiente.sum(), "COL$"),
              delta="saldo pendiente de pago")

    st.divider()
    st.markdown("### Score de calificación para factoring estatal")
    st.caption(
        "Score = 70% ejecución de pago histórica + 30% trayectoria (número de contratos, "
        "normalizado). Es una regla de negocio determinística sobre datos reales, no una "
        "predicción del modelo."
    )

    proveedores = contratos.groupby(["proveedor", "documento_proveedor", "es_pyme"]).agg(
        contratos=("valor_del_contrato", "size"),
        valor_total=("valor_del_contrato", "sum"),
        pendiente=("valor_pendiente", "sum"),
        ratio_pago=("ratio_pago", "mean"),
    ).reset_index()

    trayectoria = proveedores.contratos / proveedores.contratos.max()
    proveedores["score_factoring"] = (0.7 * proveedores.ratio_pago + 0.3 * trayectoria) * 100
    proveedores["calificacion"] = pd.cut(
        proveedores.score_factoring, bins=[-0.1, 35, 60, 100],
        labels=["RECHAZAR", "REVISAR", "APROBAR"],
    )

    g1, g2 = st.columns([3, 2])
    with g1:
        top = proveedores.nlargest(15, "score_factoring")
        fig = px.bar(top, x="score_factoring", y="proveedor", color="calificacion",
                     orientation="h", template=PLOTLY_TEMPLATE,
                     color_discrete_map={"APROBAR": "#22c55e", "REVISAR": "#f59e0b",
                                         "RECHAZAR": "#ef4444"},
                     labels={"score_factoring": "Score (0-100)", "proveedor": ""})
        fig.update_layout(height=460, margin=dict(t=20, b=10), yaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        conteo = proveedores.calificacion.value_counts().reset_index()
        conteo.columns = ["calificacion", "proveedores"]
        fig = px.pie(conteo, names="calificacion", values="proveedores", hole=0.5,
                     template=PLOTLY_TEMPLATE,
                     color="calificacion",
                     color_discrete_map={"APROBAR": "#22c55e", "REVISAR": "#f59e0b",
                                         "RECHAZAR": "#ef4444"})
        fig.update_layout(height=280, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.metric("Proveedores evaluados", f"{len(proveedores):,}",
                  delta=f"{int(proveedores.es_pyme.sum()):,} PYME")

    st.dataframe(
        proveedores.sort_values("score_factoring", ascending=False)
        .assign(score_factoring=lambda d: d.score_factoring.round(1),
                ratio_pago=lambda d: (d.ratio_pago * 100).round(1))
        .head(300),
        use_container_width=True, height=280,
    )


# ---------------------------------------------------------------------------
# Tab — Segmentación y personas
# ---------------------------------------------------------------------------

CONSULTAS_SEGMENTOS = {
    "retail": """
        SELECT f.sk_cliente AS entidad, s.sk_cluster_retail AS cluster,
               f.balance_usd, f.duracion_contacto_seg, f.tiene_hipoteca,
               f.tiene_prestamo_personal, f.suscrito_deposito
        FROM fact_campana_marcado f
        JOIN cluster_retail s ON s.sk_cliente = f.sk_cliente
    """,
    "proveedores": """
        SELECT p.sk_proveedor AS entidad, s.sk_cluster_supplier AS cluster,
               p.proveedor_adjudicado, p.es_pyme,
               SUM(f.valor_del_contrato) AS valor_del_contrato,
               SUM(f.valor_pagado)       AS valor_pagado,
               COUNT(*)                  AS contratos_adjudicados
        FROM fact_contrato_estatal f
        JOIN dim_proveedor_estatal p ON p.sk_proveedor = f.sk_proveedor
        JOIN cluster_proveedores s ON s.sk_proveedor = p.sk_proveedor
        WHERE f.valor_del_contrato > 0
        GROUP BY p.sk_proveedor, p.proveedor_adjudicado, p.es_pyme, s.sk_cluster_supplier
    """,
    # Se traen las V1..V28 porque la proyección 3D las necesita para transformar con
    # el PCA ajustado; se ocultan de la tabla de detalle, donde no aportan lectura.
    "transaccional": """
        SELECT f.id_evento_tarjeta AS entidad, s.sk_cluster_behavior AS cluster, f.*
        FROM fact_fraude_tarjeta f
        JOIN cluster_transaccional s ON s.id_evento_tarjeta = f.id_evento_tarjeta
    """,
}


@st.cache_data(ttl=600, show_spinner=False)
def datos_segmento(dominio: str) -> pd.DataFrame:
    df = run_query(CONSULTAS_SEGMENTOS[dominio])
    if dominio == "proveedores" and not df.empty:
        df["ratio_desembolso"] = (df.valor_pagado / df.valor_del_contrato).clip(0, 1)
    return df


def _proyeccion_3d(artefacto_dominio: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Reconstruye las 3 componentes de visualización con los transformadores ajustados."""
    features = artefacto_dominio["features"]
    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        return pd.DataFrame()
    viz = artefacto_dominio["pca_viz"]
    escalado = viz["pasos_previos"].transform(df[features])
    componentes = viz["pca"].transform(escalado)
    return pd.DataFrame(componentes[:, :3], columns=["PC1", "PC2", "PC3"], index=df.index)


def tab_segmentacion():
    st.subheader("Segmentación y personas multi-dominio")

    artefacto = cargar_clustering()
    if not artefacto:
        st.warning("No hay artefacto de segmentación disponible.")
        st.info("Genéralo con:  `python models/ml_clustering_pipeline.py`")
        return

    st.caption(
        "Los tres dominios se agrupan por separado, en su grano nativo. No se unen entre sí "
        "porque no comparten llave natural. Las features son solo financieras y de "
        "comportamiento: género, edad y ubicación quedaron fuera del entrenamiento y solo "
        "se usan después, para auditar sesgo."
    )

    etiquetas_dominio = {
        "retail": "👤 Clientes retail",
        "proveedores": "🏗️ Proveedores del Estado",
        "transaccional": "💳 Perfiles transaccionales",
    }
    disponibles = [d for d in etiquetas_dominio if d in artefacto]
    if not disponibles:
        st.warning("El artefacto no contiene dominios segmentados.")
        return

    dominio = st.radio("Dominio", disponibles, horizontal=True,
                       format_func=lambda d: etiquetas_dominio[d])
    info = artefacto[dominio]
    df = datos_segmento(dominio)

    if df.empty:
        st.warning(
            f"El warehouse no tiene asignaciones de cluster para `{dominio}` "
            f"(tabla `cluster_{dominio}`). Genéralas con "
            "`python models/ml_clustering_pipeline.py`."
        )
        return

    metricas = info["metricas_finales"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Clusters (K)", info["k"], delta=f"codo en K={info['k_codo']}")
    m2.metric("Silueta", f"{metricas['silueta']:.3f}",
              delta="supera 0.50" if metricas["silueta"] > 0.5 else "bajo objetivo 0.50",
              delta_color="normal" if metricas["silueta"] > 0.5 else "inverse")
    m3.metric("Davies-Bouldin", f"{metricas['davies_bouldin']:.3f}", delta="menor es mejor",
              delta_color="off")
    m4.metric("Calinski-Harabasz", f"{metricas['calinski_harabasz']:,.0f}", delta="mayor es mejor",
              delta_color="off")

    perfiles = pd.DataFrame(info["perfiles"])
    perfiles.index = perfiles.index.astype(int)
    nombres = {int(k): v for k, v in info["nombres"].items()}
    df["persona"] = df.cluster.map(nombres).fillna("Sin asignar")

    st.divider()
    izq, der = st.columns([3, 2])

    with izq:
        st.markdown("**Separación de clusters en el espacio PCA (3 componentes)**")
        proyeccion = _proyeccion_3d(info, df)
        if proyeccion.empty:
            st.info("No se pudo reconstruir la proyección 3D con las features guardadas.")
        else:
            muestra = pd.concat([df[["persona", "cluster"]], proyeccion], axis=1)
            if len(muestra) > 4000:
                muestra = muestra.sample(4000, random_state=42)
            fig = px.scatter_3d(muestra, x="PC1", y="PC2", z="PC3", color="persona",
                                template=PLOTLY_TEMPLATE, opacity=0.6)
            fig.update_traces(marker=dict(size=2.5))
            fig.update_layout(height=520, margin=dict(t=10, b=10, l=0, r=0),
                              legend=dict(font=dict(size=9), y=0.5))
            st.plotly_chart(fig, use_container_width=True)
            if len(df) > 4000:
                st.caption(f"Muestra de 4.000 de {len(df):,} entidades para mantener el gráfico fluido.")

    with der:
        st.markdown("**Radar comparativo de personas**")
        ejes = [c for c in perfiles.columns if c != "n_entidades"][:6]
        if not ejes:
            st.info("Sin features numéricas para el radar.")
        else:
            # Se normaliza cada eje a [0,1] entre clusters: las features vienen en
            # unidades incomparables (USD, segundos, proporciones binarias).
            normalizado = perfiles[ejes].copy()
            for eje in ejes:
                rango = normalizado[eje].max() - normalizado[eje].min()
                normalizado[eje] = 0.5 if rango == 0 else (normalizado[eje] - normalizado[eje].min()) / rango

            seleccion = st.multiselect(
                "Personas a comparar", sorted(nombres), default=sorted(nombres)[:4],
                format_func=lambda c: nombres.get(c, str(c)),
            )
            radar = go.Figure()
            for cluster in seleccion:
                if cluster not in normalizado.index:
                    continue
                valores = normalizado.loc[cluster, ejes].tolist()
                radar.add_trace(go.Scatterpolar(
                    r=valores + valores[:1], theta=ejes + ejes[:1],
                    fill="toself", name=nombres.get(cluster, str(cluster)),
                ))
            radar.update_layout(template=PLOTLY_TEMPLATE, height=460,
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                margin=dict(t=30, b=10),
                                legend=dict(orientation="h", y=-0.12, font=dict(size=9)))
            st.plotly_chart(radar, use_container_width=True)

    st.divider()
    st.markdown("### Composición y perfiles representativos")

    c1, c2 = st.columns([2, 3])
    with c1:
        conteo = df.persona.value_counts().reset_index()
        conteo.columns = ["persona", "entidades"]
        fig = px.bar(conteo, x="entidades", y="persona", orientation="h",
                     template=PLOTLY_TEMPLATE, color_discrete_sequence=["#38bdf8"])
        fig.update_layout(height=340, margin=dict(t=10, b=10),
                          yaxis=dict(tickfont=dict(size=9)), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Promedios por persona (unidades originales)**")
        tabla = perfiles.copy()
        tabla.insert(0, "persona", [nombres.get(i, str(i)) for i in tabla.index])
        st.dataframe(tabla.round(3), use_container_width=True, height=340)

    persona_sel = st.selectbox("Ver entidades de la persona", sorted(df.persona.unique()))
    componentes_pca = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    detalle = (df[df.persona == persona_sel]
               .drop(columns=["cluster", "sk_cluster_behavior", *componentes_pca], errors="ignore")
               .loc[:, lambda d: ~d.columns.duplicated()])
    st.dataframe(detalle.head(300), use_container_width=True, height=260)
    st.caption(f"{len(detalle):,} entidades en esta persona (se muestran hasta 300).")

    _render_auditoria(info, dominio)


def _render_auditoria(info: dict, dominio: str):
    st.divider()
    st.markdown("### Auditoría de equidad")

    auditoria = info.get("auditoria_equidad", {})
    if not auditoria:
        st.info("Este dominio no tiene atributos demográficos que auditar.")
    else:
        filas = []
        for atributo, resultado in auditoria.items():
            if not resultado.get("auditable", False):
                filas.append({"atributo": atributo, "estado": "No auditable",
                              "desviacion_maxima": None, "detalle": resultado.get("motivo", "")})
            else:
                supera = resultado["supera_regla_4_5"]
                filas.append({
                    "atributo": atributo,
                    "estado": "Revisar" if supera else "Dentro de la regla 4/5",
                    "desviacion_maxima": resultado["desviacion_maxima"],
                    "detalle": "Algún cluster sobre/sub-representa una categoría más allá de 0.8x–1.25x"
                               if supera else "Representación proporcional a la población",
                })
        st.dataframe(pd.DataFrame(filas), use_container_width=True)
        st.caption(
            "Razón de representación por cluster frente a la población (regla de los cuatro "
            "quintos). Estos atributos NO participaron del entrenamiento: la auditoría mide "
            "si el algoritmo los reconstruyó de forma implícita a partir del comportamiento "
            "financiero."
        )

    if dominio == "transaccional" and info.get("concentracion_fraude"):
        st.markdown("**Concentración de fraude por cluster**")
        conc = pd.DataFrame(info["concentracion_fraude"])
        conc.index = conc.index.astype(int)
        conc.insert(0, "persona", [info["nombres"].get(i, info["nombres"].get(str(i), str(i)))
                                   for i in conc.index])
        st.dataframe(conc.round(5), use_container_width=True)
        st.caption(
            "`es_fraude` quedó fuera de la matriz de entrenamiento. Que unos clusters "
            "concentren mucho más fraude que otros indica que la segmentación capturó "
            "estructura de riesgo real sin haber visto nunca la etiqueta."
        )

    if dominio == "proveedores" and info.get("outliers_dbscan") is not None:
        st.metric("Proveedores atípicos según DBSCAN", f"{info['outliers_dbscan']:,}",
                  delta="no encajan en ninguna región densa", delta_color="off")


# ---------------------------------------------------------------------------
# Tab 5 — Scoring + explicación con LLM local (Ollama)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60, show_spinner=False)
def ollama_disponible() -> tuple:
    """Verifica que el servidor local responda y que el modelo esté descargado."""
    try:
        respuesta = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
        respuesta.raise_for_status()
        modelos = [m["name"] for m in respuesta.json().get("models", [])]
        if OLLAMA_MODEL not in modelos:
            return False, f"El modelo `{OLLAMA_MODEL}` no está descargado (`ollama pull {OLLAMA_MODEL}`)."
        return True, OLLAMA_MODEL
    except Exception:
        return False, f"Ollama no responde en {OLLAMA_HOST}."


def explicar_con_ollama(prompt: str) -> tuple:
    disponible, detalle = ollama_disponible()
    if not disponible:
        return False, detalle
    try:
        respuesta = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2, "num_predict": 600},
            },
            timeout=180,
        )
        respuesta.raise_for_status()
        texto = respuesta.json().get("response", "").strip()
        return (True, texto) if texto else (False, "El modelo devolvió una respuesta vacía.")
    except requests.Timeout:
        return False, f"`{OLLAMA_MODEL}` superó el tiempo de espera. Prueba un modelo más liviano."
    except Exception as exc:
        return False, f"Error consultando Ollama ({OLLAMA_MODEL}): {exc}"


def tab_scoring():
    st.subheader("Scoring de riesgo con explicación asistida")

    artefacto = cargar_modelos()
    if not artefacto or "credit" not in artefacto or not artefacto["credit"]:
        st.warning("No hay modelos entrenados disponibles.")
        st.info("Entrénalos con:  `python models/ml_perabanck_official.py`")
        return

    credit = artefacto["credit"]
    metricas = credit["metricas"]
    features = credit["features"]

    m1, m2, m3 = st.columns(3)
    m1.metric("ROC-AUC (mora)", f"{metricas['roc_auc']:.3f}")
    m2.metric("PR-AUC (mora)", f"{metricas['pr_auc']:.3f}",
              delta=f"baseline {metricas['baseline_pr']:.3f}")
    fraude = artefacto.get("fraud", {}).get("metricas", {})
    m3.metric("ROC-AUC (fraude tarjeta)", f"{fraude.get('roc_auc', float('nan')):.3f}")

    st.divider()
    st.markdown("### Perfil a evaluar")

    entrada = {}
    columnas = st.columns(3)
    for i, campo in enumerate(features["categoricas"]):
        opciones = credit["categorias"].get(campo, [])
        entrada[campo] = columnas[i % 3].selectbox(campo.replace("_", " ").title(), opciones)

    columnas = st.columns(3)
    binarias = {"tiene_hipoteca", "tiene_prestamo_personal", "suscrito_deposito"}
    for i, campo in enumerate(features["numericas"]):
        etiqueta = campo.replace("_", " ").title()
        base = credit["defaults"].get(campo, 0.0)
        if campo in binarias:
            entrada[campo] = 1 if columnas[i % 3].checkbox(etiqueta, value=bool(base)) else 0
        else:
            entrada[campo] = columnas[i % 3].number_input(etiqueta, value=float(base))

    if not st.button("Calcular riesgo", type="primary", use_container_width=True):
        return

    X = pd.DataFrame([entrada])
    try:
        prob_mora = float(credit["pipeline"].predict_proba(X)[0][1])
    except Exception as exc:
        st.error(f"No se pudo evaluar el perfil: {exc}")
        return

    if prob_mora >= 0.5:
        clase, color = "ALTO", "inverse"
    elif prob_mora >= 0.2:
        clase, color = "MEDIO", "off"
    else:
        clase, color = "BAJO", "normal"

    r1, r2, r3 = st.columns(3)
    r1.metric("Clase de riesgo", clase, delta=f"{prob_mora:.1%} probabilidad de mora",
              delta_color=color)
    r2.metric("Probabilidad de mora", f"{prob_mora:.2%}")
    r3.metric("Umbral de decisión", "20% / 50%", delta="revisar / rechazar")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_mora * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#38bdf8"},
            "steps": [
                {"range": [0, 20], "color": "#14532d"},
                {"range": [20, 50], "color": "#78350f"},
                {"range": [50, 100], "color": "#7f1d1d"},
            ],
        },
        title={"text": "Probabilidad de mora"},
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=280, margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

    importancias = pd.Series(credit["importancias"]).sort_values(ascending=False).head(8)
    st.markdown("**Factores más influyentes del modelo (importancia global)**")
    st.bar_chart(importancias)

    st.divider()
    st.markdown("### Explicación ejecutiva")
    prompt = f"""Eres un analista de riesgo crediticio de un banco colombiano.
Redacta una explicación ejecutiva breve (máximo 4 párrafos, en español) de esta decisión.

Perfil evaluado: {json.dumps(entrada, ensure_ascii=False, default=str)}
Probabilidad de mora estimada por el modelo: {prob_mora:.2%}
Clase de riesgo asignada: {clase}
Desempeño del modelo: ROC-AUC {metricas['roc_auc']:.3f}, PR-AUC {metricas['pr_auc']:.3f}
sobre una base con {metricas['baseline_pr']:.2%} de casos de mora.

Explica los factores que sustentan la decisión, menciona la incertidumbre del modelo
dado el desbalance de clases, y cierra con una recomendación accionable."""

    with st.spinner(f"Generando explicación con {OLLAMA_MODEL} (local)..."):
        ok, texto = explicar_con_ollama(prompt)
    if ok:
        st.markdown(texto)
    else:
        st.info(f"Explicación automática no disponible. {texto}")
        st.markdown(
            f"**Lectura sin IA:** el perfil obtuvo una probabilidad de mora de "
            f"**{prob_mora:.2%}**, lo que lo ubica en riesgo **{clase}**. El modelo "
            f"alcanza ROC-AUC {metricas['roc_auc']:.3f} sobre una base con apenas "
            f"{metricas['baseline_pr']:.2%} de casos de mora, así que conviene usarlo "
            "como apoyo a la decisión y no como criterio único."
        )


# ---------------------------------------------------------------------------

def main():
    st.title("🏦 PeraBank — Risk & Market Intelligence")
    st.caption("Warehouse Gold en modelo copo de nieve · Parquet consultado con DuckDB")

    if not exigir_warehouse():
        return

    with st.sidebar:
        st.header("Estado del warehouse")
        tablas = tablas_disponibles()
        st.metric("Tablas disponibles", len(tablas))
        st.metric("Tamaño en Parquet", f"{warehouse.size_bytes() / 1e6:,.0f} MB")
        st.write("**Modelos ML:**", "✅ cargados" if cargar_modelos() else "⚠️ sin entrenar")
        st.write("**Segmentación:**", "✅ cargada" if cargar_clustering() else "⚠️ sin generar")
        listo, detalle = ollama_disponible()
        st.write("**Ollama (local):**", f"✅ {detalle}" if listo else "⚠️ no disponible")
        if not listo:
            st.caption(detalle)
        if st.button("Limpiar caché de consultas"):
            st.cache_data.clear()
            st.rerun()

    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Portafolio", "📈 Tesorería y mercado", "🏦 Tasas bancarias",
        "🏛️ Factoring estatal", "🎯 Segmentación y personas", "🛡️ Scoring de riesgo",
    ])
    with t1:
        tab_portafolio()
    with t2:
        tab_mercado()
    with t3:
        tab_tasas_bancarias()
    with t4:
        tab_factoring()
    with t5:
        tab_segmentacion()
    with t6:
        tab_scoring()


if __name__ == "__main__":
    main()
