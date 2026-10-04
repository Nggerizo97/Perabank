"""Segmentación multi-dominio no supervisada sobre el warehouse Gold de PeraBank.

Tres dominios se agrupan por separado, en su grano nativo. NO se fuerzan joins entre
ellos porque no comparten llave natural: un proveedor del SECOP y un cliente de
campañas no son la misma entidad, y unirlos fabricaría una relación inexistente.

  A. segment_retail_customers   -> dim_cliente + fact_campana_marcado  (grano: cliente)
  B. segment_state_suppliers    -> dim_proveedor_estatal + fact_contrato_estatal
                                   (grano: proveedor, agregando sus contratos)
  C. segment_transaction_profiles -> fact_fraude_tarjeta (grano: evento de tarjeta)

Mitigación de sesgo: la matriz de features es exclusivamente financiera/conductual.
Los atributos protegidos se excluyen de la generación de clusters y solo se usan
DESPUÉS, para auditar si el algoritmo los reconstruyó implícitamente.

Las asignaciones se escriben como tablas propias (cluster_<dominio>.parquet) junto
a gold, nunca dentro de las tablas del ETL: reconstruir gold no las borra y un
SELECT * sobre un hecho no las arrastra como feature a otro modelo.
"""
from datetime import datetime, timezone
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, MiniBatchKMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, RobustScaler, StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from etl.common import warehouse  # noqa: E402  (requiere REPO_ROOT en sys.path)
from etl.common.config import GOLD_DIR  # noqa: E402

ARTIFACT_PATH = REPO_ROOT / "models" / "perabank_clustering_models_v1.joblib"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("perabank.clustering")

RANDOM_STATE = 42
K_RANGE = range(3, 9)
SILUETA_MUESTRA = 5000

# Atributos que NUNCA entran a la matriz de features (sesgo demográfico) y
# etiquetas/reglas derivadas que provocarían fuga de información.
ATRIBUTOS_PROTEGIDOS = ["genero", "edad", "ubicacion", "estado_civil", "nivel_educativo", "ocupacion"]
COLUMNAS_CON_FUGA = ["tiene_mora", "es_mora", "es_fraude", "spread_tasa_credito", "tasa_oferta_estimada"]


def _verificar_matriz(X: pd.DataFrame, dominio: str) -> None:
    """Falla ruidosamente si un atributo protegido o con fuga entró a las features."""
    prohibidas = set(ATRIBUTOS_PROTEGIDOS + COLUMNAS_CON_FUGA) & set(X.columns)
    if prohibidas:
        raise ValueError(f"[{dominio}] La matriz de features contiene columnas prohibidas: {sorted(prohibidas)}")
    logger.info("[%s] Matriz limpia: %s features, ninguna protegida ni con fuga.", dominio, X.shape[1])


# ---------------------------------------------------------------------------
# Selección de K
# ---------------------------------------------------------------------------

def _codo_por_segunda_derivada(inercias: list, ks: list) -> int:
    """Punto de inflexión de la curva de inercia (máxima segunda derivada discreta)."""
    if len(inercias) < 3:
        return ks[0]
    segunda = np.diff(inercias, n=2)
    return ks[int(np.argmax(segunda)) + 1]


def buscar_k_optimo(X: np.ndarray, dominio: str, usar_minibatch: bool = False) -> tuple:
    """Evalúa K en [3,8] con silueta, Davies-Bouldin, Calinski-Harabasz e inercia.

    El K elegido maximiza un puntaje compuesto que normaliza las tres métricas de
    calidad a [0,1] (Davies-Bouldin invertido, porque menor es mejor). El codo se
    calcula y se reporta, pero no decide por sí solo: es la métrica más subjetiva.
    """
    filas = []
    for k in K_RANGE:
        modelo = (MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=1024)
                  if usar_minibatch else
                  KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10))
        etiquetas = modelo.fit_predict(X)
        if len(np.unique(etiquetas)) < 2:
            continue
        filas.append({
            "k": k,
            "silueta": silhouette_score(X, etiquetas, sample_size=min(SILUETA_MUESTRA, len(X)),
                                        random_state=RANDOM_STATE),
            "davies_bouldin": davies_bouldin_score(X, etiquetas),
            "calinski_harabasz": calinski_harabasz_score(X, etiquetas),
            "inercia": float(modelo.inertia_),
        })

    metricas = pd.DataFrame(filas)

    def _norm(serie, invertir=False):
        rango = serie.max() - serie.min()
        if rango == 0:
            return pd.Series(0.5, index=serie.index)
        escalado = (serie - serie.min()) / rango
        return 1 - escalado if invertir else escalado

    metricas["puntaje"] = (
        _norm(metricas.silueta)
        + _norm(metricas.davies_bouldin, invertir=True)
        + _norm(metricas.calinski_harabasz)
    ) / 3

    k_optimo = int(metricas.loc[metricas.puntaje.idxmax(), "k"])
    k_codo = _codo_por_segunda_derivada(metricas.inercia.tolist(), metricas.k.tolist())

    logger.info("[%s] Métricas por K:\n%s", dominio, metricas.round(4).to_string(index=False))
    logger.info("[%s] K óptimo (compuesto) = %s | K por codo = %s", dominio, k_optimo, k_codo)
    return k_optimo, k_codo, metricas


# ---------------------------------------------------------------------------
# Auditoría de equidad
# ---------------------------------------------------------------------------

def auditar_equidad(df: pd.DataFrame, etiquetas: np.ndarray, atributos: list, dominio: str) -> dict:
    """Compara la composición de cada cluster contra la población.

    Usa la razón de representación (regla de los cuatro quintos): si una categoría
    aparece en un cluster con menos de 0.8x o más de 1.25x su peso poblacional, el
    cluster está sobre/sub-representando ese grupo y queda marcado.
    """
    auditoria = {}
    df = df.assign(_cluster=etiquetas)

    for atributo in atributos:
        if atributo not in df.columns:
            continue
        serie = df[atributo]
        if serie.nunique(dropna=True) < 2:
            auditoria[atributo] = {
                "auditable": False,
                "motivo": f"La fuente reporta un único valor ('{serie.dropna().unique()[:1]}'), no hay variación que auditar.",
            }
            continue

        if pd.api.types.is_numeric_dtype(serie):
            serie = pd.qcut(serie, q=4, duplicates="drop").astype(str)

        base = serie.value_counts(normalize=True)
        tabla = (
            pd.crosstab(df._cluster, serie, normalize="index")
            .reindex(columns=base.index, fill_value=0.0)
        )
        razon = tabla.div(base, axis=1)
        desviacion = float((razon - 1).abs().max().max())

        auditoria[atributo] = {
            "auditable": True,
            "razon_representacion": razon.round(3).to_dict(),
            "desviacion_maxima": round(desviacion, 3),
            "supera_regla_4_5": bool(((razon < 0.8) | (razon > 1.25)).any().any()),
        }
        estado = "REVISAR" if auditoria[atributo]["supera_regla_4_5"] else "OK"
        logger.info("[%s] Equidad '%s': desviación máx %.2fx -> %s", dominio, atributo, desviacion, estado)

    return auditoria


# ---------------------------------------------------------------------------
# Dominio A — clientes retail
# ---------------------------------------------------------------------------

FEATURES_RETAIL = ["balance_usd", "duracion_contacto_seg", "tiene_hipoteca",
                   "tiene_prestamo_personal", "suscrito_deposito"]


def segment_retail_customers() -> dict:
    logger.info("=" * 78)
    logger.info("DOMINIO A — Segmentación de clientes retail")

    df = warehouse.query(f"""
        SELECT f.sk_cliente, {', '.join('f.' + c for c in FEATURES_RETAIL)},
               c.genero, c.ubicacion, c.edad, c.estado_civil, c.nivel_educativo, c.ocupacion
        FROM fact_campana_marcado f
        JOIN dim_cliente c ON c.sk_cliente = f.sk_cliente
        ORDER BY f.id_campana_contacto
    """)
    if df.empty:
        logger.warning("Sin datos para el dominio retail.")
        return {}

    X = df[FEATURES_RETAIL].copy()
    _verificar_matriz(X, "retail")

    preproceso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)),
    ])
    Z = preproceso.fit_transform(X)
    logger.info("[retail] PCA conservó %s componentes para el 90%% de varianza (%.1f%% real).",
                Z.shape[1], preproceso.named_steps["pca"].explained_variance_ratio_.sum() * 100)

    k, k_codo, metricas = buscar_k_optimo(Z, "retail")
    modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    etiquetas = modelo.fit_predict(Z)

    auditoria = auditar_equidad(df, etiquetas, ATRIBUTOS_PROTEGIDOS, "retail")
    perfiles = _perfilar(df, X, etiquetas, FEATURES_RETAIL)
    nombres = _nombrar_retail(perfiles)

    return {
        "dominio": "retail",
        "clave": "sk_cliente",
        "columna_cluster": "sk_cluster_retail",
        "tabla_destino": "dim_cliente",
        "features": FEATURES_RETAIL,
        "preproceso": preproceso,
        "modelo": modelo,
        "pca_viz": _pca_visualizacion(preproceso, X),
        "k": k,
        "k_codo": k_codo,
        "metricas": metricas.to_dict("records"),
        "metricas_finales": _metricas_finales(Z, etiquetas),
        "perfiles": perfiles.to_dict(),
        "nombres": nombres,
        "auditoria_equidad": auditoria,
        "asignaciones": pd.DataFrame({"clave": df.sk_cliente, "cluster": etiquetas}),
    }


def _nombrar_retail(perfiles: pd.DataFrame) -> dict:
    """Compone el nombre de cada persona a partir de patrimonio, mix de productos y
    conversión. Al componer en vez de encadenar if/elif, cada cluster obtiene una
    etiqueta descriptiva y distinta sin depender de un orden arbitrario de reglas."""
    nombres = {}
    for cluster, fila in perfiles.iterrows():
        patrimonio = _tramo(perfiles["balance_usd"], fila["balance_usd"],
                            ["Patrimonio bajo", "Patrimonio medio", "Alto patrimonio"])
        if fila["tiene_prestamo_personal"] >= 0.5 and fila["tiene_hipoteca"] >= 0.5:
            producto = "doble apalancamiento"
        elif fila["tiene_prestamo_personal"] >= 0.5:
            producto = "préstamo personal"
        elif fila["tiene_hipoteca"] >= 0.5:
            producto = "hipoteca"
        else:
            producto = "sin crédito"
        conversion = "convierte" if fila["suscrito_deposito"] >= 0.5 else "no convierte"
        engagement = _tramo(perfiles["duracion_contacto_seg"], fila["duracion_contacto_seg"],
                            ["contacto breve", "contacto medio", "contacto extenso"])
        nombres[int(cluster)] = f"{patrimonio} · {producto} · {conversion} · {engagement}"
    return _desambiguar(nombres)


def _tramo(serie: pd.Series, valor: float, etiquetas: list) -> str:
    """Ubica un valor en tramos por cuantiles de la serie de perfiles."""
    if serie.nunique() <= 1:
        return etiquetas[len(etiquetas) // 2]
    cortes = np.quantile(serie, np.linspace(0, 1, len(etiquetas) + 1)[1:-1])
    return etiquetas[int(np.searchsorted(cortes, valor, side="right"))]


# ---------------------------------------------------------------------------
# Dominio B — proveedores del Estado
# ---------------------------------------------------------------------------

FEATURES_PROVEEDOR = ["valor_del_contrato", "valor_pagado", "ratio_desembolso",
                      "contratos_adjudicados", "es_pyme"]

# Columnas de cola pesada que necesitan log. El ratio y es_pyme ya están acotados a [0,1].
# Se usa np.log1p directamente (y no una función propia) porque joblib la serializa por
# referencia: una función definida aquí quedaría anclada a __main__ y el artefacto
# resultaría imposible de cargar desde la app de Streamlit.
_IDX_LOG_PROVEEDOR = [0, 1, 3]


def segment_state_suppliers() -> dict:
    logger.info("=" * 78)
    logger.info("DOMINIO B — Segmentación de proveedores estatales")

    df = warehouse.query("""
        SELECT p.sk_proveedor, p.proveedor_adjudicado, p.es_pyme,
               SUM(f.valor_del_contrato) AS valor_del_contrato,
               SUM(f.valor_pagado)       AS valor_pagado,
               COUNT(*)                  AS contratos_adjudicados
        FROM fact_contrato_estatal f
        JOIN dim_proveedor_estatal p ON p.sk_proveedor = f.sk_proveedor
        WHERE f.valor_del_contrato > 0
        GROUP BY p.sk_proveedor, p.proveedor_adjudicado, p.es_pyme
        ORDER BY p.sk_proveedor
    """)
    if df.empty:
        logger.warning("Sin datos para el dominio de proveedores.")
        return {}

    df["ratio_desembolso"] = (df.valor_pagado / df.valor_del_contrato).clip(0, 1)

    X = df[FEATURES_PROVEEDOR].copy()
    _verificar_matriz(X, "proveedores")

    # El valor contratado abarca 7 órdenes de magnitud (1e5 a 2e12). Sin log, el
    # escalado deja un único cluster con el 99.7% de proveedores y cuatro clusters
    # de 1-22 megacontratos: la silueta sube a 0.99 pero no segmenta nada útil.
    # En escala log los tramos de tamaño se vuelven comparables entre sí.
    preproceso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", ColumnTransformer(
            [("log", FunctionTransformer(np.log1p), _IDX_LOG_PROVEEDOR)],
            remainder="passthrough",
        )),
        ("scaler", RobustScaler()),
    ])
    Z = preproceso.fit_transform(X)

    k, k_codo, metricas = buscar_k_optimo(Z, "proveedores")
    modelo = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    etiquetas = modelo.fit_predict(Z)

    # DBSCAN complementario: no reemplaza a KMeans, sirve para cuantificar cuántos
    # proveedores son atípicos (megacontratos) y no encajan en ninguna densidad.
    ruido = DBSCAN(eps=1.5, min_samples=10).fit_predict(Z)
    n_ruido = int((ruido == -1).sum())
    logger.info("[proveedores] DBSCAN marcó %s proveedores como atípicos (%.2f%%).",
                n_ruido, 100 * n_ruido / len(df))

    perfiles = _perfilar(df, X, etiquetas, FEATURES_PROVEEDOR)
    auditoria = auditar_equidad(df, etiquetas, ["es_pyme"], "proveedores")

    return {
        "dominio": "proveedores",
        "clave": "sk_proveedor",
        "columna_cluster": "sk_cluster_supplier",
        "tabla_destino": "dim_proveedor_estatal",
        "features": FEATURES_PROVEEDOR,
        "preproceso": preproceso,
        "modelo": modelo,
        "pca_viz": _pca_visualizacion(preproceso, X),
        "k": k,
        "k_codo": k_codo,
        "metricas": metricas.to_dict("records"),
        "metricas_finales": _metricas_finales(Z, etiquetas),
        "perfiles": perfiles.to_dict(),
        "nombres": _nombrar_proveedores(perfiles),
        "auditoria_equidad": auditoria,
        "outliers_dbscan": n_ruido,
        "asignaciones": pd.DataFrame({"clave": df.sk_proveedor, "cluster": etiquetas}),
    }


def _nombrar_proveedores(perfiles: pd.DataFrame) -> dict:
    """Persona = tramo de tamaño contratado + ejecución de pago + recurrencia."""
    nombres = {}
    for cluster, fila in perfiles.iterrows():
        tamano = _tramo(perfiles["valor_del_contrato"], fila["valor_del_contrato"],
                        ["Contratación pequeña", "Contratación media",
                         "Contratación alta", "Gran contratación"])
        if fila["ratio_desembolso"] >= 0.6:
            pago = "desembolso alto"
        elif fila["ratio_desembolso"] >= 0.35:
            pago = "desembolso parcial"
        else:
            pago = "desembolso rezagado"
        recurrencia = "recurrente" if fila["contratos_adjudicados"] >= 1.8 else "ocasional"
        marca_pyme = " · PYME" if fila["es_pyme"] >= 0.5 else ""
        nombres[int(cluster)] = f"{tamano} · {pago} · {recurrencia}{marca_pyme}"
    return _desambiguar(nombres)


# ---------------------------------------------------------------------------
# Dominio C — perfiles transaccionales
# ---------------------------------------------------------------------------

def segment_transaction_profiles() -> dict:
    logger.info("=" * 78)
    logger.info("DOMINIO C — Segmentación de perfiles transaccionales")

    df = warehouse.query("SELECT * FROM fact_fraude_tarjeta")
    if df.empty:
        logger.warning("Sin datos para el dominio transaccional.")
        return {}

    componentes = sorted([c for c in df.columns if c.startswith("V") and c[1:].isdigit()],
                         key=lambda c: int(c[1:]))
    features = componentes + ["monto_usd"]

    X = df[features].copy()
    _verificar_matriz(X, "transaccional")

    preproceso = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    Z = preproceso.fit_transform(X)

    k, k_codo, metricas = buscar_k_optimo(Z, "transaccional", usar_minibatch=True)
    modelo = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10, batch_size=1024)
    etiquetas = modelo.fit_predict(Z)

    # volatilidad_fx_30d queda fuera del perfil: es constante para todos los eventos
    # (un único promedio macro), así que en un radar solo añadiría un eje plano.
    perfiles = _perfilar(df, X, etiquetas,
                         ["monto_usd", "segundos_transcurridos"] + componentes[:6])

    # es_fraude quedó FUERA del entrenamiento: aquí solo se mide, a posteriori, si
    # los clusters concentran fraude sin haberlo visto nunca.
    concentracion = df.assign(_c=etiquetas).groupby("_c")["es_fraude"].agg(["mean", "sum", "size"])
    concentracion.columns = ["tasa_fraude", "fraudes", "eventos"]
    logger.info("[transaccional] Concentración de fraude por cluster (target no usado en el ajuste):\n%s",
                concentracion.round(5).to_string())

    return {
        "dominio": "transaccional",
        "clave": "id_evento_tarjeta",
        "columna_cluster": "sk_cluster_behavior",
        "tabla_destino": "fact_fraude_tarjeta",
        "features": features,
        "preproceso": preproceso,
        "modelo": modelo,
        "pca_viz": _pca_visualizacion(preproceso, X),
        "k": k,
        "k_codo": k_codo,
        "metricas": metricas.to_dict("records"),
        "metricas_finales": _metricas_finales(Z, etiquetas),
        "perfiles": perfiles.to_dict(),
        "nombres": _nombrar_transaccional(perfiles, concentracion),
        "concentracion_fraude": concentracion.to_dict(),
        "auditoria_equidad": {},
        "asignaciones": pd.DataFrame({"clave": df.id_evento_tarjeta, "cluster": etiquetas}),
    }


def _nombrar_transaccional(perfiles: pd.DataFrame, concentracion: pd.DataFrame) -> dict:
    """Persona = tramo de monto + concentración de fraude observada a posteriori.

    La tasa de fraude no participó del ajuste; se usa solo para etiquetar el bucket
    operativo resultante, que es justamente lo que hace accionable la segmentación.
    """
    nombres = {}
    tasa_global = concentracion["fraudes"].sum() / max(concentracion["eventos"].sum(), 1)
    for cluster, fila in perfiles.iterrows():
        monto = _tramo(perfiles["monto_usd"], fila["monto_usd"],
                       ["Micro-volumen", "Volumen medio", "Alto valor"])
        tasa = float(concentracion.loc[cluster, "tasa_fraude"]) if cluster in concentracion.index else 0.0
        lift = tasa / tasa_global if tasa_global > 0 else 0.0
        if lift >= 3:
            riesgo = "riesgo elevado"
        elif lift >= 1:
            riesgo = "riesgo moderado"
        else:
            riesgo = "riesgo bajo"
        nombres[int(cluster)] = f"{monto} · {riesgo} ({lift:.1f}x)"
    return _desambiguar(nombres)


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------

def _desambiguar(nombres: dict) -> dict:
    """Evita que dos clusters compartan el mismo nombre de persona."""
    vistos, salida = {}, {}
    for cluster, nombre in nombres.items():
        vistos[nombre] = vistos.get(nombre, 0) + 1
        salida[cluster] = nombre if vistos[nombre] == 1 else f"{nombre} ({vistos[nombre]})"
    return salida


def _perfilar(df: pd.DataFrame, X: pd.DataFrame, etiquetas: np.ndarray, columnas: list) -> pd.DataFrame:
    """Promedio de cada feature por cluster, en unidades originales (no escaladas)."""
    disponibles = [c for c in columnas if c in df.columns or c in X.columns]
    base = pd.concat([df[[c for c in disponibles if c in df.columns]],
                      X[[c for c in disponibles if c in X.columns and c not in df.columns]]], axis=1)
    perfil = base.assign(_c=etiquetas).groupby("_c").mean(numeric_only=True)
    perfil["n_entidades"] = pd.Series(etiquetas).value_counts().sort_index()
    return perfil


def _metricas_finales(Z: np.ndarray, etiquetas: np.ndarray) -> dict:
    return {
        "silueta": float(silhouette_score(Z, etiquetas, sample_size=min(SILUETA_MUESTRA, len(Z)),
                                          random_state=RANDOM_STATE)),
        "davies_bouldin": float(davies_bouldin_score(Z, etiquetas)),
        "calinski_harabasz": float(calinski_harabasz_score(Z, etiquetas)),
    }


def _pca_visualizacion(preproceso: Pipeline, X: pd.DataFrame):
    """PCA de 3 componentes solo para el scatter 3D; independiente del PCA del pipeline."""
    pasos = [(n, t) for n, t in preproceso.steps if n != "pca"]
    escalado = Pipeline(pasos).fit_transform(X)
    n = min(3, escalado.shape[1])
    pca = PCA(n_components=n, random_state=RANDOM_STATE).fit(escalado)
    return {"pca": pca, "pasos_previos": Pipeline(pasos)}


def escribir_clusters(resultado: dict) -> None:
    """Persiste las asignaciones como tabla cluster_<dominio> (clave, cluster),
    reemplazando la corrida anterior. Se une a su tabla de origen por la clave."""
    if not resultado:
        return
    asignaciones = resultado["asignaciones"]
    tabla = pd.DataFrame({
        resultado["clave"]: asignaciones.clave.to_numpy(),
        resultado["columna_cluster"]: asignaciones.cluster.astype("int64").to_numpy(),
    })
    path = warehouse.write_table(tabla, f"cluster_{resultado['dominio']}")
    logger.info("%s: %s asignaciones -> %s", resultado["columna_cluster"], len(tabla), path)


def main():
    if not warehouse.tables():
        logger.error("No hay tablas gold en %s. Corre primero: python -m etl.run_pipeline", GOLD_DIR)
        return

    resultados = {
        "retail": segment_retail_customers(),
        "proveedores": segment_state_suppliers(),
        "transaccional": segment_transaction_profiles(),
    }

    artefacto = {"metadata": {"entrenado_en": datetime.now(timezone.utc).isoformat(),
                              "fuente": str(GOLD_DIR)}}
    logger.info("=" * 78)
    logger.info("RESUMEN")
    for nombre, resultado in resultados.items():
        if not resultado:
            continue
        escribir_clusters(resultado)
        m = resultado["metricas_finales"]
        logger.info("%-14s K=%s  silueta=%.3f  DB=%.3f  CH=%.0f  %s",
                    nombre, resultado["k"], m["silueta"], m["davies_bouldin"],
                    m["calinski_harabasz"],
                    "✔ supera 0.50" if m["silueta"] > 0.50 else "✘ bajo el objetivo 0.50")
        # El DataFrame de asignaciones no va al artefacto: ya vive en cluster_<dominio>.parquet.
        artefacto[nombre] = {k: v for k, v in resultado.items() if k != "asignaciones"}

    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefacto, ARTIFACT_PATH)
    logger.info("Artefacto guardado: %s (%.2f MB)", ARTIFACT_PATH, ARTIFACT_PATH.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
