"""Carga de silver para la capa gold, con muestreo determinístico compartido.

Dimensiones y hechos DEBEN leer la misma muestra: si dim_cliente se construyera
sobre una muestra distinta a la de fact_transaccion, las FK apuntarían a clientes
inexistentes. Por eso el muestreo vive aquí, cacheado por corrida, y no dentro de
cada builder.
"""
from functools import lru_cache
import hashlib

import pandas as pd

from etl.common.config import SILVER_DIR

RANDOM_STATE = 42

# Topes de muestreo para ejecución local. None = tabla completa.
SAMPLE_SIZES = {
    "paysim": 50000,
    "creditcard": 30000,
    "enrichment_secop_contratos": 10000,
}

# Miembro "DESCONOCIDO" de Kimball: permite que un hecho sin entidad/tipo/fecha
# conocida mantenga integridad referencial sin inventar una relación ni borrar la fila.
UNKNOWN_LABEL = "DESCONOCIDO"
SK_FECHA_DESCONOCIDA = -1

# Rango soportado por dim_fecha. Arranca en 1991 porque la TRM oficial se publica
# desde 1991-12-02, y llega a 2056 por contratos SECOP de muy largo plazo. Fechas
# fuera de este rango (errores de captura en fuentes públicas, p.ej. un contrato
# con fin en 2133) se enrutan al miembro DESCONOCIDO en vez de estirar la dimensión.
DIM_FECHA_INICIO = "1991-01-01"
DIM_FECHA_FIN = "2056-12-31"


def generate_surrogate_key(source_system: str, natural_id: str) -> str:
    """Genera una clave sustituta (surrogate key) determinística mediante hash MD5."""
    text = f"{source_system}::{str(natural_id).strip()}"
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def map_surrogate_keys(series: pd.Series, source_system: str) -> pd.Series:
    """Mapea una serie de IDs naturales a surrogate keys hasheando solo los valores
    únicos (no una vez por fila) y resolviendo el resto con Series.map vectorizado."""
    lookup = {value: generate_surrogate_key(source_system, value) for value in series.dropna().unique()}
    return series.map(lookup)


def unknown_key(dimension: str) -> str:
    """Surrogate key del miembro DESCONOCIDO de una dimensión."""
    return generate_surrogate_key(dimension, UNKNOWN_LABEL)


@lru_cache(maxsize=None)
def load_silver(name: str) -> pd.DataFrame:
    """Lee una tabla silver aplicando su tope de muestreo, una sola vez por corrida."""
    path = SILVER_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    sample_size = SAMPLE_SIZES.get(name)
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=RANDOM_STATE)
    return df.reset_index(drop=True)
