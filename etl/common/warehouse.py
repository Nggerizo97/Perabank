"""Acceso de solo lectura al warehouse gold.

Gold vive únicamente en Parquet (data/gold/*.parquet). DuckDB expone cada archivo
como una vista con el nombre de su tabla y consulta los Parquet en sitio: no hay una
segunda copia de los datos en otra base ni un paso de exportación que mantener.

DuckDB conserva el orden de las filas en lecturas simples, pero NO tras un JOIN o un
GROUP BY (ejecuta en paralelo). Toda consulta que alimente un modelo con esas
operaciones lleva ORDER BY explícito: KMeans y train_test_split dependen del orden
de las filas aunque fijen random_state.
"""
from pathlib import Path

import duckdb
import pandas as pd

from etl.common.config import GOLD_DIR


def connect() -> duckdb.DuckDBPyConnection:
    """Conexión en memoria con una vista por cada tabla gold. Crear las vistas solo
    lee metadatos del Parquet, así que abrir una conexión por consulta es barato."""
    con = duckdb.connect()
    for path in _parquet_files():
        ruta = path.as_posix().replace("'", "''")
        con.execute(f"CREATE VIEW \"{path.stem}\" AS SELECT * FROM read_parquet('{ruta}')")
    return con


def query(sql: str, params: tuple | list = ()) -> pd.DataFrame:
    with connect() as con:
        return con.execute(sql, list(params)).df()


def tables() -> set[str]:
    return {path.stem for path in _parquet_files()}


def size_bytes() -> int:
    return sum(path.stat().st_size for path in _parquet_files())


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Persiste una tabla derivada (p.ej. asignaciones de cluster) junto a gold."""
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLD_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def _parquet_files() -> list[Path]:
    return sorted(GOLD_DIR.glob("*.parquet")) if GOLD_DIR.exists() else []
