"""Validaciones ligeras de calidad de datos para límites entre capas."""
from typing import List, Dict, Optional
import pandas as pd

from etl.common.logging_utils import get_logger

logger = get_logger(__name__)


def assert_quality(
    df: pd.DataFrame,
    table_name: str,
    layer: str,
    primary_keys: List[str],
    non_null_cols: Optional[List[str]] = None,
    ref_checks: Optional[Dict[str, pd.Series]] = None
) -> None:
    """Verifica nulos, duplicados e integridad referencial en un DataFrame de cualquier capa.

    Args:
        df: DataFrame a validar.
        table_name: Nombre descriptivo de la tabla.
        layer: Nombre de la capa (bronze, silver, gold).
        primary_keys: Lista de columnas que componen la clave primaria / surrogate key.
        non_null_cols: Columnas obligatorias que no pueden tener valores nulos.
        ref_checks: Diccionario de {col_fk: serie_de_pks_validos} para validar integridad referencial.
    """
    logger.info("Iniciando validación de calidad de datos para [%s/%s] (%s filas)...", layer, table_name, len(df))

    # 1. Duplicados en Clave Primaria / Surrogate Key
    if primary_keys and all(pk in df.columns for pk in primary_keys):
        dupes = df.duplicated(subset=primary_keys).sum()
        if dupes > 0:
            msg = f"[%s/%s] ERROR DQ: {dupes} filas duplicadas encontradas en clave primaria {primary_keys}."
            logger.error(msg, layer, table_name)
            raise ValueError(msg % (layer, table_name))

    # 2. Nulos en columnas obligatorias
    cols_to_check = (non_null_cols or []) + (primary_keys or [])
    cols_to_check = list(set([c for c in cols_to_check if c in df.columns]))
    for col in cols_to_check:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            msg = f"[%s/%s] ERROR DQ: Columna obligatoria '{col}' contiene {null_count} valores nulos."
            logger.error(msg, layer, table_name)
            raise ValueError(msg % (layer, table_name))

    # 3. Integridad Referencial
    if ref_checks:
        for fk_col, valid_pks in ref_checks.items():
            if fk_col in df.columns:
                invalid_fks = ~df[fk_col].isin(valid_pks)
                invalid_count = invalid_fks.sum()
                if invalid_count > 0:
                    msg = f"[%s/%s] ERROR DQ: {invalid_count} filas violan integridad referencial en FK '{fk_col}'."
                    logger.error(msg, layer, table_name)
                    raise ValueError(msg % (layer, table_name))

    logger.info("[%s/%s] Validaciones DQ completadas con éxito sin violaciones.", layer, table_name)
