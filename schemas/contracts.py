"""Utilidades de validación de esquemas y contratos de datos."""
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd

from etl.common.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnContract:
    name: str
    dtype: str
    nullable: bool = True
    description: str = ""


@dataclass
class TableContract:
    table_name: str
    layer: str  # bronze, silver, gold
    columns: List[ColumnContract]
    primary_keys: List[str]
    foreign_keys: Optional[Dict[str, str]] = None  # col_name -> target_table.target_col

    def get_column_names(self) -> List[str]:
        return [c.name for c in self.columns]

    def validate(self, df: pd.DataFrame) -> bool:
        """Verifica que un DataFrame cumpla los nombres de columna del contrato.

        Falla ruidosamente: un contrato que solo loguea y deja pasar la tabla no es
        un contrato. Si falta una columna, la corrida se detiene aquí y no propaga
        datos incompletos a las capas siguientes.
        """
        missing = [c.name for c in self.columns if c.name not in df.columns]
        if missing:
            raise ValueError(
                f"Tabla '{self.table_name}' en capa {self.layer} no cumple el contrato. "
                f"Columnas faltantes: {missing}"
            )
        logger.info("Tabla '%s' (%s) validada exitosamente contra contrato (%s columnas).",
                    self.table_name, self.layer, len(self.columns))
        return True
