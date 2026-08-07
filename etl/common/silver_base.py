"""Clase base para las transformaciones silver: cada fuente solo implementa clean()."""
from abc import ABC, abstractmethod

import pandas as pd

from etl.common.config import BRONZE_DIR, SILVER_DIR
from etl.common.logging_utils import get_logger

logger = get_logger(__name__)


class SilverTransform(ABC):
    source_name: str

    def read_bronze(self) -> pd.DataFrame:
        return pd.read_parquet(BRONZE_DIR / f"{self.source_name}.parquet")

    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Dedup, tipos y nulos. No renombra hacia el modelo gold, eso es trabajo de gold."""

    def run(self) -> pd.DataFrame:
        df = self.clean(self.read_bronze())

        SILVER_DIR.mkdir(parents=True, exist_ok=True)
        out_path = SILVER_DIR / f"{self.source_name}.parquet"
        df.to_parquet(out_path, index=False)

        logger.info("silver/%s: %s filas -> %s", self.source_name, len(df), out_path)
        return df
