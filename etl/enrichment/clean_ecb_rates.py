"""Transformación silver para datos de referencia de tipos de cambio del BCE."""
import pandas as pd

from etl.common.silver_base import SilverTransform
from schemas.silver_schemas import SILVER_ENRICHMENT_ECB_CONTRACT


class ECBRatesClean(SilverTransform):
    source_name = "enrichment_ecb"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
        
        # Formateo explícito de dtypes numeric float64
        for col in ["tasa_eur_usd", "tasa_eur_cop", "tasa_eur_inr", "tasa_eur_gbp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Validar columnas contra contrato silver
        SILVER_ENRICHMENT_ECB_CONTRACT.validate(df)
        return df


def main():
    ECBRatesClean().run()


if __name__ == "__main__":
    main()
