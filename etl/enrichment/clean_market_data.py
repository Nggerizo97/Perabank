"""Transformación silver para datos de mercado externos."""
import pandas as pd

from etl.common.silver_base import SilverTransform
from schemas.silver_schemas import SILVER_ENRICHMENT_MARKET_CONTRACT


class EnrichmentMarketClean(SilverTransform):
    source_name = "enrichment_market"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["fecha"]).sort_values("fecha").reset_index(drop=True)
        
        # Formateo explícito de dtypes float64 para tasas de mercado
        for col in ["tasa_usd_inr", "tasa_usd_eur", "tasa_usd_cop", "tasa_ibr_overnight", "tasa_treasury_10y", "tasa_tbill_3m"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Calcular volatilidad FX a 30 días móvil (StdDev del ratio USD/COP)
        df["volatilidad_fx_30d"] = df["tasa_usd_cop"].rolling(window=30, min_periods=1).std().fillna(0.0).round(4)
        
        # Validar columnas contra contrato silver
        SILVER_ENRICHMENT_MARKET_CONTRACT.validate(df)
        return df



def main():
    EnrichmentMarketClean().run()


if __name__ == "__main__":
    main()
