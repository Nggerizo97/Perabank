"""Transformaciones silver para los 4 conjuntos de datos financieros de datos.gov.co."""
import pandas as pd

from etl.common.silver_base import SilverTransform
from schemas.silver_schemas import (
    SILVER_ENRICHMENT_SECOP_CONTRATOS_CONTRACT,
    SILVER_ENRICHMENT_TASAS_ACTIVAS_CONTRACT,
    SILVER_ENRICHMENT_TASAS_CAPTACION_CONTRACT,
    SILVER_ENRICHMENT_TRM_GOV_CONTRACT,
)


class TRMGovClean(SilverTransform):
    source_name = "enrichment_trm_gov"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop_duplicates(subset=["vigenciadesde"]).sort_values("vigenciadesde", ascending=False).reset_index(drop=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        SILVER_ENRICHMENT_TRM_GOV_CONTRACT.validate(df)
        return df


class TasasActivasClean(SilverTransform):
    source_name = "enrichment_tasas_activas"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        # Asegurar columnas requeridas
        cols = [c.name for c in SILVER_ENRICHMENT_TASAS_ACTIVAS_CONTRACT.columns]
        for col in cols:
            if col not in df.columns:
                df[col] = None

        # Nota: (fecha_corte, nombre_entidad, tipo_de_cr_dito) NO es llave única, el mismo
        # trío tiene múltiples filas reales por tamaño de empresa, plazo y rango de monto.
        # Se deduplica por fila completa, que es lo único que realmente identifica un duplicado.
        df["tasa_efectiva_promedio"] = pd.to_numeric(df["tasa_efectiva_promedio"], errors="coerce")
        df["margen_adicional_a_la"] = pd.to_numeric(df["margen_adicional_a_la"], errors="coerce")
        df["montos_desembolsados"] = pd.to_numeric(df["montos_desembolsados"], errors="coerce")
        df["numero_de_creditos"] = pd.to_numeric(df["numero_de_creditos"], errors="coerce")

        df = df[cols].drop_duplicates().reset_index(drop=True)
        SILVER_ENRICHMENT_TASAS_ACTIVAS_CONTRACT.validate(df)
        return df


class TasasCaptacionClean(SilverTransform):
    source_name = "enrichment_tasas_captacion"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c.name for c in SILVER_ENRICHMENT_TASAS_CAPTACION_CONTRACT.columns]
        for col in cols:
            if col not in df.columns:
                df[col] = None

        df["tasa"] = pd.to_numeric(df["tasa"], errors="coerce")
        df["monto"] = pd.to_numeric(df["monto"], errors="coerce")

        # Nota: (fechacorte, nombreentidad, descripcion) NO es llave única, la misma
        # combinación tiene múltiples filas reales por uca/subcuenta. Se deduplica por
        # fila completa, que es lo único que realmente identifica un duplicado.
        df = df[cols].drop_duplicates().reset_index(drop=True)
        SILVER_ENRICHMENT_TASAS_CAPTACION_CONTRACT.validate(df)
        return df


class SECOPContratosClean(SilverTransform):
    source_name = "enrichment_secop_contratos"

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c.name for c in SILVER_ENRICHMENT_SECOP_CONTRATOS_CONTRACT.columns]
        for col in cols:
            if col not in df.columns:
                df[col] = None

        df["valor_del_contrato"] = pd.to_numeric(df["valor_del_contrato"], errors="coerce")
        df["valor_pagado"] = pd.to_numeric(df["valor_pagado"], errors="coerce")

        df = df[cols].drop_duplicates(subset=["id_contrato"]).reset_index(drop=True)
        SILVER_ENRICHMENT_SECOP_CONTRATOS_CONTRACT.validate(df)
        return df


ALL_GOV_TRANSFORMS = [TRMGovClean, TasasActivasClean, TasasCaptacionClean, SECOPContratosClean]


def main():
    for transform_cls in ALL_GOV_TRANSFORMS:
        transform_cls().run()


if __name__ == "__main__":
    main()
