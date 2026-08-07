"""Ingesta de tasas de cambio de referencia del Banco Central Europeo (BCE / ECB).

Aterriza los datos crudos en bronze/enrichment_ecb.parquet.
Fuente: Frankfurter API (api.frankfurter.app), tasas oficiales públicas del BCE.
"""
from datetime import datetime, timedelta, timezone
import json
import urllib.request

import pandas as pd

from etl.common.config import BRONZE_DIR
from etl.common.logging_utils import get_logger

logger = get_logger(__name__)

FRANKFURTER_BASE_URL = "https://api.frankfurter.app"


def fetch_ecb_fx_history(days: int = 400) -> pd.DataFrame:
    """Obtiene el histórico diario real de tipos de cambio del BCE desde Frankfurter API."""
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=days)
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")
    
    url = f"{FRANKFURTER_BASE_URL}/{start_str}..{end_str}?from=EUR&to=USD,COP,INR,GBP"
    logger.info("Solicitando histórico de tipos de cambio BCE a Frankfurter API (%s..%s)...", start_str, end_str)

    rows = []
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "PeraBank-ETL/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            rates_by_date = data.get("rates", {})
            for date_str, rates in rates_by_date.items():
                rows.append({
                    "fecha": date_str,
                    "tasa_eur_usd": float(rates.get("USD")) if rates.get("USD") is not None else None,
                    "tasa_eur_cop": float(rates.get("COP")) if rates.get("COP") is not None else None,
                    "tasa_eur_inr": float(rates.get("INR")) if rates.get("INR") is not None else None,
                    "tasa_eur_gbp": float(rates.get("GBP")) if rates.get("GBP") is not None else None,
                })
    except Exception as e:
        logger.warning("No se pudo obtener datos de Frankfurter API (%s). Generando conjunto vacío de respuesta.", e)

    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["fecha", "tasa_eur_usd", "tasa_eur_cop", "tasa_eur_inr", "tasa_eur_gbp"])
    
    df["source_system"] = "enrichment_ecb"
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return df


def main() -> pd.DataFrame:
    df = fetch_ecb_fx_history(days=400)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DIR / "enrichment_ecb.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("bronze/enrichment_ecb: %s filas -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    main()
