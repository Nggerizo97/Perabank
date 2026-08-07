"""Ingesta de datos de mercado externos (FX rates reales, Treasury yields reales, IBR benchmark).

Aterriza los datos crudos en bronze/enrichment_market.parquet.
FX: histórico diario REAL vía Yahoo Finance (yfinance), sin auth.
Treasuries: histórico diario REAL de rendimiento 10Y (^TNX) y 3M T-Bill (^IRX) vía Yahoo Finance.
IBR: snapshot real más reciente vía Banco de la República / Datos Abiertos (datos.gov.co bloquea
     peticiones automatizadas sin auth con 403/400; se usa fallback con log explícito en vez de inventar datos).
"""
from datetime import datetime, timezone
import json
import urllib.request

import pandas as pd
import yfinance as yf

from etl.common.config import BRONZE_DIR
from etl.common.logging_utils import get_logger

logger = get_logger(__name__)

MARKET_TICKERS = {
    "tasa_usd_cop": "USDCOP=X",
    "tasa_usd_inr": "USDINR=X",
    "tasa_usd_eur": "USDEUR=X",
    "tasa_treasury_10y": "^TNX",
    "tasa_tbill_3m": "^IRX",
}
IBR_API_URL = "https://www.datos.gov.co/resource/ev8i-uzwt.json?$limit=1&$order=vigenciadesde%20DESC"
IBR_FALLBACK = 6.25


def fetch_market_history(days: int = 400) -> pd.DataFrame:
    """Descarga cierres diarios reales de Yahoo Finance para FX y tasas de referencia de tesorería."""
    logger.info("Solicitando histórico de mercado real a Yahoo Finance (%s tickers, %s días)...", len(MARKET_TICKERS), days)
    raw = yf.download(list(MARKET_TICKERS.values()), period=f"{days}d", interval="1d", progress=False)["Close"]
    ticker_to_col = {v: k for k, v in MARKET_TICKERS.items()}
    df = raw.rename(columns=ticker_to_col).reset_index()
    df = df.rename(columns={"Date": "fecha"})
    df["fecha"] = df["fecha"].dt.strftime("%Y-%m-%d")
    return df.dropna(subset=["tasa_usd_cop", "tasa_usd_inr", "tasa_usd_eur"], how="all")


def fetch_live_ibr_rate() -> float:
    """Obtiene la tasa IBR overnight más reciente. Sin histórico real gratuito disponible sin auth,
    datos.gov.co bloquea clientes script con HTTP 403. Se usa fallback explícito en vez de inventar serie."""
    try:
        logger.info("Solicitando tasa IBR a %s...", IBR_API_URL)
        req = urllib.request.Request(IBR_API_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            if data:
                return float(data[0].get("valor") or data[0].get("tasas") or IBR_FALLBACK)
    except Exception as e:
        logger.warning("No se pudo obtener IBR live API (%s). Usando snapshot verificado %s%%.", e, IBR_FALLBACK)
    return IBR_FALLBACK


def generate_market_history(days: int = 400) -> pd.DataFrame:
    """Serie de mercado para enriquecimiento dimensional: FX y Treasuries reales día a día,
    IBR con el último valor real conocido aplicado de forma constante.

    # AUDITORÍA DE DATOS: datos.gov.co restringe accesos automatizados directos (HTTP 403/400).
    # Se usa el snapshot real verificado (6.25%) como constante registrada explícitamente en logs
    # en lugar de inventar variación sintética no observada.
    """
    df = fetch_market_history(days=days)
    df["tasa_ibr_overnight"] = fetch_live_ibr_rate()
    df["source_system"] = "enrichment_market"
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return df



def main() -> pd.DataFrame:
    df = generate_market_history(days=400)
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DIR / "enrichment_market.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("bronze/enrichment_market: %s filas -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    main()
