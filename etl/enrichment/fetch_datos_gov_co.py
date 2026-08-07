"""Ingesta de 4 conjuntos de datos financieros públicos desde datos.gov.co (Socrata API).

Aterriza datos crudos en bronze:
1. TRM Oficial (32sa-8pi3) -> bronze/enrichment_trm_gov.parquet
2. Tasas Activas de Crédito (w9zh-vetq) -> bronze/enrichment_tasas_activas.parquet
3. Tasas de Captación Fondeo (axk9-g2nh) -> bronze/enrichment_tasas_captacion.parquet
4. SECOP II Contratos (jbjy-vk9h) -> bronze/enrichment_secop_contratos.parquet
"""
from datetime import datetime, timezone
import json
import urllib.request
from typing import Dict, List
from urllib.parse import urlencode

import pandas as pd

from etl.common.config import BRONZE_DIR
from etl.common.logging_utils import get_logger

logger = get_logger(__name__)

# max_rows es un tope explícito por dataset, no un límite de página escondido:
# TRM trae histórico diario completo (pocos miles de filas en total). SECOP II es
# nacional y tiene millones de contratos históricos, así que se acota a los más
# recientes en vez de traer todo; tasas activas/captación son reportes periódicos
# por entidad, un corte reciente ya es representativo para el caso de uso actual.
SOCRATA_ENDPOINTS: Dict[str, Dict] = {
    "enrichment_trm_gov": {
        "dataset_id": "32sa-8pi3",
        "order_by": "vigenciadesde",
        "max_rows": 10000,
    },
    "enrichment_tasas_activas": {
        "dataset_id": "w9zh-vetq",
        "order_by": "fecha_corte",
        "max_rows": 5000,
    },
    "enrichment_tasas_captacion": {
        "dataset_id": "axk9-g2nh",
        "order_by": "fechacorte",
        "max_rows": 5000,
    },
    "enrichment_secop_contratos": {
        "dataset_id": "jbjy-vk9h",
        # El dataset NO expone 'fecha_de_firma'; la fecha mejor poblada es
        # fecha_de_fin_del_contrato (~87% no nula). Ordenar por un campo inexistente
        # devolvía un orden arbitrario y dejaba la columna de fecha 100% nula.
        "order_by": "fecha_de_fin_del_contrato",
        # Sin este filtro el corte más reciente son contratos en Borrador/Cancelado
        # con valor_pagado = 0 en el 100% de las filas, lo que hace imposible calcular
        # cualquier ratio de ejecución de pago (base del scoring de factoring).
        "where": "valor_pagado > 0",
        "max_rows": 20000,
    },
}

PAGE_SIZE = 1000


def fetch_socrata_dataset(source_name: str, config: Dict) -> pd.DataFrame:
    """Descarga datos crudos desde un endpoint Socrata de datos.gov.co, paginando
    con $limit/$offset hasta agotar el dataset o alcanzar max_rows."""
    dataset_id = config["dataset_id"]
    order_by = config["order_by"]
    max_rows = config["max_rows"]
    logger.info("Solicitando datos públicos a datos.gov.co (%s, ID %s, hasta %s filas)...",
                source_name, dataset_id, max_rows)

    rows: List[dict] = []
    offset = 0
    try:
        while offset < max_rows:
            page_limit = min(PAGE_SIZE, max_rows - offset)
            params = {
                "$limit": page_limit,
                "$offset": offset,
                "$order": f"{order_by} DESC",
            }
            if config.get("where"):
                params["$where"] = config["where"]
            url = f"https://www.datos.gov.co/resource/{dataset_id}.json?{urlencode(params)}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                page = json.loads(resp.read().decode("utf-8"))
            if not page:
                break
            rows.extend(page)
            offset += len(page)
            if len(page) < page_limit:
                break
    except Exception as e:
        logger.warning("No se pudo obtener datos de %s (%s) en offset %s: %s", source_name, dataset_id, offset, e)

    df = pd.DataFrame(rows)
    df["source_system"] = source_name
    df["ingested_at"] = datetime.now(timezone.utc).isoformat()
    return df


def main() -> Dict[str, pd.DataFrame]:
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    results = {}
    for source_name, config in SOCRATA_ENDPOINTS.items():
        df = fetch_socrata_dataset(source_name, config)
        out_path = BRONZE_DIR / f"{source_name}.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("bronze/%s: %s filas -> %s", source_name, len(df), out_path)
        results[source_name] = df
    return results


if __name__ == "__main__":
    main()
