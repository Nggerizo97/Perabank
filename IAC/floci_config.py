"""Configuración compartida de clientes boto3 contra el emulador local (Floci).

SALVAGUARDA CRÍTICA: el .env del proyecto contiene credenciales AWS reales. Si un
script de IaC se ejecutara sin endpoint local, boto3 las tomaría de la cadena de
credenciales y crearía infraestructura REAL, con costo real. Por eso aquí se
inyectan credenciales ficticias de forma explícita y se rechaza cualquier endpoint
que no sea local antes de construir un cliente.
"""
import os
import sys
from urllib.parse import urlparse

import boto3
from botocore.config import Config

# La consola de Windows usa cp1252 por defecto y lanza UnicodeEncodeError con los
# símbolos de estado. Se fuerza UTF-8 aquí, que es el módulo que todos importan.
for _flujo in (sys.stdout, sys.stderr):
    try:
        _flujo.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

FLOCI_ENDPOINT = os.getenv("FLOCI_ENDPOINT", "http://localhost:4566")
REGION = os.getenv("AWS_LOCAL_REGION", "us-east-1")
ACCOUNT_ID = "000000000000"

HOSTS_LOCALES = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}

# Reintentos cortos: si el emulador no está arriba queremos fallar rápido y con un
# mensaje claro, no quedarnos colgados el timeout por defecto de boto3.
BOTO_CONFIG = Config(
    region_name=REGION,
    retries={"max_attempts": 2, "mode": "standard"},
    connect_timeout=5,
    read_timeout=15,
)


class EndpointNoLocalError(RuntimeError):
    """El endpoint configurado no apunta a un emulador local."""


def _validar_endpoint(endpoint: str) -> None:
    host = urlparse(endpoint).hostname
    if host not in HOSTS_LOCALES:
        raise EndpointNoLocalError(
            f"FLOCI_ENDPOINT apunta a '{host}', que no es un host local. "
            "Estos scripts solo deben correr contra el emulador; abortando para no "
            "crear infraestructura AWS real con las credenciales del .env."
        )


def cliente(servicio: str, endpoint: str = None):
    """Devuelve un cliente boto3 apuntando al emulador, con credenciales ficticias."""
    endpoint = endpoint or FLOCI_ENDPOINT
    _validar_endpoint(endpoint)
    return boto3.client(
        servicio,
        endpoint_url=endpoint,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_session_token="test",
        config=BOTO_CONFIG,
    )


def recurso(servicio: str, endpoint: str = None):
    endpoint = endpoint or FLOCI_ENDPOINT
    _validar_endpoint(endpoint)
    return boto3.resource(
        servicio,
        endpoint_url=endpoint,
        region_name=REGION,
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_session_token="test",
        config=BOTO_CONFIG,
    )


# --- Nombres de recursos (fuente única de verdad para todos los scripts) ---

BUCKETS = {
    "bronze": "perabank-bronze-datalake",
    "silver": "perabank-silver-datalake",
    "gold": "perabank-gold-datalake",
    "ml": "perabank-ml-artifacts",
}

COLA_INGESTA = "perabank-transaction-ingest.fifo"
COLA_DLQ = "perabank-transaction-dlq.fifo"
MAX_RECEIVE_COUNT = 3

TABLA_FEATURES = "PeraBank_Customer_Features"
TABLA_ALERTAS = "PeraBank_Fraud_Alerts"
GSI_RIESGO_MACRO = "gsi_macro_risk"

ROL_LAMBDA = "PeraBank_Lambda_Role"
POLITICA_LAMBDA = "PeraBank_Lambda_LeastPrivilege"

LAMBDA_ETL = "lambda_etl_trigger"
LAMBDA_SCORER = "lambda_risk_scorer"
API_NOMBRE = "perabank-http-api"

MODELO_KEY = "models/perabank_risk_pipeline_v1.joblib"


def arn_bucket(nombre: str) -> str:
    return f"arn:aws:s3:::{nombre}"


def arn_tabla(nombre: str) -> str:
    return f"arn:aws:dynamodb:{REGION}:{ACCOUNT_ID}:table/{nombre}"


def arn_cola(nombre: str) -> str:
    return f"arn:aws:sqs:{REGION}:{ACCOUNT_ID}:{nombre}"


def arn_lambda(nombre: str) -> str:
    return f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{nombre}"
