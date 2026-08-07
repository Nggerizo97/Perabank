"""Disparador de ETL: aterriza transacciones crudas en bronze de forma particionada.

Se invoca de dos formas:
  1. Evento SQS (batch de transacciones desde la cola FIFO de ingesta).
  2. Evento S3 ObjectCreated (llegada de un archivo crudo).

Idempotencia: la clave de S3 se deriva del id de transacción, no de un timestamp de
ejecución. Un reintento de SQS reescribe el MISMO objeto en vez de crear un duplicado,
que es lo que ocurriría con una clave basada en uuid4() o en la hora de proceso.
"""
import json
import os
import re
from datetime import datetime, timezone

import boto3

BUCKET_BRONZE = os.getenv("BUCKET_BRONZE", "perabank-bronze-datalake")
ENDPOINT = os.getenv("FLOCI_ENDPOINT")

_SEGURO = re.compile(r"[^A-Za-z0-9_.-]")


def _s3():
    kwargs = {"region_name": os.getenv("AWS_REGION", "us-east-1")}
    if ENDPOINT:
        kwargs.update(endpoint_url=ENDPOINT, aws_access_key_id="test",
                      aws_secret_access_key="test")
    return boto3.client("s3", **kwargs)


def _clave_particionada(transaccion: dict) -> str:
    """Ruta Hive year=/month=/day=: permite leer un mes sin listar todo el bucket.

    Sin particionar, una consulta de un día tendría que hacer LIST sobre millones de
    objetos, y el costo de S3 en un data lake se dispara por LIST/GET, no por GB.
    """
    fecha_txt = transaccion.get("fecha") or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        fecha = datetime.strptime(fecha_txt[:10], "%Y-%m-%d")
    except ValueError:
        fecha = datetime.now(timezone.utc)

    id_tx = _SEGURO.sub("_", str(transaccion.get("id_transaccion", "sin_id")))
    return (f"raw/transacciones/year={fecha:%Y}/month={fecha:%m}/day={fecha:%d}/"
            f"{id_tx}.json")


def _extraer_transacciones(event: dict) -> list:
    """Normaliza eventos de SQS y de S3 a una lista de transacciones."""
    transacciones = []
    for registro in event.get("Records", []):
        origen = registro.get("eventSource") or registro.get("EventSource", "")

        if "sqs" in origen:
            try:
                cuerpo = json.loads(registro["body"])
            except (json.JSONDecodeError, KeyError):
                continue
            transacciones.extend(cuerpo if isinstance(cuerpo, list) else [cuerpo])

        elif "s3" in origen:
            # Un ObjectCreated no trae el contenido: solo se registra el aterrizaje
            # para que la capa silver sepa qué compactar después.
            transacciones.append({
                "id_transaccion": registro["s3"]["object"]["key"],
                "origen": "s3_object_created",
                "bucket": registro["s3"]["bucket"]["name"],
            })
    return transacciones


def lambda_handler(event, context=None):
    s3 = _s3()
    transacciones = _extraer_transacciones(event)
    escritas, fallidas = [], []

    for transaccion in transacciones:
        try:
            clave = _clave_particionada(transaccion)
            s3.put_object(
                Bucket=BUCKET_BRONZE,
                Key=clave,
                Body=json.dumps(transaccion, ensure_ascii=False).encode("utf-8"),
                ContentType="application/json",
                Metadata={"ingested_at": datetime.now(timezone.utc).isoformat()},
            )
            escritas.append(clave)
        except Exception as exc:
            # No se relanza: un mensaje corrupto no debe tumbar el batch completo.
            # SQS reintentará el batch y tras 3 intentos lo aislará en la DLQ.
            fallidas.append({"transaccion": transaccion.get("id_transaccion"),
                             "error": str(exc)[:200]})

    return {
        "statusCode": 200 if not fallidas else 207,
        "body": json.dumps({
            "objetos_escritos": len(escritas),
            "fallidos": len(fallidas),
            "claves": escritas[:10],
            "errores": fallidas[:5],
            "bucket": BUCKET_BRONZE,
        }),
    }
