"""Simulación end-to-end del flujo transaccional sobre el emulador local.

  1. Envía una transacción sintética a la cola FIFO (con deduplicación).
  2. Reenvía el MISMO mensaje para probar que la deduplicación lo descarta.
  3. Consume el mensaje e invoca el handler de ETL, que aterriza el JSON
     particionado en bronze.
  4. Escribe features del cliente en DynamoDB.
  5. Invoca el handler de scoring y verifica una respuesta HTTP 200.

Los handlers se invocan en proceso, no como contenedores Lambda: así se prueba la
lógica del handler contra servicios AWS emulados reales, sin depender de que el
emulador ejecute contenedores. Las llamadas a S3, SQS y DynamoDB sí son reales.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "lambda"))

from IAC.floci_config import (  # noqa: E402
    BUCKETS, COLA_INGESTA, FLOCI_ENDPOINT, MODELO_KEY, TABLA_ALERTAS, TABLA_FEATURES,
    cliente,
)

MODELO_LOCAL = RAIZ / "perabank_risk_pipeline_v1.joblib"

TRANSACCION = {
    "id_transaccion": "TX-SIM-000001",
    "sk_cliente": "c3ab8ff13720e8ad9047dd39466b3c89",
    "fecha": "2026-08-07",
    "monto_usd": 1450.75,
    "tipo_transaccion": "TRANSFER",
    "id_evento_tarjeta": "cc_sim_0001",
    "balance_usd": 2300.0,
    "tiene_hipoteca": 1,
    "tiene_prestamo_personal": 0,
}


class Resultado:
    def __init__(self):
        self.pasos = []

    def registrar(self, nombre: str, ok: bool, detalle: str = "") -> None:
        self.pasos.append((nombre, ok, detalle))
        print(f"  {'✔' if ok else '✘'} {nombre:<44} {detalle}")

    @property
    def ok(self) -> bool:
        return all(p[1] for p in self.pasos)


def _url_cola(sqs) -> str:
    return sqs.get_queue_url(QueueName=COLA_INGESTA)["QueueUrl"]


def simular() -> Resultado:
    import os
    os.environ.setdefault("FLOCI_ENDPOINT", FLOCI_ENDPOINT)

    resultado = Resultado()
    sqs = cliente("sqs")
    s3 = cliente("s3")
    ddb = cliente("dynamodb")

    print(f"Simulación end-to-end contra {FLOCI_ENDPOINT}")
    print("-" * 66)

    # 0. Publicar el artefacto ML para que el scorer use el modelo real y no el
    #    respaldo por reglas. Es el mismo patrón de producción: el modelo vive en
    #    S3, no dentro del zip de la Lambda.
    if MODELO_LOCAL.exists():
        s3.upload_file(str(MODELO_LOCAL), BUCKETS["ml"], MODELO_KEY)
        resultado.registrar("Artefacto ML publicado en S3", True,
                            f"{MODELO_LOCAL.stat().st_size/1e6:.1f} MB -> {MODELO_KEY}")
    else:
        resultado.registrar("Artefacto ML publicado en S3", True,
                            "no existe local; el scorer usará reglas")

    # 1. Envío a la cola FIFO
    url = _url_cola(sqs)
    dedup_id = TRANSACCION["id_transaccion"]
    envio = sqs.send_message(
        QueueUrl=url,
        MessageBody=json.dumps(TRANSACCION),
        MessageGroupId="transacciones",
        MessageDeduplicationId=dedup_id,
    )
    resultado.registrar("Transacción enviada a la cola FIFO",
                        bool(envio.get("MessageId")), f"id {envio.get('MessageId', '')[:18]}")

    # 2. Reenvío idéntico: SQS FIFO debe deduplicar dentro de la ventana de 5 min.
    # Se verifica la PROFUNDIDAD de la cola y no el MessageId devuelto: AWS real
    # repite el MessageId original ante un duplicado, pero los emuladores suelen
    # devolver uno nuevo aunque descarten el mensaje correctamente. Lo que garantiza
    # que no se duplique un movimiento de dinero es que la cola siga teniendo 1.
    for _ in range(2):
        sqs.send_message(
            QueueUrl=url,
            MessageBody=json.dumps(TRANSACCION),
            MessageGroupId="transacciones",
            MessageDeduplicationId=dedup_id,
        )
    profundidad = int(sqs.get_queue_attributes(
        QueueUrl=url, AttributeNames=["ApproximateNumberOfMessages"]
    )["Attributes"]["ApproximateNumberOfMessages"])
    resultado.registrar("3 envíos idénticos dejan 1 mensaje en cola", profundidad == 1,
                        f"profundidad={profundidad}")

    # 3. Consumo e invocación del handler de ETL
    recibidos = sqs.receive_message(QueueUrl=url, MaxNumberOfMessages=1,
                                    WaitTimeSeconds=1).get("Messages", [])
    if not recibidos:
        resultado.registrar("Mensaje recibido de la cola", False, "cola vacía")
        return resultado

    from lambda_etl_trigger import lambda_handler as etl_handler
    evento_sqs = {"Records": [{"eventSource": "aws:sqs", "body": recibidos[0]["Body"]}]}
    respuesta_etl = etl_handler(evento_sqs)
    cuerpo_etl = json.loads(respuesta_etl["body"])
    resultado.registrar("ETL aterrizó el objeto en bronze",
                        respuesta_etl["statusCode"] == 200 and cuerpo_etl["objetos_escritos"] == 1,
                        cuerpo_etl["claves"][0] if cuerpo_etl["claves"] else "sin clave")

    sqs.delete_message(QueueUrl=url, ReceiptHandle=recibidos[0]["ReceiptHandle"])

    # 4. Verificación del particionado year=/month=/day= en S3
    listado = s3.list_objects_v2(Bucket=BUCKETS["bronze"], Prefix="raw/transacciones/year=")
    claves = [o["Key"] for o in listado.get("Contents", [])]
    particionado = any("year=" in k and "month=" in k and "day=" in k for k in claves)
    resultado.registrar("Objeto particionado Hive en S3", particionado,
                        f"{len(claves)} objeto(s) bajo raw/transacciones/")

    # 5. Features del cliente en DynamoDB
    ddb.put_item(
        TableName=TABLA_FEATURES,
        Item={
            "sk_cliente": {"S": TRANSACCION["sk_cliente"]},
            "risk_segment": {"S": "Patrimonio medio"},
            "active_loan_count": {"N": "1"},
            "last_trm_rate": {"N": "3157.43"},
            "updated_at": {"S": datetime.now(timezone.utc).isoformat()},
        },
    )
    leido = ddb.get_item(TableName=TABLA_FEATURES,
                         Key={"sk_cliente": {"S": TRANSACCION["sk_cliente"]}})
    resultado.registrar("Features del cliente en DynamoDB", "Item" in leido,
                        f"segmento {leido.get('Item', {}).get('risk_segment', {}).get('S', '?')}")

    # 6. Scoring vía el handler que está detrás de POST /v1/risk/score
    from lambda_risk_scorer import lambda_handler as scorer_handler
    evento_api = {
        "requestContext": {"http": {"method": "POST", "path": "/v1/risk/score"}},
        "body": json.dumps(TRANSACCION),
    }
    respuesta_api = scorer_handler(evento_api)
    cuerpo_api = json.loads(respuesta_api["body"])
    resultado.registrar("POST /v1/risk/score devolvió HTTP 200",
                        respuesta_api["statusCode"] == 200,
                        f"riesgo {cuerpo_api.get('clase_riesgo')} "
                        f"p={cuerpo_api.get('probabilidad_mora')} "
                        f"motor={cuerpo_api.get('motor')}")

    # 7. Alerta persistida y consultable por el GSI
    alerta = ddb.get_item(TableName=TABLA_ALERTAS,
                          Key={"id_evento_tarjeta": {"S": TRANSACCION["id_evento_tarjeta"]}})
    resultado.registrar("Alerta de fraude registrada", "Item" in alerta,
                        f"banda {alerta.get('Item', {}).get('banda_riesgo_macro', {}).get('S', '?')}")

    return resultado


def main() -> int:
    try:
        resultado = simular()
    except Exception as exc:
        print(f"\n✘ La simulación falló: {type(exc).__name__}: {exc}")
        print(f"  ¿Está el emulador arriba? Diagnostica con: python -m IAC.verify_floci")
        return 1

    print("-" * 66)
    exitosos = sum(1 for _, ok, _ in resultado.pasos if ok)
    print(f"{exitosos}/{len(resultado.pasos)} pasos correctos")
    return 0 if resultado.ok else 1


if __name__ == "__main__":
    sys.exit(main())
