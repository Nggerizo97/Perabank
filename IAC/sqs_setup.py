"""Cola FIFO de ingesta transaccional con cola de mensajes muertos (DLQ).

Por qué FIFO y no estándar: en un flujo de transacciones bancarias, una cola
estándar entrega "al menos una vez", lo que ante un reintento de red duplicaría un
movimiento de dinero. FIFO garantiza exactly-once dentro de la ventana de
deduplicación de 5 minutos.

La DLQ aísla el mensaje que falla 3 veces en vez de dejarlo reintentando para
siempre y bloqueando su grupo de mensajes (en FIFO, un mensaje atascado detiene
todo su MessageGroupId).
"""
import json

from IAC.floci_config import COLA_DLQ, COLA_INGESTA, MAX_RECEIVE_COUNT, arn_cola, cliente


def _crear(sqs, nombre: str, atributos: dict) -> str:
    return sqs.create_queue(QueueName=nombre, Attributes=atributos)["QueueUrl"]


def provisionar(log=print) -> dict:
    sqs = cliente("sqs")
    log("SQS — ingesta FIFO con DLQ")

    url_dlq = _crear(sqs, COLA_DLQ, {
        "FifoQueue": "true",
        "ContentBasedDeduplication": "true",
        # 14 días: máximo de SQS, para dar tiempo real a investigar un fallo.
        "MessageRetentionPeriod": "1209600",
    })
    log(f"  ✔ {COLA_DLQ:<34} retención 14 días")

    try:
        arn_dlq = sqs.get_queue_attributes(
            QueueUrl=url_dlq, AttributeNames=["QueueArn"])["Attributes"]["QueueArn"]
    except Exception:
        arn_dlq = arn_cola(COLA_DLQ)

    url_principal = _crear(sqs, COLA_INGESTA, {
        "FifoQueue": "true",
        "ContentBasedDeduplication": "true",
        # 30s: debe superar la duración máxima del Lambda consumidor, o el mensaje
        # reaparecería mientras todavía se está procesando.
        "VisibilityTimeout": "30",
        "MessageRetentionPeriod": "345600",
        "RedrivePolicy": json.dumps({
            "deadLetterTargetArn": arn_dlq,
            "maxReceiveCount": MAX_RECEIVE_COUNT,
        }),
    })
    log(f"  ✔ {COLA_INGESTA:<34} dedup por contenido · redrive a DLQ tras "
        f"{MAX_RECEIVE_COUNT} intentos")

    return {"ingesta": url_principal, "dlq": url_dlq, "arn_dlq": arn_dlq}


if __name__ == "__main__":
    provisionar()
