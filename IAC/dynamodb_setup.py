"""Almacén NoSQL de baja latencia: features de cliente y alertas de fraude.

Modo PAY_PER_REQUEST y no PROVISIONED: con tráfico intermitente, el modo aprovisionado
cobra la capacidad reservada 24/7 aunque no llegue ni una petición. On-demand cuesta
$0 en reposo y absorbe picos sin throttling ni auto-scaling que calibrar.

Solo se declaran los atributos que son llave. DynamoDB es schema-on-read: el resto
(risk_segment, active_loan_count, last_trm_rate, updated_at) se escribe por ítem.
"""
from botocore.exceptions import ClientError

from IAC.floci_config import GSI_RIESGO_MACRO, TABLA_ALERTAS, TABLA_FEATURES, cliente

TABLAS = [
    {
        "TableName": TABLA_FEATURES,
        "BillingMode": "PAY_PER_REQUEST",
        "KeySchema": [{"AttributeName": "sk_cliente", "KeyType": "HASH"}],
        "AttributeDefinitions": [{"AttributeName": "sk_cliente", "AttributeType": "S"}],
    },
    {
        "TableName": TABLA_ALERTAS,
        "BillingMode": "PAY_PER_REQUEST",
        "KeySchema": [{"AttributeName": "id_evento_tarjeta", "KeyType": "HASH"}],
        "AttributeDefinitions": [
            {"AttributeName": "id_evento_tarjeta", "AttributeType": "S"},
            {"AttributeName": "banda_riesgo_macro", "AttributeType": "S"},
            {"AttributeName": "monto_usd", "AttributeType": "N"},
        ],
        "GlobalSecondaryIndexes": [{
            "IndexName": GSI_RIESGO_MACRO,
            "KeySchema": [
                {"AttributeName": "banda_riesgo_macro", "KeyType": "HASH"},
                {"AttributeName": "monto_usd", "KeyType": "RANGE"},
            ],
            # KEYS_ONLY mantiene el índice pequeño: el analista consulta el GSI para
            # obtener los ids de alta exposición y luego lee esos ítems por PK.
            # INCLUDE/ALL duplicaría el almacenamiento y el costo de escritura.
            "Projection": {"ProjectionType": "KEYS_ONLY"},
        }],
    },
]


def provisionar(log=print) -> dict:
    ddb = cliente("dynamodb")
    existentes = set(ddb.list_tables().get("TableNames", []))
    resumen = {}

    log("DynamoDB — tablas on-demand")
    for definicion in TABLAS:
        nombre = definicion["TableName"]
        if nombre in existentes:
            resumen[nombre] = "ya existía"
        else:
            try:
                ddb.create_table(**definicion)
                ddb.get_waiter("table_exists").wait(
                    TableName=nombre, WaiterConfig={"Delay": 1, "MaxAttempts": 25})
                resumen[nombre] = "creada"
            except ClientError as exc:
                if exc.response["Error"]["Code"] != "ResourceInUseException":
                    raise
                resumen[nombre] = "ya existía"

        gsis = definicion.get("GlobalSecondaryIndexes", [])
        detalle = f"· GSI {gsis[0]['IndexName']}" if gsis else ""
        log(f"  ✔ {nombre:<28} {resumen[nombre]} · PAY_PER_REQUEST {detalle}")

    return resumen


if __name__ == "__main__":
    provisionar()
