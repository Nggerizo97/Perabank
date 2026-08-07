"""Rol de ejecución de las Lambdas con política de privilegio mínimo.

Cada ARN se enumera explícitamente en vez de usar un prefijo comodín como
`arn:aws:s3:::perabank-*`. Un comodín de prefijo concede acceso a cualquier bucket
futuro que alguien cree con ese nombre, incluido uno de otra cuenta si el nombre
queda libre. Enumerar cuesta unas líneas más y cierra esa puerta.

La única excepción es `.../*` dentro de cada bucket, que es obligatorio: las
acciones sobre objetos exigen un ARN de objeto, no de bucket.
"""
import json

from botocore.exceptions import ClientError

from IAC.floci_config import (
    BUCKETS, COLA_DLQ, COLA_INGESTA, GSI_RIESGO_MACRO, POLITICA_LAMBDA, ROL_LAMBDA,
    TABLA_ALERTAS, TABLA_FEATURES, arn_bucket, arn_cola, arn_tabla, cliente,
)

CONFIANZA = {
    "Version": "2012-10-17",
    "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "lambda.amazonaws.com"},
        "Action": "sts:AssumeRole",
    }],
}


def construir_politica() -> dict:
    arns_objeto = [f"{arn_bucket(b)}/*" for b in BUCKETS.values()]
    arns_bucket = [arn_bucket(b) for b in BUCKETS.values()]
    arn_alertas = arn_tabla(TABLA_ALERTAS)

    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "LecturaEscrituraObjetosDataLake",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject"],
                "Resource": arns_objeto,
            },
            {
                "Sid": "ListadoSoloDeSusBuckets",
                "Effect": "Allow",
                "Action": ["s3:ListBucket"],
                "Resource": arns_bucket,
            },
            {
                "Sid": "FeaturesYAlertas",
                "Effect": "Allow",
                "Action": ["dynamodb:GetItem", "dynamodb:PutItem", "dynamodb:UpdateItem"],
                "Resource": [arn_tabla(TABLA_FEATURES), arn_alertas],
            },
            {
                "Sid": "ConsultaGSIRiesgoMacro",
                "Effect": "Allow",
                "Action": ["dynamodb:Query"],
                "Resource": [f"{arn_alertas}/index/{GSI_RIESGO_MACRO}"],
            },
            {
                "Sid": "ConsumoColaIngesta",
                "Effect": "Allow",
                "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
                "Resource": [arn_cola(COLA_INGESTA)],
            },
            {
                "Sid": "EnvioAColaMuertos",
                "Effect": "Allow",
                "Action": ["sqs:SendMessage"],
                "Resource": [arn_cola(COLA_DLQ)],
            },
        ],
    }


def auditar_comodines(politica: dict) -> list:
    """Devuelve los Sid cuyo Resource es un comodín total. Debe salir vacío."""
    infractores = []
    for stmt in politica["Statement"]:
        recursos = stmt["Resource"]
        recursos = recursos if isinstance(recursos, list) else [recursos]
        if any(r == "*" for r in recursos):
            infractores.append(stmt.get("Sid", "sin-sid"))
    return infractores


def provisionar(log=print) -> dict:
    iam = cliente("iam")
    politica = construir_politica()

    infractores = auditar_comodines(politica)
    if infractores:
        raise ValueError(f"Política con Resource='*' en: {infractores}")

    log("IAM — rol de ejecución con privilegio mínimo")
    try:
        iam.create_role(
            RoleName=ROL_LAMBDA,
            AssumeRolePolicyDocument=json.dumps(CONFIANZA),
            Description="Ejecución de Lambdas de PeraBank (privilegio mínimo)",
        )
        estado = "creado"
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "EntityAlreadyExists":
            raise
        estado = "ya existía"

    iam.put_role_policy(
        RoleName=ROL_LAMBDA,
        PolicyName=POLITICA_LAMBDA,
        PolicyDocument=json.dumps(politica),
    )

    try:
        arn = iam.get_role(RoleName=ROL_LAMBDA)["Role"]["Arn"]
    except ClientError:
        arn = f"arn:aws:iam::000000000000:role/{ROL_LAMBDA}"

    n_recursos = sum(len(s["Resource"]) for s in politica["Statement"])
    log(f"  ✔ {ROL_LAMBDA:<28} {estado}")
    log(f"  ✔ {POLITICA_LAMBDA:<28} {len(politica['Statement'])} sentencias · "
        f"{n_recursos} ARNs explícitos · 0 comodines")
    return {"arn": arn, "politica": politica}


if __name__ == "__main__":
    provisionar()
