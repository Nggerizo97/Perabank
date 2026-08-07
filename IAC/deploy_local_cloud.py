"""Despliegue declarativo de toda la arquitectura serverless de PeraBank.

Orden obligatorio: IAM antes que Lambda (la función necesita el ARN del rol) y
Lambda antes que API Gateway (la integración necesita el ARN de la función).

Es idempotente: cada módulo detecta el recurso existente y actualiza en vez de
fallar, así que se puede correr las veces que haga falta.
"""
import sys

from botocore.exceptions import EndpointConnectionError

from IAC import (
    apigateway_setup, cost_model, dynamodb_setup, iam_setup, lambda_setup,
    s3_setup, sqs_setup,
)
from IAC.floci_config import FLOCI_ENDPOINT


def desplegar(log=print) -> dict:
    log("=" * 62)
    log(f"PeraBank — despliegue en emulador local ({FLOCI_ENDPOINT})")
    log("=" * 62)

    estado = {}
    estado["s3"] = s3_setup.provisionar(log)
    log("")
    estado["sqs"] = sqs_setup.provisionar(log)
    log("")
    estado["dynamodb"] = dynamodb_setup.provisionar(log)
    log("")

    iam = iam_setup.provisionar(log)
    estado["iam"] = iam
    log("")

    try:
        estado["lambda"] = lambda_setup.provisionar(iam["arn"], log)
        log("")
        estado["apigateway"] = apigateway_setup.provisionar(log)
    except Exception as exc:
        # Lambda y API Gateway son los servicios que un emulador ligero puede no
        # soportar (Floci los ejecuta en contenedores Docker reales). El resto de
        # la infraestructura ya quedó provisionada y es utilizable.
        log(f"  ⚠ Lambda/API Gateway no provisionados: {type(exc).__name__}: {str(exc)[:120]}")
        log("    S3, SQS, DynamoDB e IAM sí quedaron listos.")
        estado["lambda"] = {"error": str(exc)[:200]}

    log("")
    log("=" * 62)
    cost_model.reporte(log=log)
    return estado


def main() -> int:
    try:
        desplegar()
    except EndpointConnectionError:
        print(f"✘ No hay emulador escuchando en {FLOCI_ENDPOINT}.")
        print("  Diagnostica con:  python -m IAC.verify_floci")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
