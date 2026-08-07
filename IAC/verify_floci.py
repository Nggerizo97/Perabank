"""Diagnóstico de disponibilidad del emulador AWS local (Floci).

Prueba servicio por servicio con una llamada real de solo lectura, en vez de
asumir que "el puerto está abierto" significa que el servicio responde.
"""
import socket
import sys
from urllib.parse import urlparse

from botocore.exceptions import ClientError, EndpointConnectionError

from IAC.floci_config import FLOCI_ENDPOINT, REGION, cliente

# (servicio, operación de solo lectura, kwargs)
SONDAS = [
    ("s3", "list_buckets", {}),
    ("sqs", "list_queues", {}),
    ("dynamodb", "list_tables", {}),
    ("iam", "list_roles", {}),
    ("lambda", "list_functions", {}),
    ("apigatewayv2", "get_apis", {}),
]


def puerto_abierto(endpoint: str) -> bool:
    partes = urlparse(endpoint)
    puerto = partes.port or (443 if partes.scheme == "https" else 80)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(3)
        return sock.connect_ex((partes.hostname, puerto)) == 0


def sondear(servicio: str, operacion: str, kwargs: dict) -> tuple:
    try:
        getattr(cliente(servicio), operacion)(**kwargs)
        return True, "operativo"
    except EndpointConnectionError:
        return False, "sin conexión al endpoint"
    except ClientError as exc:
        # Una respuesta de error del servicio significa que el servicio SÍ está
        # atendiendo; solo rechazó la llamada concreta.
        codigo = exc.response.get("Error", {}).get("Code", "?")
        return True, f"responde (ClientError {codigo})"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {str(exc)[:70]}"


def main() -> int:
    print(f"Emulador AWS local -> {FLOCI_ENDPOINT}  (región {REGION})")
    print("-" * 66)

    if not puerto_abierto(FLOCI_ENDPOINT):
        print(f"✘ Nada escuchando en {FLOCI_ENDPOINT}\n")
        print("  Arranca el emulador antes de provisionar:")
        print("    floci start            # CLI nativo (https://floci.io)")
        print("    docker compose up      # imagen floci/floci:latest")
        print("\n  Floci necesita Docker para Lambda, ECS y RDS; S3, SQS, DynamoDB,")
        print("  IAM y API Gateway funcionan sin contenedores.")
        return 1

    print(f"✔ Puerto abierto en {FLOCI_ENDPOINT}\n")
    fallos = 0
    for servicio, operacion, kwargs in SONDAS:
        ok, detalle = sondear(servicio, operacion, kwargs)
        print(f"  {'✔' if ok else '✘'} {servicio:<14} {detalle}")
        fallos += 0 if ok else 1

    print("-" * 66)
    if fallos:
        print(f"{fallos} servicio(s) no disponibles. Revisa qué expone tu emulador.")
        return 1
    print("Todos los servicios requeridos responden. Listo para desplegar:")
    print("  python -m IAC.deploy_local_cloud")
    return 0


if __name__ == "__main__":
    sys.exit(main())
