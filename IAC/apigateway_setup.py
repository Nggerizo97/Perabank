"""API HTTP (API Gateway v2) con rutas hacia las Lambdas.

Se usa HTTP API y no REST API: cuesta ~$1.00 por millón de peticiones frente a
~$3.50 del REST API, y para un backend Lambda con integración proxy no se pierde
nada relevante. A 500k peticiones/mes la diferencia es el grueso del presupuesto.
"""
from botocore.exceptions import ClientError

from IAC.floci_config import (
    API_NOMBRE, LAMBDA_ETL, LAMBDA_SCORER, REGION, arn_lambda, cliente,
)

RUTAS = [
    ("POST /v1/risk/score", LAMBDA_SCORER),
    ("GET /v1/market/trm", LAMBDA_SCORER),
]


def _api_existente(api, nombre: str):
    for item in api.get_apis().get("Items", []):
        if item.get("Name") == nombre:
            return item
    return None


def _permitir_invocacion(nombre_funcion: str, api_id: str, log) -> None:
    """Sin esta política de recurso, API Gateway recibe 403 al invocar la Lambda."""
    lam = cliente("lambda")
    try:
        lam.add_permission(
            FunctionName=nombre_funcion,
            StatementId=f"apigw-{api_id}-{nombre_funcion}"[:64],
            Action="lambda:InvokeFunction",
            Principal="apigateway.amazonaws.com",
            SourceArn=f"arn:aws:execute-api:{REGION}:000000000000:{api_id}/*/*",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ResourceConflictException",):
            log(f"    · permiso de invocación no aplicado: {exc.response['Error']['Code']}")


def provisionar(log=print) -> dict:
    api = cliente("apigatewayv2")
    log("API Gateway — HTTP API")

    existente = _api_existente(api, API_NOMBRE)
    if existente:
        api_id = existente["ApiId"]
        estado = "ya existía"
    else:
        creada = api.create_api(
            Name=API_NOMBRE,
            ProtocolType="HTTP",
            Description="Fachada HTTP de PeraBank (scoring y mercado)",
        )
        api_id = creada["ApiId"]
        estado = "creada"

    rutas_existentes = {r["RouteKey"] for r in api.get_routes(ApiId=api_id).get("Items", [])}

    for route_key, funcion in RUTAS:
        if route_key in rutas_existentes:
            log(f"  · {route_key:<24} ya existía")
            continue
        integracion = api.create_integration(
            ApiId=api_id,
            IntegrationType="AWS_PROXY",
            IntegrationUri=arn_lambda(funcion),
            PayloadFormatVersion="2.0",
            IntegrationMethod="POST",
        )
        api.create_route(ApiId=api_id, RouteKey=route_key,
                         Target=f"integrations/{integracion['IntegrationId']}")
        _permitir_invocacion(funcion, api_id, log)
        log(f"  ✔ {route_key:<24} -> {funcion}")

    try:
        api.create_stage(ApiId=api_id, StageName="$default", AutoDeploy=True)
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in ("ConflictException", "BadRequestException"):
            log(f"    · stage no creado: {exc.response['Error']['Code']}")

    log(f"  ✔ API {API_NOMBRE} {estado} · id {api_id}")
    return {"api_id": api_id, "rutas": [r for r, _ in RUTAS]}


if __name__ == "__main__":
    provisionar()
