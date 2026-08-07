"""Empaquetado y despliegue de las funciones Lambda.

El zip contiene SOLO el handler. Las dependencias pesadas (scikit-learn, numpy,
pandas) van en una Lambda Layer y el modelo de ~32 MB se descarga de S3 en caliente.
Así el paquete queda en kilobytes, muy por debajo del límite de 50 MB de subida
directa, y el arranque en frío no paga descomprimir cientos de MB.
"""
import io
import zipfile
from pathlib import Path

from botocore.exceptions import ClientError

from IAC.floci_config import (
    BUCKETS, LAMBDA_ETL, LAMBDA_SCORER, MODELO_KEY, TABLA_ALERTAS, TABLA_FEATURES,
    arn_lambda, cliente,
)

RAIZ = Path(__file__).resolve().parents[1]
DIR_LAMBDA = RAIZ / "lambda"
LIMITE_ZIP_MB = 50

FUNCIONES = {
    LAMBDA_ETL: {
        "archivo": "lambda_etl_trigger.py",
        "descripcion": "Aterriza transacciones crudas particionadas en bronze",
        "timeout": 30,
        "memoria": 256,
    },
    LAMBDA_SCORER: {
        "archivo": "lambda_risk_scorer.py",
        "descripcion": "Scoring de riesgo crediticio en tiempo real",
        # 1024 MB no es por consumo de RAM sino por CPU: Lambda asigna CPU
        # proporcional a la memoria, y cargar un pipeline de sklearn es CPU-bound.
        # Más memoria reduce el arranque en frío y puede costar menos por invocación.
        "timeout": 60,
        "memoria": 1024,
    },
}


def _empaquetar(nombre_archivo: str) -> bytes:
    origen = DIR_LAMBDA / nombre_archivo
    if not origen.exists():
        raise FileNotFoundError(f"No existe el handler {origen}")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(nombre_archivo, origen.read_text(encoding="utf-8"))
    return buffer.getvalue()


def provisionar(arn_rol: str, log=print) -> dict:
    lam = cliente("lambda")
    resumen = {}
    log("Lambda — despliegue de handlers")

    entorno = {
        "BUCKET_BRONZE": BUCKETS["bronze"],
        "BUCKET_ML": BUCKETS["ml"],
        "MODELO_KEY": MODELO_KEY,
        "TABLA_FEATURES": TABLA_FEATURES,
        "TABLA_ALERTAS": TABLA_ALERTAS,
    }

    for nombre, cfg in FUNCIONES.items():
        paquete = _empaquetar(cfg["archivo"])
        tamano_mb = len(paquete) / 1e6
        if tamano_mb > LIMITE_ZIP_MB:
            raise ValueError(f"{nombre}: el zip pesa {tamano_mb:.1f} MB (límite {LIMITE_ZIP_MB} MB)")

        try:
            lam.create_function(
                FunctionName=nombre,
                Runtime="python3.11",
                Role=arn_rol,
                Handler=f"{cfg['archivo'][:-3]}.lambda_handler",
                Code={"ZipFile": paquete},
                Description=cfg["descripcion"],
                Timeout=cfg["timeout"],
                MemorySize=cfg["memoria"],
                Environment={"Variables": entorno},
                Publish=True,
            )
            estado = "creada"
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ResourceConflictException":
                raise
            lam.update_function_code(FunctionName=nombre, ZipFile=paquete, Publish=True)
            estado = "actualizada"

        resumen[nombre] = {"estado": estado, "zip_kb": round(len(paquete) / 1024, 1),
                           "arn": arn_lambda(nombre)}
        log(f"  ✔ {nombre:<22} {estado} · zip {len(paquete)/1024:.1f} KB "
            f"· {cfg['memoria']} MB · {cfg['timeout']}s")

    log(f"  · Paquetes muy por debajo del límite de {LIMITE_ZIP_MB} MB: "
        "las dependencias ML van en Layer y el modelo se lee de S3.")
    return resumen


if __name__ == "__main__":
    from IAC.iam_setup import provisionar as provisionar_iam
    provisionar(provisionar_iam()["arn"])
