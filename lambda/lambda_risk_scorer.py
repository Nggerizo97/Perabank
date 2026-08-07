"""Scoring de riesgo en tiempo real tras API Gateway (POST /v1/risk/score).

Presupuesto de tamaño del paquete: el zip del handler debe quedar por debajo de
50 MB. Aquí se logra con dos decisiones:

  1. scikit-learn, numpy y pandas viajan en una Lambda Layer, no en el zip.
  2. El artefacto perabank_risk_pipeline_v1.joblib pesa ~32 MB y NO se empaqueta:
     se descarga de S3 al arranque en frío y se cachea en /tmp (512 MB de espacio
     efímero) y en memoria del contenedor. Las invocaciones tibias no vuelven a
     descargarlo, así que el costo de arranque se paga una vez por contenedor.

Si el modelo no está disponible, el handler responde con un scoring degradado
basado en reglas y lo declara en la respuesta, en vez de devolver 500: en un flujo
de decisión crediticia es preferible una respuesta explícitamente degradada a una
caída del endpoint.
"""
import json
import os
from datetime import datetime, timezone
from decimal import Decimal

import boto3

BUCKET_ML = os.getenv("BUCKET_ML", "perabank-ml-artifacts")
MODELO_KEY = os.getenv("MODELO_KEY", "models/perabank_risk_pipeline_v1.joblib")
TABLA_FEATURES = os.getenv("TABLA_FEATURES", "PeraBank_Customer_Features")
TABLA_ALERTAS = os.getenv("TABLA_ALERTAS", "PeraBank_Fraud_Alerts")
RUTA_CACHE = "/tmp/perabank_risk_pipeline_v1.joblib"
ENDPOINT = os.getenv("FLOCI_ENDPOINT")

UMBRAL_REVISAR = 0.20
UMBRAL_RECHAZAR = 0.50

_modelo = None  # cache entre invocaciones tibias del mismo contenedor


def _cliente(servicio: str):
    kwargs = {"region_name": os.getenv("AWS_REGION", "us-east-1")}
    if ENDPOINT:
        kwargs.update(endpoint_url=ENDPOINT, aws_access_key_id="test",
                      aws_secret_access_key="test")
    return boto3.client(servicio, **kwargs)


def cargar_modelo():
    """Descarga el pipeline desde S3 una sola vez por contenedor."""
    global _modelo
    if _modelo is not None:
        return _modelo
    try:
        import joblib
        if not os.path.exists(RUTA_CACHE):
            _cliente("s3").download_file(BUCKET_ML, MODELO_KEY, RUTA_CACHE)
        artefacto = joblib.load(RUTA_CACHE)
        _modelo = artefacto.get("credit", artefacto) if isinstance(artefacto, dict) else artefacto
        return _modelo
    except Exception:
        return None


def _leer_features(sk_cliente: str) -> dict:
    try:
        respuesta = _cliente("dynamodb").get_item(
            TableName=TABLA_FEATURES,
            Key={"sk_cliente": {"S": sk_cliente}},
        )
    except Exception:
        return {}
    item = respuesta.get("Item", {})
    return {k: list(v.values())[0] for k, v in item.items()}


def _score_reglas(payload: dict) -> tuple:
    """Respaldo determinístico si el modelo no está disponible."""
    riesgo = 0.05
    riesgo += 0.15 if str(payload.get("tiene_mora", "")).lower() in ("1", "true", "si") else 0.0
    riesgo += 0.10 if float(payload.get("tiene_prestamo_personal", 0) or 0) else 0.0
    riesgo += 0.05 if float(payload.get("tiene_hipoteca", 0) or 0) else 0.0
    if float(payload.get("balance_usd", 0) or 0) < 0:
        riesgo += 0.20
    return min(riesgo, 0.99), "reglas_degradado"


def _score_modelo(modelo, payload: dict) -> tuple:
    import pandas as pd
    pipeline = modelo["pipeline"] if isinstance(modelo, dict) else modelo
    esquema = modelo.get("features", {}) if isinstance(modelo, dict) else {}
    columnas = list(esquema.get("numericas", [])) + list(esquema.get("categoricas", []))
    defaults = modelo.get("defaults", {}) if isinstance(modelo, dict) else {}

    fila = {}
    for columna in columnas:
        valor = payload.get(columna, defaults.get(columna))
        fila[columna] = valor if valor is not None else 0
    return float(pipeline.predict_proba(pd.DataFrame([fila]))[0][1]), "modelo_ml"


def _clasificar(probabilidad: float) -> str:
    if probabilidad >= UMBRAL_RECHAZAR:
        return "ALTO"
    return "MEDIO" if probabilidad >= UMBRAL_REVISAR else "BAJO"


def _registrar_alerta(payload: dict, probabilidad: float, banda: str) -> bool:
    id_evento = payload.get("id_evento_tarjeta")
    if not id_evento:
        return False
    try:
        _cliente("dynamodb").put_item(
            TableName=TABLA_ALERTAS,
            Item={
                "id_evento_tarjeta": {"S": str(id_evento)},
                "banda_riesgo_macro": {"S": banda},
                "monto_usd": {"N": str(Decimal(str(payload.get("monto_usd", 0))))},
                "probabilidad_mora": {"N": f"{probabilidad:.6f}"},
                "evaluado_en": {"S": datetime.now(timezone.utc).isoformat()},
            },
        )
        return True
    except Exception:
        return False


def _respuesta(codigo: int, cuerpo: dict) -> dict:
    return {
        "statusCode": codigo,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(cuerpo, ensure_ascii=False),
    }


def lambda_handler(event, context=None):
    try:
        cuerpo = event.get("body", event)
        payload = json.loads(cuerpo) if isinstance(cuerpo, str) else dict(cuerpo)
    except (json.JSONDecodeError, TypeError, ValueError):
        return _respuesta(400, {"error": "Cuerpo JSON inválido"})

    sk_cliente = payload.get("sk_cliente")
    if not sk_cliente:
        return _respuesta(400, {"error": "Falta 'sk_cliente'"})

    # Las features persistidas son la base; el payload de la petición las sobrescribe.
    combinado = {**_leer_features(sk_cliente), **payload}

    modelo = cargar_modelo()
    if modelo is not None:
        try:
            probabilidad, motor = _score_modelo(modelo, combinado)
        except Exception:
            probabilidad, motor = _score_reglas(combinado)
    else:
        probabilidad, motor = _score_reglas(combinado)

    banda = _clasificar(probabilidad)
    return _respuesta(200, {
        "sk_cliente": sk_cliente,
        "probabilidad_mora": round(probabilidad, 6),
        "clase_riesgo": banda,
        "motor": motor,
        "alerta_registrada": _registrar_alerta(combinado, probabilidad, banda),
        "umbrales": {"revisar": UMBRAL_REVISAR, "rechazar": UMBRAL_RECHAZAR},
        "evaluado_en": datetime.now(timezone.utc).isoformat(),
    })
