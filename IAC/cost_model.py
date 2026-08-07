"""Estimación del costo mensual en AWS real de esta arquitectura.

Precios on-demand de us-east-1 (USD, referencia 2026). El objetivo es demostrar que
el diseño se sostiene bajo $5/mes a 500k transacciones, y —más importante— mostrar
qué componente rompería el presupuesto primero si el volumen crece.
"""
PRECIOS = {
    "s3_gb_mes": 0.023,
    "s3_glacier_gb_mes": 0.004,
    "s3_put_por_1000": 0.005,
    "s3_get_por_1000": 0.0004,
    "lambda_por_1m_req": 0.20,
    "lambda_gb_segundo": 0.0000166667,
    "dynamodb_wru_por_1m": 1.25,
    "dynamodb_rru_por_1m": 0.25,
    "dynamodb_gb_mes": 0.25,
    "sqs_por_1m_req": 0.40,
    "apigw_http_por_1m": 1.00,
}

GRATIS = {
    "s3_gb": 5, "s3_put": 2000, "s3_get": 20000,
    "lambda_req": 1_000_000, "lambda_gb_seg": 400_000,
    "dynamodb_gb": 25, "sqs_req": 1_000_000, "apigw_req": 1_000_000,
}


def estimar(transacciones_mes: int = 500_000, gb_bronze: float = 3.0,
            gb_silver_gold: float = 2.0, ms_por_lambda: int = 250) -> dict:
    # Cada transacción: 1 SQS + 1 Lambda ETL + 1 PUT en S3 + 1 escritura DynamoDB.
    # Se asume además scoring en el 20% (no toda transacción se puntúa).
    scorings = int(transacciones_mes * 0.20)

    lambda_req = transacciones_mes + scorings
    gb_seg = (transacciones_mes * (256 / 1024) * (ms_por_lambda / 1000)
              + scorings * (1024 / 1024) * (ms_por_lambda / 1000))

    def cobrable(uso, franquicia):
        return max(0, uso - franquicia)

    lineas = {
        "S3 almacenamiento": (
            cobrable(gb_silver_gold, GRATIS["s3_gb"]) * PRECIOS["s3_gb_mes"]
            + gb_bronze * PRECIOS["s3_glacier_gb_mes"]
        ),
        "S3 peticiones PUT": cobrable(transacciones_mes, GRATIS["s3_put"]) / 1000 * PRECIOS["s3_put_por_1000"],
        "Lambda invocaciones": cobrable(lambda_req, GRATIS["lambda_req"]) / 1e6 * PRECIOS["lambda_por_1m_req"],
        "Lambda cómputo": cobrable(gb_seg, GRATIS["lambda_gb_seg"]) * PRECIOS["lambda_gb_segundo"],
        "DynamoDB escrituras": transacciones_mes / 1e6 * PRECIOS["dynamodb_wru_por_1m"],
        "DynamoDB lecturas": scorings / 1e6 * PRECIOS["dynamodb_rru_por_1m"],
        "SQS peticiones": cobrable(transacciones_mes * 3, GRATIS["sqs_req"]) / 1e6 * PRECIOS["sqs_por_1m_req"],
        "API Gateway HTTP": cobrable(scorings, GRATIS["apigw_req"]) / 1e6 * PRECIOS["apigw_http_por_1m"],
    }
    return {"lineas": lineas, "total": sum(lineas.values()),
            "transacciones_mes": transacciones_mes}


def reporte(transacciones_mes: int = 500_000, log=print) -> float:
    estimacion = estimar(transacciones_mes)
    log(f"Costo mensual estimado en AWS real ({transacciones_mes:,} transacciones/mes)")
    log("-" * 62)
    for concepto, valor in estimacion["lineas"].items():
        log(f"  {concepto:<26} ${valor:>8.4f}")
    total = estimacion["total"]
    log("-" * 62)
    log(f"  {'TOTAL':<26} ${total:>8.4f}   "
        f"{'✔ bajo el objetivo de $5' if total < 5 else '✘ supera $5'}")
    return total


if __name__ == "__main__":
    for volumen in (100_000, 500_000, 2_000_000):
        reporte(volumen)
        print()
