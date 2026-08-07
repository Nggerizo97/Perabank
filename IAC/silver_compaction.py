"""Compactación bronze -> silver: resuelve el problema de los archivos pequeños.

Hallazgo que motiva este módulo: en el modelo de costos, escribir un objeto S3 por
transacción hace que los PUT sean el 74% del gasto a 500k tx/mes, y a 2M tx/mes
disparan el total a ~$15 (sobre el objetivo de $5). El almacenamiento en GB es
irrelevante en comparación.

La compactación agrupa los JSON crudos de una partición en un único Parquet. Reduce
los PUT a razón de 1 por partición en vez de 1 por transacción, y de paso elimina el
costo de miles de GET al leer la capa silver.
"""
from collections import defaultdict
from datetime import datetime, timezone
from io import BytesIO
import json

from IAC.floci_config import BUCKETS, cliente

PREFIJO_BRONZE = "raw/transacciones/"
PREFIJO_SILVER = "curated/transacciones/"


def _particion(clave: str) -> str:
    """Extrae 'year=YYYY/month=MM/day=DD' de la clave del objeto."""
    partes = [p for p in clave.split("/") if p.startswith(("year=", "month=", "day="))]
    return "/".join(partes)


def compactar(log=print) -> dict:
    s3 = cliente("s3")
    paginador = s3.get_paginator("list_objects_v2")

    por_particion = defaultdict(list)
    for pagina in paginador.paginate(Bucket=BUCKETS["bronze"], Prefix=PREFIJO_BRONZE):
        for objeto in pagina.get("Contents", []):
            if objeto["Key"].endswith(".json"):
                por_particion[_particion(objeto["Key"])].append(objeto["Key"])

    if not por_particion:
        log("  · No hay objetos crudos que compactar.")
        return {"particiones": 0, "objetos_leidos": 0, "puts_ahorrados": 0}

    try:
        import pandas as pd
    except ImportError:
        log("  ⚠ pandas no disponible; se omite la compactación.")
        return {"particiones": 0, "objetos_leidos": 0, "puts_ahorrados": 0}

    total_leidos = 0
    for particion, claves in sorted(por_particion.items()):
        registros = []
        for clave in claves:
            cuerpo = s3.get_object(Bucket=BUCKETS["bronze"], Key=clave)["Body"].read()
            try:
                registros.append(json.loads(cuerpo))
            except json.JSONDecodeError:
                continue

        if not registros:
            continue

        buffer = BytesIO()
        pd.DataFrame(registros).to_parquet(buffer, index=False)
        destino = f"{PREFIJO_SILVER}{particion}/part-0000.parquet"
        s3.put_object(
            Bucket=BUCKETS["silver"],
            Key=destino,
            Body=buffer.getvalue(),
            ContentType="application/octet-stream",
            Metadata={"compactado_en": datetime.now(timezone.utc).isoformat(),
                      "objetos_origen": str(len(claves))},
        )
        total_leidos += len(registros)
        log(f"  ✔ {particion}: {len(registros)} objetos -> 1 Parquet")

    ahorro = max(0, total_leidos - len(por_particion))
    log(f"  · {total_leidos} objetos compactados en {len(por_particion)} Parquet "
        f"({ahorro} PUT evitados en la capa silver)")
    return {"particiones": len(por_particion), "objetos_leidos": total_leidos,
            "puts_ahorrados": ahorro}


if __name__ == "__main__":
    print("Compactación bronze -> silver")
    compactar()
