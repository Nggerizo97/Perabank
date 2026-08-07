"""Data lake de 4 capas en S3, con cifrado en reposo y ciclo de vida.

Control de costo: el bucket bronze acumula crudos que casi nunca se releen, así que
transiciona a GLACIER a los 30 días (~$0.004/GB vs $0.023/GB en Standard). Silver y
gold se releen en cada corrida analítica y se quedan en Standard.
"""
from botocore.exceptions import ClientError

from IAC.floci_config import BUCKETS, REGION, cliente

# Particionado Hive year=/month=: permite a un consumidor (Athena, Spark, pandas)
# leer un mes sin listar el bucket completo, que es donde se dispara el costo de GET.
PREFIJOS = {
    "bronze": ["raw/transacciones/", "raw/mercado/", "raw/secop/"],
    "silver": ["curated/transacciones/", "curated/mercado/"],
    "gold": ["dim/", "fact/"],
    "ml": ["models/", "metrics/"],
}

CICLO_VIDA = {
    "bronze": [{
        "ID": "bronze-a-glacier-30d",
        "Filter": {"Prefix": "raw/"},
        "Status": "Enabled",
        "Transitions": [{"Days": 30, "StorageClass": "GLACIER"}],
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
    }],
    "silver": [{
        "ID": "silver-limpieza-versiones",
        "Filter": {"Prefix": "curated/"},
        "Status": "Enabled",
        "NoncurrentVersionExpiration": {"NoncurrentDays": 30},
        "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 7},
    }],
    "gold": [{
        "ID": "gold-limpieza-versiones",
        "Filter": {"Prefix": ""},
        "Status": "Enabled",
        "NoncurrentVersionExpiration": {"NoncurrentDays": 90},
    }],
    "ml": [{
        "ID": "ml-artefactos-antiguos",
        "Filter": {"Prefix": "models/"},
        "Status": "Enabled",
        "NoncurrentVersionExpiration": {"NoncurrentDays": 180},
    }],
}


def _crear_bucket(s3, nombre: str) -> bool:
    try:
        # us-east-1 es la única región que NO admite CreateBucketConfiguration.
        if REGION == "us-east-1":
            s3.create_bucket(Bucket=nombre)
        else:
            s3.create_bucket(Bucket=nombre,
                             CreateBucketConfiguration={"LocationConstraint": REGION})
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return False
        raise


def _aplicar(s3, nombre: str, capa: str, log) -> None:
    s3.put_bucket_encryption(
        Bucket=nombre,
        ServerSideEncryptionConfiguration={
            "Rules": [{
                "ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"},
                "BucketKeyEnabled": True,
            }]
        },
    )
    s3.put_public_access_block(
        Bucket=nombre,
        PublicAccessBlockConfiguration={
            "BlockPublicAcls": True, "IgnorePublicAcls": True,
            "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
        },
    )
    for opcional, accion in (
        ("versionado", lambda: s3.put_bucket_versioning(
            Bucket=nombre, VersioningConfiguration={"Status": "Enabled"})),
        ("ciclo de vida", lambda: s3.put_bucket_lifecycle_configuration(
            Bucket=nombre, LifecycleConfiguration={"Rules": CICLO_VIDA[capa]})),
    ):
        try:
            accion()
        except ClientError as exc:
            # Un emulador puede no implementar todas las sub-APIs de S3. No es
            # motivo para abortar el despliegue completo.
            log(f"    · {opcional} no aplicado: {exc.response['Error']['Code']}")


def provisionar(log=print) -> dict:
    s3 = cliente("s3")
    resumen = {}
    log("S3 — data lake de 4 capas")
    for capa, nombre in BUCKETS.items():
        creado = _crear_bucket(s3, nombre)
        _aplicar(s3, nombre, capa, log)
        for prefijo in PREFIJOS[capa]:
            s3.put_object(Bucket=nombre, Key=prefijo, Body=b"")
        resumen[nombre] = "creado" if creado else "ya existía"
        log(f"  ✔ {nombre:<28} {resumen[nombre]} · AES256 · {len(PREFIJOS[capa])} prefijos")
    return resumen


if __name__ == "__main__":
    provisionar()
