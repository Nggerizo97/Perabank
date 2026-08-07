# PeraBank — Arquitectura serverless AWS: reporte de verificación

Fecha de ejecución: 2026-08-07 · Región simulada: `us-east-1` · Endpoint: `http://localhost:4566`

---

## 1. Estado del emulador

**Floci no pudo ejecutarse en esta máquina.** Floci es real ([floci.io](https://floci.io/aws/),
MIT, drop-in de LocalStack en el puerto 4566), pero su CLI orquesta **contenedores Docker**
para Lambda, ECS y RDS, y en este equipo:

| Requisito | Estado |
|---|---|
| Docker Desktop | ✘ no instalado |
| `podman` | ✘ no encontrado |
| CLI `floci` | ✘ no instalado |
| Java (runtime de Floci) | ✔ presente |

La verificación se ejecutó por tanto contra **`moto` 5.2.2 en modo servidor**, un emulador
AWS en Python puro que no requiere Docker y expone la misma API en el mismo puerto.
**El código no cambia**: son llamadas `boto3` con `endpoint_url`, idénticas contra Floci,
moto o AWS real. Para correrlo sobre Floci basta arrancarlo y repetir los mismos comandos.

```bash
python -m moto.server -p 4566      # emulador usado en esta verificación
python -m IAC.verify_floci         # diagnóstico servicio por servicio
```

---

## 2. Recursos provisionados

Ejecutado con `python -m IAC.deploy_local_cloud` (idempotente, se puede repetir).

### S3 — data lake de 4 capas
| Bucket | Cifrado | Ciclo de vida |
|---|---|---|
| `perabank-bronze-datalake` | AES256 | → GLACIER a los 30 días |
| `perabank-silver-datalake` | AES256 | purga de versiones no vigentes a 30 días |
| `perabank-gold-datalake` | AES256 | purga de versiones no vigentes a 90 días |
| `perabank-ml-artifacts` | AES256 | purga de versiones no vigentes a 180 días |

Acceso público bloqueado en los cuatro. Particionado Hive `year=/month=/day=`.

### SQS — ingesta FIFO
- `perabank-transaction-ingest.fifo` · deduplicación por contenido · visibilidad 30 s
- `perabank-transaction-dlq.fifo` · retención 14 días · redrive tras **3** intentos

### DynamoDB — `PAY_PER_REQUEST`
- `PeraBank_Customer_Features` · PK `sk_cliente` (S)
- `PeraBank_Fraud_Alerts` · PK `id_evento_tarjeta` (S) · GSI `gsi_macro_risk`
  (HASH `banda_riesgo_macro`, RANGE `monto_usd`, proyección `KEYS_ONLY`)

### IAM — privilegio mínimo
`PeraBank_Lambda_Role` + `PeraBank_Lambda_LeastPrivilege`:
**6 sentencias, 13 ARNs enumerados, 0 comodines** (`auditar_comodines()` falla el
despliegue si aparece `Resource: "*"`).

### Lambda y API Gateway
| Función | Zip | Memoria | Timeout |
|---|---|---|---|
| `lambda_etl_trigger` | 1.8 KB | 256 MB | 30 s |
| `lambda_risk_scorer` | 2.5 KB | 1024 MB | 60 s |

API HTTP `perabank-http-api` con `POST /v1/risk/score` y `GET /v1/market/trm`.

---

## 3. Simulación end-to-end — 8/8 pasos

```
✔ Artefacto ML publicado en S3                 31.8 MB -> models/perabank_risk_pipeline_v1.joblib
✔ Transacción enviada a la cola FIFO           id 98478dde-0d91-4012
✔ 3 envíos idénticos dejan 1 mensaje en cola   profundidad=1
✔ ETL aterrizó el objeto en bronze             raw/transacciones/year=2026/month=08/day=07/TX-SIM-000001.json
✔ Objeto particionado Hive en S3               1 objeto(s) bajo raw/transacciones/
✔ Features del cliente en DynamoDB             segmento Patrimonio medio
✔ POST /v1/risk/score devolvió HTTP 200        riesgo BAJO p=0.0 motor=modelo_ml
✔ Alerta de fraude registrada                  banda BAJO
```

`motor=modelo_ml` confirma que el pipeline real de scikit-learn se descargó desde S3
y puntuó; no se usó el respaldo por reglas.

Compactación bronze → silver: **26 objetos → 1 Parquet, 25 PUT evitados**.

### Qué NO se ejerció
Los handlers se invocan **en proceso**, no dentro de contenedores Lambda, y las rutas se
prueban llamando al handler en vez de hacer HTTP contra el API Gateway. Ejecutar Lambda
como contenedor y enrutar HTTP real requiere Docker. La lógica de los handlers y todas
las llamadas a S3, SQS y DynamoDB sí son reales contra el emulador.

---

## 4. Costo mensual estimado en AWS real

| Concepto | 100k tx | 500k tx | 2M tx |
|---|---:|---:|---:|
| S3 almacenamiento | $0.0120 | $0.0120 | $0.0120 |
| **S3 peticiones PUT** | **$0.4900** | **$2.4900** | **$9.9900** |
| Lambda invocaciones | $0.0000 | $0.0000 | $0.2800 |
| Lambda cómputo | $0.0000 | $0.0000 | $0.0000 |
| DynamoDB escrituras | $0.1250 | $0.6250 | $2.5000 |
| DynamoDB lecturas | $0.0050 | $0.0250 | $0.1000 |
| SQS peticiones | $0.0000 | $0.2000 | $2.0000 |
| API Gateway HTTP | $0.0000 | $0.0000 | $0.0000 |
| **TOTAL** | **$0.63** ✔ | **$3.35** ✔ | **$14.88** ✘ |

**El objetivo de <$5/mes se cumple hasta ~500k transacciones.**

El hallazgo que importa: **los PUT de S3 son el 74% del costo a 500k tx** y el que rompe
el presupuesto a 2M. No es el almacenamiento (centavos), es escribir *un objeto por
transacción*. `IAC/silver_compaction.py` ataca exactamente eso agrupando por partición;
llevar esa lógica al ETL (bufferizar y escribir un objeto por lote de N transacciones en
vez de uno por transacción) es lo que mantiene el diseño bajo $5 más allá del millón.

---

## 5. Salvaguarda contra despliegue accidental en AWS real

El `.env` del proyecto contiene credenciales AWS reales. Si un script de IaC corriera sin
endpoint local, `boto3` las tomaría de la cadena de credenciales y crearía infraestructura
**real, con costo real**. `IAC/floci_config.py` lo bloquea en dos capas:

1. Inyecta credenciales ficticias (`test`/`test`) en cada cliente.
2. `_validar_endpoint()` lanza `EndpointNoLocalError` si el host no es local, **antes** de
   construir el cliente.

---

## 6. Cómo reproducir

```bash
python -m moto.server -p 4566     # o: floci start (requiere Docker)
python -m IAC.verify_floci
python -m IAC.deploy_local_cloud
python -m IAC.simulate_pipeline
python -m IAC.silver_compaction
python -m IAC.cost_model
```
