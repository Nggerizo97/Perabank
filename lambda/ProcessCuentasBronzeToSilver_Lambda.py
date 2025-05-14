# ProcessCuentasBronzeToSilver_Lambda.py
import boto3
import pandas as pd
import io
import logging
from datetime import datetime
import unidecode
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
import json

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cargar variables de entorno
try:
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if not os.path.exists(env_path):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

    if os.path.exists(env_path):
        logger.info(f"Cargando .env desde: {env_path}")
        load_dotenv(env_path)
    else:
        logger.info("Archivo .env no encontrado, asumiendo entorno Lambda")
except Exception as e:
    logger.info(f"No se pudo cargar .env: {e}")

# Buckets S3
BRONZE_BUCKET = os.getenv('AWS_S3_BUCKET_BRONCE')
SILVER_BUCKET = os.getenv('AWS_S3_BUCKET_SILVER')
GOLD_BUCKET = os.getenv('AWS_S3_BUCKET_GOLD')

def get_s3_client():
    """Obtiene cliente S3 configurado según el entorno"""
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        return boto3.client('s3')
    else:
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        
        if not aws_access_key or not aws_secret_key:
            logger.warning("Credenciales AWS no encontradas en .env")
            return boto3.client('s3', region_name=aws_region if aws_region else None)

        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

def generate_s3_output_path(input_s3_path, target_bucket_name, suffix_and_new_extension):
    """Genera ruta de salida S3 manteniendo estructura de carpetas"""
    try:
        parsed_url = urlparse(input_s3_path)
        if not parsed_url.scheme == 's3':
            raise ValueError(f"Input path '{input_s3_path}' no es una ruta S3 válida.")
        
        original_key = parsed_url.path.lstrip('/')
        directory_path, original_filename = os.path.split(original_key)
        
        if not original_filename:
            raise ValueError(f"No se pudo extraer el nombre del archivo de la ruta '{original_key}'.")

        base_filename, _ = os.path.splitext(original_filename)
        new_filename = base_filename + suffix_and_new_extension
        temp_key = os.path.join(directory_path, new_filename) if directory_path else new_filename
        final_key = temp_key.replace('\\', '/')

        return f"s3://{target_bucket_name}/{final_key}"
    except Exception as e:
        logger.error(f"Error generando ruta de salida: {e}")
        raise

def lambda_handler(event, context):
    """Procesa datos de cuentas desde Bronze a Silver y Gold"""
    start_time = datetime.now()
    logger.info(f"Iniciando procesamiento. Evento recibido: {event}")
    
    s3 = get_s3_client()

    try:
        # Obtener bucket y key del evento
        if 'Records' in event and event['Records'] and 's3' in event['Records'][0]:
            source_bucket = event['Records'][0]['s3']['bucket']['name']
            source_key = event['Records'][0]['s3']['object']['key']
            input_path = f"s3://{source_bucket}/{source_key}"
        elif 'input_path' in event:
            input_path = event['input_path']
            parsed_url = urlparse(input_path)
            source_bucket = parsed_url.netloc
            source_key = parsed_url.path.lstrip('/')
        else:
            logger.error("No se encontró 'Records' ni 'input_path' en el evento.")
            raise ValueError("Ruta del archivo de entrada no especificada.")

        logger.info(f"Ruta de entrada S3: {input_path}")
        
        # Generar rutas de salida
        silver_path = generate_s3_output_path(input_path, SILVER_BUCKET, '_cleaned.parquet')
        gold_path = generate_s3_output_path(input_path, GOLD_BUCKET, '_enriched.parquet')
        
        logger.info(f"Ruta Silver: {silver_path}")
        logger.info(f"Ruta Gold: {gold_path}")
        
        # Descargar y procesar datos
        logger.info(f"Descargando datos desde: s3://{source_bucket}/{source_key}")
        response = s3.get_object(Bucket=source_bucket, Key=source_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        logger.info(f"Datos descargados. Filas: {len(df)}, Columnas: {list(df.columns)}")
        
        # Transformaciones
        logger.info("Transformando a Silver...")
        df_silver = clean_and_transform_accounts(df.copy())
        upload_to_s3(df_silver, silver_path, s3)
        
        logger.info("Transformando a Gold...")
        df_gold = create_gold_accounts(df_silver.copy())
        upload_to_s3(df_gold, gold_path, s3)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Procesamiento completado en {duration:.2f} segundos")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed_rows': len(df),
                'source': input_path,
                'silver': silver_path,
                'gold': gold_path,
                'execution_time': f"{duration:.2f} segundos"
            })
        }
        
    except Exception as e:
        logger.error(f"Error en el procesamiento: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Error procesando datos: {str(e)}"})
        }

def upload_to_s3(df, s3_path, s3_client):
    """Sube DataFrame a S3 como Parquet"""
    parsed = urlparse(s3_path)
    target_bucket = parsed.netloc
    target_key = parsed.path.lstrip('/')

    buffer = io.BytesIO()
    df.to_parquet(buffer, engine='pyarrow', index=False) 
    buffer.seek(0)
    
    s3_client.put_object(
        Bucket=target_bucket,
        Key=target_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Datos guardados en {s3_path}")

def clean_and_transform_accounts(df):
    """Limpieza y transformación para nivel Silver"""
    logger.info("Iniciando clean_and_transform_accounts")
    
    # Limpieza de texto
    text_columns = ['tipo_cuenta', 'moneda', 'estado_cuenta']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: unidecode.unidecode(str(x)).strip().upper() if pd.notna(x) else x)
    
    # Convertir fechas
    if 'fecha_apertura' in df.columns:
        df['fecha_apertura'] = pd.to_datetime(df['fecha_apertura'], errors='coerce')
    
    # Limpiar montos
    if 'saldo_actual' in df.columns:
        df['saldo_actual'] = pd.to_numeric(df['saldo_actual'], errors='coerce')
    
    # Normalizar tipos de cuenta
    if 'tipo_cuenta' in df.columns:
        df['tipo_cuenta'] = df['tipo_cuenta'].apply(normalizar_tipo_cuenta)
    
    # Normalizar estados de cuenta
    if 'estado_cuenta' in df.columns:
        df['estado_cuenta'] = df['estado_cuenta'].apply(normalizar_estado_cuenta)
    
    # Enmascarar número de cuenta (mostrar solo últimos 4 dígitos)
    if 'numero_cuenta' in df.columns:
        df['numero_cuenta_masked'] = df['numero_cuenta'].apply(lambda x: '****' + str(x)[-4:] if pd.notna(x) and len(str(x)) > 4 else str(x))
    
    # Extraer componentes de fecha
    if 'fecha_apertura' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_apertura']):
        df['anio_apertura'] = df['fecha_apertura'].dt.year
        df['mes_apertura'] = df['fecha_apertura'].dt.month
        df['dia_apertura'] = df['fecha_apertura'].dt.day
    
    logger.info(f"Columnas después de clean_and_transform_accounts: {list(df.columns)}")
    return df

def normalizar_tipo_cuenta(valor):
    """Normaliza los tipos de cuenta"""
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    
    if 'CORRIENTE' in valor: return 'CORRIENTE'
    if 'AHORROS' in valor: return 'AHORROS'
    if 'NÓMINA' in valor or 'NOMINA' in valor: return 'NOMINA'
    
    return valor

def normalizar_estado_cuenta(valor):
    """Normaliza los estados de cuenta"""
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    
    if 'ACTIV' in valor: return 'ACTIVA'
    if 'INACTIV' in valor: return 'INACTIVA'
    if 'BLOQUEAD' in valor: return 'BLOQUEADA'
    
    return valor

def create_gold_accounts(df_silver):
    """Transformación para nivel Gold"""
    logger.info("Iniciando create_gold_accounts")
    df_gold = df_silver.copy()
    
    # Agregar categorización de saldos
    if 'saldo_actual' in df_gold.columns:
        df_gold['categoria_saldo'] = pd.cut(
            df_gold['saldo_actual'],
            bins=[-float('inf'), 1000000, 5000000, 10000000, 20000000, float('inf')],
            labels=['MUY_BAJO', 'BAJO', 'MEDIO', 'ALTO', 'MUY_ALTO'],
            right=False
        )
    
    # Agregar antigüedad de la cuenta en años
    if 'fecha_apertura' in df_gold.columns:
        df_gold['antiguedad_anios'] = ((datetime.now() - df_gold['fecha_apertura']).dt.days / 365).round(1)
    
    # Agregar indicador de cuenta protegida (ahorros)
    if 'tipo_cuenta' in df_gold.columns:
        df_gold['cuenta_protegida'] = (df_gold['tipo_cuenta'] == 'AHORROS').astype(int)
    
    # Agregar indicador de cuenta bloqueada
    if 'estado_cuenta' in df_gold.columns:
        df_gold['cuenta_bloqueada'] = (df_gold['estado_cuenta'] == 'BLOQUEADA').astype(int)
    
    # Seleccionar columnas finales para Gold
    columns_gold = [
        'account_id', 'user_id', 'numero_cuenta_masked', 'tipo_cuenta', 
        'fecha_apertura', 'anio_apertura', 'mes_apertura', 'dia_apertura',
        'antiguedad_anios', 'saldo_actual', 'categoria_saldo', 
        'moneda', 'estado_cuenta', 'cuenta_protegida', 'cuenta_bloqueada'
    ]
    
    # Filtrar solo las columnas que existen
    final_columns_gold = [col for col in columns_gold if col in df_gold.columns]
    logger.info(f"Columnas seleccionadas para Gold: {final_columns_gold}")
    
    return df_gold[final_columns_gold]