# ProcessDestinatariosBronzeToSilver_Lambda.py
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
    """Procesa datos de destinatarios desde Bronze a Silver y Gold"""
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
        df_silver = clean_and_transform_recipients(df.copy())
        upload_to_s3(df_silver, silver_path, s3)
        
        logger.info("Transformando a Gold...")
        df_gold = create_gold_recipients(df_silver.copy())
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

def clean_and_transform_recipients(df):
    """Limpieza y transformación para nivel Silver"""
    logger.info("Iniciando clean_and_transform_recipients")
    
    # Limpieza de texto
    text_columns = ['nombre_registrado', 'banco_destinatario', 'tipo_cuenta_dest']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: unidecode.unidecode(str(x)).strip().upper() if pd.notna(x) else x)
    
    # Normalizar nombres de bancos
    if 'banco_destinatario' in df.columns:
        df['banco_destinatario'] = df['banco_destinatario'].apply(normalizar_banco)
    
    # Normalizar tipos de cuenta
    if 'tipo_cuenta_dest' in df.columns:
        df['tipo_cuenta_dest'] = df['tipo_cuenta_dest'].apply(normalizar_tipo_cuenta)
    
    # Extraer componentes del nombre
    if 'nombre_registrado' in df.columns:
        df['primer_nombre'] = df['nombre_registrado'].apply(extraer_primer_nombre)
        df['primer_apellido'] = df['nombre_registrado'].apply(extraer_primer_apellido)
    
    # Enmascarar número de cuenta (mostrar solo últimos 4 dígitos)
    if 'numero_cuenta_dest' in df.columns:
        df['numero_cuenta_masked'] = df['numero_cuenta_dest'].apply(lambda x: '****' + str(x)[-4:] if pd.notna(x) and len(str(x)) > 4 else str(x))
    
    logger.info(f"Columnas después de clean_and_transform_recipients: {list(df.columns)}")
    return df

def normalizar_banco(valor):
    """Normaliza los nombres de bancos"""
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    
    if 'DAVIPLATA' in valor: return 'DAVIPLATA'
    if 'DAVIVIENDA' in valor: return 'DAVIVIENDA'
    if 'BOGOTA' in valor or 'BOGOTÁ' in valor: return 'BANCO_DE_BOGOTA'
    if 'BANCOLOMBIA' in valor: return 'BANCOLOMBIA'
    if 'ITAU' in valor or 'ITÁU' in valor: return 'ITAU'
    if 'NEQUI' in valor: return 'NEQUI'
    if 'SCOTIABANK' in valor or 'COLPATRIA' in valor: return 'SCOTIABANK_COLPATRIA'
    if 'AGRARIO' in valor: return 'BANCO_AGRARIO'
    if 'BBVA' in valor: return 'BBVA'
    
    return valor

def normalizar_tipo_cuenta(valor):
    """Normaliza los tipos de cuenta"""
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    
    if 'CORRIENTE' in valor: return 'CORRIENTE'
    if 'AHORRO' in valor: return 'AHORROS'
    
    return valor

def extraer_primer_nombre(nombre_completo):
    """Extrae el primer nombre del nombre completo"""
    if pd.isna(nombre_completo): return None
    return str(nombre_completo).split()[0]

def extraer_primer_apellido(nombre_completo):
    """Extrae el primer apellido del nombre completo"""
    if pd.isna(nombre_completo): return None
    partes = str(nombre_completo).split()
    return partes[1] if len(partes) > 1 else None

def create_gold_recipients(df_silver):
    """Transformación para nivel Gold"""
    logger.info("Iniciando create_gold_recipients")
    df_gold = df_silver.copy()
    
    # Agregar categorización de bancos (nacionales vs. digitales)
    if 'banco_destinatario' in df_gold.columns:
        bancos_digitales = ['DAVIPLATA', 'NEQUI']
        df_gold['tipo_banco'] = df_gold['banco_destinatario'].apply(
            lambda x: 'DIGITAL' if x in bancos_digitales else 'TRADICIONAL'
        )
    
    # Agregar indicador de cuenta protegida (basado en tipo de cuenta)
    if 'tipo_cuenta_dest' in df_gold.columns:
        df_gold['cuenta_protegida'] = (df_gold['tipo_cuenta_dest'] == 'AHORROS').astype(int)
    
    # Seleccionar columnas finales para Gold
    columns_gold = [
        'recipient_id', 'nombre_registrado', 'primer_nombre', 'primer_apellido',
        'banco_destinatario', 'tipo_banco', 'tipo_cuenta_dest', 'cuenta_protegida',
        'numero_cuenta_masked'
    ]
    
    # Filtrar solo las columnas que existen
    final_columns_gold = [col for col in columns_gold if col in df_gold.columns]
    logger.info(f"Columnas seleccionadas para Gold: {final_columns_gold}")
    
    return df_gold[final_columns_gold]