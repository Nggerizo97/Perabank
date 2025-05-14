import boto3
import pandas as pd
import io
import logging
from datetime import datetime
import unidecode # Asegúrate de que esta librería esté en tu paquete Lambda o Layer
from dotenv import load_dotenv # Para desarrollo local
import os
from urllib.parse import urlparse
import json

# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Cargar variables de entorno solo en desarrollo local
try:
    # Ajusta la ruta si tu script está en una subcarpeta del proyecto principal donde está .env
    # Por ejemplo, si tu lambda está en 'lambdas/process_users/lambda_function.py'
    # y .env está en la raíz del proyecto.
    # env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env') # Sube dos niveles
    env_path = os.path.join(os.path.dirname(__file__), '.env') # Si .env está en la misma carpeta
    if not os.path.exists(env_path): # Intenta una ruta común si está en una subcarpeta 'src' o similar
         env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

    if os.path.exists(env_path):
        logger.info(f"Cargando .env desde: {env_path}")
        load_dotenv(env_path)
    else:
        logger.info("Archivo .env no encontrado en rutas comunes, asumiendo entorno Lambda o variables ya cargadas.")
except Exception as e:
    logger.info(f"No se pudo cargar .env (esto es normal en Lambda): {e}")

# Nombres exactos de los buckets (definidos como variables)
BRONZE_BUCKET = os.getenv('AWS_S3_BUCKET_BRONCE')
SILVER_BUCKET = os.getenv('AWS_S3_BUCKET_SILVER')
GOLD_BUCKET = os.getenv('AWS_S3_BUCKET_GOLD')

def get_s3_client():
    """Obtiene cliente S3 configurado según el entorno"""
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ: # Indicador más fiable de entorno Lambda
        logger.info("Entorno Lambda detectado, usando rol IAM para cliente S3.")
        return boto3.client('s3')
    else:
        logger.info("Entorno local detectado, intentando usar credenciales de .env para cliente S3.")
        # Para desarrollo local con .env
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1') # Default si no está en .env

        if not aws_access_key or not aws_secret_key:
            logger.warning("Credenciales AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) no encontradas en .env para desarrollo local. Boto3 intentará encontrarlas de otras formas (ej. ~/.aws/credentials).")
            # Si no hay credenciales en .env, boto3 intentará otros métodos (ej. shared credentials file)
            return boto3.client('s3', region_name=aws_region if aws_region else None)

        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )

# --- FUNCIÓN DE RUTAS CORREGIDA ---
def generate_s3_output_path(input_s3_path, target_bucket_name, suffix_and_new_extension):
    """
    Genera una ruta de salida S3 en el bucket de destino, manteniendo la
    estructura de carpetas original (usando '/') y aplicando un sufijo y nueva extensión al nombre del archivo.

    Args:
        input_s3_path: Ruta S3 completa al archivo de entrada (ej. "s3://bronze-bucket/usuarios/usuarios.csv")
        target_bucket_name: Nombre del bucket S3 de destino (ej. "perabank-silver-data-bank")
        suffix_and_new_extension: Sufijo a añadir al nombre base y la nueva extensión
                                  (ej. "_cleaned.parquet" o "_enriched.parquet")
    """
    try:
        parsed_url = urlparse(input_s3_path)
        if not parsed_url.scheme == 's3':
            raise ValueError(f"Input path '{input_s3_path}' no es una ruta S3 válida.")
        
        original_key = parsed_url.path.lstrip('/')
        if not original_key:
            raise ValueError(f"La ruta del objeto (key) está vacía en '{input_s3_path}'.")

        directory_path, original_filename = os.path.split(original_key) # Esto podría usar \ en Windows
        
        if not original_filename:
             raise ValueError(f"No se pudo extraer el nombre del archivo de la ruta '{original_key}'.")

        base_filename, _ = os.path.splitext(original_filename)
        
        new_filename = base_filename + suffix_and_new_extension
        
        # Unir y luego corregir separadores
        temp_key = os.path.join(directory_path, new_filename) if directory_path else new_filename
        final_key = temp_key.replace('\\', '/') # Asegura que se usen barras diagonales

        return f"s3://{target_bucket_name}/{final_key}"
    except Exception as e:
        logger.error(f"Error generando ruta de salida para '{input_s3_path}', target_bucket '{target_bucket_name}', suffix '{suffix_and_new_extension}': {e}")
        raise


def lambda_handler(event, context):
    """Procesa datos de usuarios desde Bronze a Silver y Gold"""
    start_time = datetime.now()
    logger.info(f"Iniciando procesamiento de datos. Evento recibido: {event}")
    
    s3 = get_s3_client() # Mover la inicialización del cliente aquí

    try:
        # --- MODIFICACIÓN: Obtener bucket y key del evento S3 si es un trigger S3 ---
        # O usar 'input_path' si se pasa directamente (ej. desde Step Functions o test manual)
        if 'Records' in event and event['Records'] and 's3' in event['Records'][0]:
            source_bucket = event['Records'][0]['s3']['bucket']['name']
            source_key = event['Records'][0]['s3']['object']['key']
            input_path = f"s3://{source_bucket}/{source_key}"
            logger.info(f"Lambda disparada por evento S3. Bucket: {source_bucket}, Key: {source_key}")
        elif 'input_path' in event:
            input_path = event['input_path']
            parsed_url = urlparse(input_path)
            source_bucket = parsed_url.netloc
            source_key = parsed_url.path.lstrip('/')
            logger.info(f"Lambda disparada con input_path: {input_path}")
        else:
            # Fallback o default para pruebas si ninguna de las anteriores se cumple
            # Es mejor que falle si no se provee la ruta de forma esperada
            logger.error("No se encontró 'Records' (evento S3) ni 'input_path' en el evento.")
            raise ValueError("La ruta del archivo de entrada no fue especificada en el evento.")

        logger.info(f"Ruta de entrada S3 a procesar: {input_path}")
        
        # Generar rutas de salida usando la función corregida
        silver_path = generate_s3_output_path(input_path, SILVER_BUCKET, '_cleaned.parquet')
        gold_path = generate_s3_output_path(input_path, GOLD_BUCKET, '_enriched.parquet')
        
        logger.info(f"Ruta de salida para Silver: {silver_path}")
        logger.info(f"Ruta de salida para Gold: {gold_path}")
        
        logger.info(f"Descargando datos desde: s3://{source_bucket}/{source_key}")
        response = s3.get_object(Bucket=source_bucket, Key=source_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        logger.info(f"Datos descargados. Filas: {len(df)}, Columnas: {list(df.columns)}")
        
        logger.info("Transformando a Silver...")
        df_silver = clean_and_transform_data(df.copy()) # Usar .copy() para evitar modificar df original si se reutiliza
        upload_to_s3(df_silver, silver_path, s3)
        
        logger.info("Transformando a Gold...")
        # df_gold va a operar sobre df_silver, que ya tiene algunas transformaciones
        df_gold = create_gold_dataset(df_silver.copy()) # Usar .copy()
        upload_to_s3(df_gold, gold_path, s3)
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"Procesamiento completado en {duration:.2f} segundos")
        
        return {
            'statusCode': 200,
            'body': json.dumps({ # Convertir el body a JSON string
                'processed_rows': len(df),
                'source': input_path,
                'silver': silver_path,
                'gold': gold_path,
                'execution_time': f"{duration:.2f} segundos"
            })
        }
        
    except Exception as e:
        logger.error(f"Error en el procesamiento para input_path '{locals().get('input_path', 'Desconocido')}': {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': f"Error procesando datos: {str(e)}"}) # Convertir el body a JSON string
        }

def upload_to_s3(df, s3_path, s3_client):
    """Sube DataFrame a S3 como Parquet"""
    parsed = urlparse(s3_path)
    target_bucket = parsed.netloc
    target_key = parsed.path.lstrip('/')

    if not target_bucket or not target_key:
        raise ValueError(f"Bucket o Key de destino no válidos en la ruta S3: '{s3_path}'")

    buffer = io.BytesIO()
    # Asegúrate de que pyarrow esté disponible en el entorno Lambda
    df.to_parquet(buffer, engine='pyarrow', index=False) 
    buffer.seek(0) # Regresar al inicio del buffer antes de leerlo para PutObject
    
    s3_client.put_object(
        Bucket=target_bucket,
        Key=target_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Datos guardados en {s3_path}")

# --- Tus funciones de transformación (clean_and_transform_data, categorizar_estado_laboral, etc.) ---
# --- Se mantienen igual que las que proporcionaste, asumiendo que están correctas ---
# --- Solo asegúrate de que 'unidecode' y 'pyarrow' estén disponibles ---

def clean_and_transform_data(df):
    logger.info("Iniciando clean_and_transform_data")
    text_columns = ['nombre', 'apellido', 'ciudad_residencia', 'profesion', 'estado_laboral', 
                    'nivel_educativo', 'estado_civil', 'tipo_vivienda']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: unidecode.unidecode(str(x)).strip().upper() if pd.notna(x) else x)
    
    date_columns = ['fecha_de_nacimiento', 'fecha_registro_banco']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Dejar como datetime si Parquet lo maneja bien, o convertir a string si se prefiere
            # df[col] = df[col].dt.strftime('%Y-%m-%d') 
    
    if 'ingreso_mensual_estimado' in df.columns:
        df['ingreso_mensual_estimado'] = pd.to_numeric(df['ingreso_mensual_estimado'], errors='coerce').fillna(0)
    
    if 'numero_dependientes' in df.columns:
        df['numero_dependientes'] = pd.to_numeric(df['numero_dependientes'], errors='coerce').fillna(0).astype(int)
    
    if 'estado_laboral' in df.columns:
        df['estado_laboral'] = df['estado_laboral'].apply(categorizar_estado_laboral)
    
    if 'nivel_educativo' in df.columns:
        df['nivel_educativo'] = df['nivel_educativo'].apply(categorizar_nivel_educativo)
    
    # Crear columna de edad (asegurarse que fecha_de_nacimiento es datetime)
    if 'fecha_de_nacimiento' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_de_nacimiento']):
        # Calcular la edad con más precisión si es posible
        now = pd.to_datetime('now', utc=True) if df['fecha_de_nacimiento'].dt.tz else pd.to_datetime('now')
        df['edad'] = (now - df['fecha_de_nacimiento']).dt.days // 365.25
        df['edad'] = df['edad'].astype(int)
    elif 'fecha_de_nacimiento' in df.columns: # Si es string y se convirtió antes
        df['fecha_de_nacimiento'] = pd.to_datetime(df['fecha_de_nacimiento'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['fecha_de_nacimiento']):
            now = pd.to_datetime('now', utc=True) if df['fecha_de_nacimiento'].dt.tz else pd.to_datetime('now')
            df['edad'] = (now - df['fecha_de_nacimiento']).dt.days // 365.25
            df['edad'] = df['edad'].astype(int)
        else:
            df['edad'] = None # O algún valor por defecto si la conversión falla
            logger.warning("No se pudo calcular la edad, fecha_de_nacimiento no es convertible a datetime")


    # Antigüedad como cliente (en años)
    if 'fecha_registro_banco' in df.columns and pd.api.types.is_datetime64_any_dtype(df['fecha_registro_banco']):
        now = pd.to_datetime('now', utc=True) if df['fecha_registro_banco'].dt.tz else pd.to_datetime('now')
        df['antiguedad_cliente'] = (now - df['fecha_registro_banco']).dt.days // 365.25
        df['antiguedad_cliente'] = df['antiguedad_cliente'].astype(int)
    elif 'fecha_registro_banco' in df.columns: # Si es string y se convirtió antes
        df['fecha_registro_banco'] = pd.to_datetime(df['fecha_registro_banco'], errors='coerce')
        if pd.api.types.is_datetime64_any_dtype(df['fecha_registro_banco']):
            now = pd.to_datetime('now', utc=True) if df['fecha_registro_banco'].dt.tz else pd.to_datetime('now')
            df['antiguedad_cliente'] = (now - df['fecha_registro_banco']).dt.days // 365.25
            df['antiguedad_cliente'] = df['antiguedad_cliente'].astype(int)
        else:
            df['antiguedad_cliente'] = None
            logger.warning("No se pudo calcular la antiguedad_cliente, fecha_registro_banco no es convertible a datetime")

    logger.info(f"Columnas después de clean_and_transform_data: {list(df.columns)}")
    return df

def categorizar_estado_laboral(valor):
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    if 'EMPLEADO' in valor: return 'EMPLEADO'
    if 'INDEPENDIENTE' in valor: return 'INDEPENDIENTE'
    if 'DESEMPLEADO' in valor: return 'DESEMPLEADO'
    if 'PENSIONADO' in valor: return 'PENSIONADO'
    if 'ESTUDIANTE' in valor: return 'ESTUDIANTE'
    return 'OTRO'

def categorizar_nivel_educativo(valor):
    if pd.isna(valor): return 'DESCONOCIDO'
    valor = str(valor).upper()
    if 'NINGUNO' in valor: return 'NINGUNO'
    if 'PRIMARIA' in valor: return 'PRIMARIA'
    if 'SECUNDARIA' in valor: return 'SECUNDARIA'
    if 'TECNICO' in valor or 'TÉCNICO' in valor: return 'TECNICO'
    if 'UNIVERSITARIO' in valor: return 'UNIVERSITARIO'
    if 'POSTGRADO' in valor: return 'POSTGRADO'
    return 'OTRO'

def create_gold_dataset(df_silver): # Recibe el df_silver
    logger.info("Iniciando create_gold_dataset")
    df_gold = df_silver.copy()
    
    if 'ingreso_mensual_estimado' in df_gold.columns:
        df_gold['categoria_ingresos'] = pd.cut(
            df_gold['ingreso_mensual_estimado'],
            bins=[-float('inf'), 1000000, 3000000, 6000000, 10000000, float('inf')], # Ajustar el primer bin
            labels=['MUY BAJOS', 'BAJOS', 'MEDIOS', 'ALTOS', 'MUY ALTOS'],
            right=False # Para que el límite inferior sea inclusivo
        )
    else:
        logger.warning("'ingreso_mensual_estimado' no encontrado para crear 'categoria_ingresos'")

    if 'numero_dependientes' in df_gold.columns:
        df_gold['tiene_dependientes'] = (df_gold['numero_dependientes'] > 0).astype(int)
    else:
        logger.warning("'numero_dependientes' no encontrado para crear 'tiene_dependientes'")
        
    if 'tipo_vivienda' in df_gold.columns:
        df_gold['vivienda_propia'] = (df_gold['tipo_vivienda'] == 'PROPIA').astype(int) # Asumiendo que ya está en mayúsculas
    else:
        logger.warning("'tipo_vivienda' no encontrado para crear 'vivienda_propia'")
    
    if 'edad' in df_gold.columns and pd.api.types.is_numeric_dtype(df_gold['edad']):
        df_gold['rango_edad'] = pd.cut(
            df_gold['edad'],
            bins=[0, 25, 35, 45, 55, 65, float('inf')], # Ajustar el último bin
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],
            right=False
        )
    else:
        logger.warning("'edad' no encontrado o no es numérico para crear 'rango_edad'")

    # Seleccionar columnas finales para Gold (definidas como antes)
    columns_gold = [
        'user_id', 'cedula', 'nombre', 'apellido', 'ciudad_residencia', 
        'fecha_de_nacimiento', 'edad', 'rango_edad', 'profesion',
        'ingreso_mensual_estimado', 'categoria_ingresos', 'estado_laboral',
        'nivel_educativo', 'estado_civil', 'numero_dependientes', 'tiene_dependientes',
        'tipo_vivienda', 'vivienda_propia', 'fecha_registro_banco', 'antiguedad_cliente'
    ]
    
    # Filtrar solo las columnas que existen en el DataFrame df_gold
    final_columns_gold = [col for col in columns_gold if col in df_gold.columns]
    logger.info(f"Columnas seleccionadas para Gold: {final_columns_gold}")
    
    return df_gold[final_columns_gold]

# --- Las funciones save_to_silver y save_to_gold no se usan directamente ---
# --- si generate_s3_output_path y upload_to_s3 se usan como en el handler. ---
# --- Puedes borrarlas si no las vas a llamar. ---

# def save_to_silver(df, bucket_silver, original_key, s3_client): ...
# def save_to_gold(df_gold, bucket_gold, original_key, s3_client): ...