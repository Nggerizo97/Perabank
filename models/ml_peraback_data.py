import boto3
import pandas as pd
import io
import logging
from datetime import datetime # No se usa directamente en este snippet, pero es bueno tenerlo
import unidecode # Lo necesitarás si tus funciones de limpieza lo usan y las llamas aquí
from dotenv import load_dotenv
import os
from urllib.parse import urlparse

# --- Configuración Inicial como en tu script ---
# Configurar logging
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Puedes cambiar a logging.DEBUG para más detalle

# Cargar variables de entorno
try:
    # Intentar encontrar .env en la misma carpeta que este script
    env_path_current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    # Intentar encontrar .env un nivel arriba (común si el script está en una subcarpeta 'src' o 'scripts')
    env_path_parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')

    if os.path.exists(env_path_current_dir):
        env_path = env_path_current_dir
    elif os.path.exists(env_path_parent_dir):
        env_path = env_path_parent_dir
    else:
        env_path = None

    if env_path:
        logger.info(f"Cargando .env desde: {env_path}")
        load_dotenv(env_path)
        # Pequeña verificación
        # logger.debug(f"AWS_ACCESS_KEY_ID desde .env: {os.getenv('AWS_ACCESS_KEY_ID') is not None}")
    else:
        logger.info("Archivo .env no encontrado en rutas comunes, asumiendo entorno Lambda o credenciales ya configuradas globalmente.")
except Exception as e:
    logger.info(f"No se pudo cargar .env (esto es normal en Lambda si no se usa): {e}")

# Buckets S3 (usando los nombres de variables de entorno que definiste)
# Asegúrate de que estas variables estén en tu .env o configuradas en tu entorno
BRONZE_BUCKET = os.getenv('AWS_S3_BUCKET_BRONCE') # Default si no está en .env
SILVER_BUCKET = os.getenv('AWS_S3_BUCKET_SILVER') # Default si no está en .env
GOLD_BUCKET = os.getenv('AWS_S3_BUCKET_GOLD')     # Default si no está en .env
ML_ARTIFACTS_BUCKET = os.getenv('AWS_S3_BUCKET_ML') # Para guardar modelos

# --- Funciones de Ayuda (incluyendo las tuyas) ---
def get_s3_client():
    """Obtiene cliente S3 configurado según el entorno"""
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        logger.info("Entorno Lambda detectado, usando rol IAM para cliente S3.")
        return boto3.client('s3')
    else:
        logger.info("Entorno local detectado, usando credenciales configuradas para cliente S3.")
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_REGION', 'us-east-1') # Default si no está en .env
        
        # Boto3 es inteligente: si aws_access_key o aws_secret_key son None,
        # intentará otros métodos (ej. ~/.aws/credentials, variables de entorno de sesión, rol de instancia EC2).
        # Así que podemos pasar los valores de os.getenv() directamente.
        return boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region # region_name es importante
        )

# Tu función generate_s3_output_path (no la usaremos para leer, pero es bueno tenerla si este script también escribe)
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
        final_key = temp_key.replace('\\', '/') # Asegura barras diagonales para S3

        return f"s3://{target_bucket_name}/{final_key}"
    except Exception as e:
        logger.error(f"Error generando ruta de salida: {e}")
        raise

def load_parquet_from_s3(s3_client, bucket_name, file_key):
    """
    Lee un archivo Parquet específico desde S3 y lo carga en un DataFrame de Pandas.
    """
    s3_path = f"s3://{bucket_name}/{file_key}"
    logger.info(f"Intentando leer Parquet desde: {s3_path}")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()), engine='pyarrow')
        logger.info(f"Datos de '{file_key}' cargados exitosamente. Dimensiones: {df.shape}")
        # logger.debug(df.head()) # Descomenta para ver las primeras filas al cargar
        return df
    except Exception as e:
        logger.error(f"ERROR al leer el archivo Parquet '{s3_path}': {e}", exc_info=True)
        return pd.DataFrame() # Retorna DataFrame vacío en caso de error

# --- Script Principal para Cargar Datos de S3 Gold y Preparar para ML ---
if __name__ == "__main__":
    logger.info("--- Iniciando script de carga de datos S3 Gold para ML (Local) ---")
    
    s3 = get_s3_client()

    # Definir las claves (rutas dentro del bucket) para tus archivos Parquet en Gold
    # Basado en tu ejemplo: s3://perabank-gold-data-bank/cuentas/cuentas_enriched.parquet
    keys_gold = {
        "usuarios": "usuarios/usuarios_enriched.parquet",
        "cuentas": "cuentas/cuentas_enriched.parquet",
        "transacciones": "transacciones/transacciones_enriched.parquet",
        "prestamos": "prestamos/prestamos_enriched.parquet",
        "destinatarios": "destinatarios/destinatarios_enriched.parquet" # Asumiendo que también tienes esta
    }

    dataframes_gold = {}

    for entity_name, entity_key in keys_gold.items():
        logger.info(f"\n--- Cargando datos de: {entity_name} (Oro) ---")
        df = load_parquet_from_s3(s3, GOLD_BUCKET, entity_key)
        if df.empty:
            logger.warning(f"El DataFrame para '{entity_name}' está vacío. Verifica la ruta S3 y los logs de error.")
        else:
            logger.info(f"Primeras filas de '{entity_name}':")
            print(df.head()) # Imprime para verificar
        dataframes_gold[entity_name] = df

    # --- Unir y Agregar Datos para el DataFrame Final de Entrenamiento ---
    # Esta sección es crucial y requiere tu lógica de negocio y conocimiento del dominio.
    # El siguiente es un ESQUELETO conceptual que debes adaptar.

    logger.info("\n--- Iniciando el proceso de unión y agregación de DataFrames ---")
    
    df_final_ml_dataset = pd.DataFrame()

    if not dataframes_gold.get("usuarios", pd.DataFrame()).empty:
        df_final_ml_dataset = dataframes_gold["usuarios"].copy()
        logger.info(f"Dataset inicial para ML basado en usuarios: {df_final_ml_dataset.shape}")

        # Ejemplo de unión con cuentas y transacciones (requiere lógica de agregación)
        df_cuentas = dataframes_gold.get("cuentas", pd.DataFrame())
        df_transacciones = dataframes_gold.get("transacciones", pd.DataFrame())

        if not df_cuentas.empty and not df_transacciones.empty:
            logger.info("Procesando agregados de cuentas y transacciones...")
            # Asegúrate de que las columnas de monto sean numéricas
            df_transacciones['monto'] = pd.to_numeric(df_transacciones['monto'], errors='coerce').fillna(0)
            
            # Agregados de transacciones por account_id
            agg_trans_por_cuenta = df_transacciones.groupby('account_id').agg(
                num_total_transacciones_cuenta=('transaction_id', 'count'),
                monto_total_transacciones_cuenta=('monto', 'sum'),
                monto_promedio_transaccion_cuenta=('monto', 'mean'),
                # Puedes añadir más aquí, como frecuencia de categorías de transacciones, etc.
            ).reset_index()

            # Unir cuentas con sus transacciones agregadas
            df_cuentas_con_trans_agg = pd.merge(df_cuentas, agg_trans_por_cuenta, on='account_id', how='left')

            # Ahora agregar la información de cuentas (que ya tiene info de transacciones) por user_id
            # Asegúrate de que 'saldo_actual' sea numérico
            df_cuentas_con_trans_agg['saldo_actual'] = pd.to_numeric(df_cuentas_con_trans_agg['saldo_actual'], errors='coerce').fillna(0)

            agg_cuentas_por_usuario = df_cuentas_con_trans_agg.groupby('user_id').agg(
                num_total_cuentas=('account_id', 'nunique'),
                saldo_total_en_cuentas=('saldo_actual', 'sum'),
                promedio_transacciones_por_cuenta=('num_total_transacciones_cuenta', 'mean'),
                monto_total_movido_por_usuario=('monto_total_transacciones_cuenta', 'sum'),
                # ... más features derivadas de cuentas y transacciones ...
            ).reset_index()
            
            df_final_ml_dataset = pd.merge(df_final_ml_dataset, agg_cuentas_por_usuario, on='user_id', how='left')
            logger.info(f"Dimensiones después de unir cuentas/transacciones: {df_final_ml_dataset.shape}")


        # Ejemplo de unión con préstamos
        df_prestamos = dataframes_gold.get("prestamos", pd.DataFrame())
        if not df_prestamos.empty:
            logger.info("Procesando agregados de préstamos...")
            df_prestamos['monto_otorgado'] = pd.to_numeric(df_prestamos['monto_otorgado'], errors='coerce').fillna(0)
            # Asegúrate que 'plazo_meses' sea numérico también si lo usas
            # df_prestamos['plazo_meses'] = pd.to_numeric(df_prestamos['plazo_meses'], errors='coerce').fillna(0)

            agg_prestamos_por_usuario = df_prestamos.groupby('user_id').agg(
                num_total_prestamos=('loan_id', 'count'),
                monto_total_prestado=('monto_otorgado', 'sum'),
                promedio_monto_prestamo=('monto_otorgado', 'mean'),
                # Podrías añadir: ratio de préstamos en mora, promedio de plazo, etc.
            ).reset_index()
            df_final_ml_dataset = pd.merge(df_final_ml_dataset, agg_prestamos_por_usuario, on='user_id', how='left')
            logger.info(f"Dimensiones después de unir préstamos: {df_final_ml_dataset.shape}")


        # Rellenar NaNs que pudieron surgir de los left joins
        # (ej. usuarios sin préstamos, o sin transacciones, etc.)
        # Esta lista de columnas dependerá de las que hayas agregado
        columnas_para_rellenar_con_cero = [
            'num_total_cuentas', 'saldo_total_en_cuentas', 'promedio_transacciones_por_cuenta',
            'monto_total_movido_por_usuario', 'num_total_prestamos', 'monto_total_prestado',
            'promedio_monto_prestamo'
        ]
        for col in columnas_para_rellenar_con_cero:
            if col in df_final_ml_dataset.columns:
                df_final_ml_dataset[col] = df_final_ml_dataset[col].fillna(0)
            else:
                logger.warning(f"Columna '{col}' no encontrada para rellenar NaNs, puede que no se haya generado en los agregados.")

        logger.info("\n--- Vista Previa del Dataset Combinado Final para ML (Local) ---")
        if not df_final_ml_dataset.empty:
            print(df_final_ml_dataset.head())
            df_final_ml_dataset.info(verbose=True, show_counts=True) # Más detallado
            
            # Opcional: Guardar el dataset combinado localmente para no reprocesar cada vez
            # df_final_ml_dataset.to_parquet("PeraBank_ML_Training_Dataset_Local.parquet", index=False)
            # logger.info("Dataset combinado para ML guardado localmente como PeraBank_ML_Training_Dataset_Local.parquet")
        else:
            logger.error("El dataset final para ML está vacío. Revisa los pasos de carga y unión.")
    else:
        logger.error("ERROR CRÍTICO: No se pudieron cargar los datos de usuarios de S3 Gold. El script no puede continuar con la preparación del dataset de ML.")

    logger.info("--- Fin del script de carga de datos S3 Gold para ML (Local) ---")

# Asumiendo que df_final_ml_dataset ya está cargado y preparado como en tu output anterior

# --- 1. CREACIÓN DE LA VARIABLE OBJETIVO (TARGET) ---
# Esta es una lógica de ejemplo MUY SIMPLIFICADA. ¡DEBES ADAPTARLA!
# Vamos a asignar 'alto_riesgo' (1) y 'bajo_riesgo' (0)

import numpy as np # Lo necesitaremos para np.select

# Definir condiciones para alto riesgo (ejemplos)
condiciones_alto_riesgo = [
    (df_final_ml_dataset['ingreso_mensual_estimado'] < 1500000) & (df_final_ml_dataset['antiguedad_cliente'] < 2),
    (df_final_ml_dataset['estado_laboral'].isin(['DESEMPLEADO', 'ESTUDIANTE'])) & (df_final_ml_dataset['num_total_prestamos'] > 0),
    (df_final_ml_dataset['ingreso_mensual_estimado'] < 2000000) & (df_final_ml_dataset['num_total_prestamos'] > 1),
    (df_final_ml_dataset['edad'] < 22) & (df_final_ml_dataset['ingreso_mensual_estimado'] == 0)
]
# Definir opciones (1 para alto riesgo)
opciones_riesgo = [1] * len(condiciones_alto_riesgo)

# Crear la columna 'riesgo'. Por defecto, bajo riesgo (0), luego aplicar condiciones de alto riesgo.
df_final_ml_dataset['riesgo'] = np.select(condiciones_alto_riesgo, opciones_riesgo, default=0)

# Verificar la distribución de la nueva variable objetivo
print("\nDistribución de la variable objetivo 'riesgo':")
print(df_final_ml_dataset['riesgo'].value_counts(normalize=True))
print(df_final_ml_dataset[['ingreso_mensual_estimado', 'antiguedad_cliente', 'estado_laboral', 'num_total_prestamos', 'riesgo']].head(10))

# ¡IMPORTANTE! Revisa la distribución. Si está muy desbalanceada (ej. 95% de una clase),
# podrías necesitar técnicas de manejo de clases desbalanceadas (SMOTE, class_weight en modelos, etc.)
# o ajustar tu lógica de creación de 'riesgo' para tener una mezcla más balanceada para la demo.

# A partir de aquí, con `df_final_ml_dataset`, continuarías con:
# 1. Preprocesamiento final específico para tu modelo (escalado, codificación que no hiciste en el ETL).
# 2. División en conjuntos de entrenamiento y prueba.
# 3. Entrenamiento de tu modelo de ML.
# 4. Evaluación del modelo.
# 5. Guardado del modelo entrenado con joblib.
# 6. Subida del modelo a S3 ML Artifacts.

# --- Continuación del Script de Modelado (asume que 'riesgo' ya existe) ---

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier # Puedes probar otros como LogisticRegression, XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib # Para guardar el modelo y el preprocesador

# --- 2. SELECCIÓN DE CARACTERÍSTICAS (FEATURES) Y TARGET ---
# Columnas a excluir (identificadores, fechas originales si ya usaste la info)
columnas_a_excluir = ['user_id', 'cedula', 'nombre', 'apellido', 
                       'fecha_de_nacimiento', 'fecha_registro_banco'] 

# Asegúrate de que 'riesgo' sea tu columna objetivo
if 'riesgo' not in df_final_ml_dataset.columns:
    raise ValueError("La columna 'riesgo' (target) no existe. Por favor, créala primero.")

TARGET_COLUMN = 'riesgo'
X = df_final_ml_dataset.drop(columns=[TARGET_COLUMN] + columnas_a_excluir, errors='ignore')
y = df_final_ml_dataset[TARGET_COLUMN]

print(f"\nDimensiones de X (features): {X.shape}")
print(f"Dimensiones de y (target): {y.shape}")
print("\nPrimeras filas de X (features):")
print(X.head())


# --- 3. PREPROCESAMIENTO (CODIFICACIÓN Y ESCALADO) ---
# Identificar columnas categóricas y numéricas
# (excluyendo las que ya quitamos y la variable objetivo)
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCaracterísticas Categóricas Identificadas: {categorical_features}")
print(f"Características Numéricas Identificadas: {numerical_features}")

# Crear transformadores para el preprocesamiento
# Para las numéricas, escalaremos. Para las categóricas, haremos One-Hot Encoding.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features) # sparse_output=False para que devuelva array denso
    ], 
    remainder='passthrough' # Dejar pasar otras columnas si las hubiera (aunque no debería haber si seleccionamos bien)
)

# --- 4. DIVISIÓN DE DATOS (ENTRENAMIENTO Y PRUEBA) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y) # stratify es bueno si las clases están desbalanceadas

print(f"\nForma de X_train: {X_train.shape}, X_test: {X_test.shape}")

# --- 5. SELECCIÓN Y ENTRENAMIENTO DEL MODELO ---
# Vamos a usar RandomForest como ejemplo.
# Crearemos un pipeline que incluya el preprocesamiento y el clasificador.

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample'))])
                        # class_weight='balanced_subsample' puede ayudar con clases desbalanceadas

print("\nEntrenando el modelo RandomForestClassifier...")
model.fit(X_train, y_train)
print("Modelo entrenado.")

# --- 6. EVALUACIÓN DEL MODELO ---
print("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")
y_pred_test = model.predict(X_test)
y_pred_proba_test = model.predict_proba(X_test)[:, 1] # Probabilidad de la clase positiva (riesgo=1)

print(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_test):.4f}") # Necesita probabilidades

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred_test))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred_test))

# Opcional: Evaluar en el conjunto de entrenamiento para detectar overfitting severo
# y_pred_train = model.predict(X_train)
# print("\n--- Evaluación del Modelo en el Conjunto de Entrenamiento ---")
# print(f"Accuracy (Train): {accuracy_score(y_train, y_pred_train):.4f}")
# print(classification_report(y_train, y_pred_train))

# --- 7. GUARDAR EL MODELO Y EL PREPROCESADOR ---
# El pipeline 'model' ya contiene el preprocesador ajustado y el clasificador entrenado.
model_filename_local = "perabank_risk_pipeline_v1.joblib"
preprocessor_filename_local = "perabank_preprocessor_v1.joblib" # Alternativamente, guardar solo el pipeline completo

try:
    joblib.dump(model, model_filename_local)
    print(f"\nPipeline de Modelo completo guardado localmente como: {model_filename_local}")

    # Si quisieras guardar solo el preprocesador (aunque ya está en el pipeline):
    # joblib.dump(model.named_steps['preprocessor'], preprocessor_filename_local)
    # print(f"Preprocesador guardado localmente como: {preprocessor_filename_local}")

    # --- 8. (TU SIGUIENTE PASO) SUBIR A S3 ---
    # Aquí iría tu código con boto3 para subir 'perabank_risk_pipeline_v1.joblib'
    # al bucket S3 'perabank-ml-artifacts-bank/models/'

    # s3_client = boto3.client('s3') # Asumiendo que ya lo tienes
    # ml_artifacts_bucket = 'perabank-ml-artifacts-bank'
    # s3_model_key = f'models/{model_filename_local}'
    # s3_client.upload_file(model_filename_local, ml_artifacts_bucket, s3_model_key)
    # print(f"Pipeline de Modelo subido a S3: s3://{ml_artifacts_bucket}/{s3_model_key}")

except Exception as e:
    print(f"Error al guardar el modelo: {e}")