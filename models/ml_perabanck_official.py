import boto3
import pandas as pd
import io
import logging
from datetime import datetime 
import unidecode 
from dotenv import load_dotenv
import os
from urllib.parse import urlparse
import numpy as np # Asegúrate de tener este import para np.select
import json # Asegúrate de tener este import para json.dumps en el return de Lambda (si este código fuera parte de una Lambda)

# --- TU CÓDIGO DE CONFIGURACIÓN INICIAL (logger, .env, buckets, get_s3_client, etc.) ---
# ... (Pega aquí todo tu bloque de configuración y funciones de ayuda: 
#      logger, carga de .env, nombres de buckets, get_s3_client, 
#      generate_s3_output_path, load_parquet_from_s3) ...
# --- Por ejemplo: ---
logger = logging.getLogger()
logger.setLevel(logging.INFO) 
try:
    env_path_current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    env_path_parent_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    if os.path.exists(env_path_current_dir): env_path = env_path_current_dir
    elif os.path.exists(env_path_parent_dir): env_path = env_path_parent_dir
    else: env_path = None
    if env_path:
        logger.info(f"Cargando .env desde: {env_path}")
        load_dotenv(env_path)
    else:
        logger.info("Archivo .env no encontrado, asumiendo credenciales globales o rol IAM.")
except Exception as e:
    logger.info(f"No se pudo cargar .env: {e}")

GOLD_BUCKET = os.getenv('AWS_S3_BUCKET_GOLD', 'perabank-gold-data-bank')
ML_ARTIFACTS_BUCKET = os.getenv('AWS_S3_BUCKET_ML', 'perabank-ml-artifacts-bank') 
# Asegúrate que ML_ARTIFACTS_BUCKET esté definido en tu .env o aquí como default.

def get_s3_client():
    # ... (tu función get_s3_client)
    if 'AWS_LAMBDA_FUNCTION_NAME' in os.environ:
        return boto3.client('s3')
    else:
        return boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )

def load_parquet_from_s3(s3_client, bucket_name, file_key):
    # ... (tu función load_parquet_from_s3)
    s3_path = f"s3://{bucket_name}/{file_key}"
    logger.info(f"Intentando leer Parquet desde: {s3_path}")
    try:
        obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        df = pd.read_parquet(io.BytesIO(obj['Body'].read()), engine='pyarrow')
        logger.info(f"Datos de '{file_key}' cargados. Dimensiones: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"ERROR al leer el archivo Parquet '{s3_path}': {e}", exc_info=True)
        return pd.DataFrame()
# --- FIN DE TU CÓDIGO DE CONFIGURACIÓN INICIAL ---


# --- SCRIPT PRINCIPAL (Continuación después de la carga de datos y preparación de df_final_ml_dataset) ---
if __name__ == "__main__":
    logger.info("--- Iniciando script de carga de datos S3 Gold y entrenamiento de ML (Local) ---")
    
    s3 = get_s3_client()

    keys_gold = {
        "usuarios": "usuarios/usuarios_enriched.parquet",
        "cuentas": "cuentas/cuentas_enriched.parquet",
        "transacciones": "transacciones/transacciones_enriched.parquet",
        "prestamos": "prestamos/prestamos_enriched.parquet",
        "destinatarios": "destinatarios/destinatarios_enriched.parquet"
    }
    dataframes_gold = {}
    for entity_name, entity_key in keys_gold.items():
        df = load_parquet_from_s3(s3, GOLD_BUCKET, entity_key)
        if df.empty: logger.warning(f"DataFrame para '{entity_name}' está vacío.")
        dataframes_gold[entity_name] = df

    # --- TU LÓGICA DE UNIÓN Y AGREGACIÓN PARA CREAR df_final_ml_dataset ---
    # ... (Asegúrate que este bloque de código se ejecute y df_final_ml_dataset se popule correctamente) ...
    logger.info("\n--- Iniciando el proceso de unión y agregación de DataFrames (SIMULADO) ---")
    # Ejemplo simplificado: asume que df_usuarios_gold es la base y ya tiene todo lo necesario
    # DEBES REEMPLAZAR ESTO CON TU LÓGICA DE UNIÓN COMPLETA
    if not dataframes_gold.get("usuarios", pd.DataFrame()).empty:
        df_final_ml_dataset = dataframes_gold["usuarios"].copy() 
        # Simulación de algunas columnas necesarias para el ejemplo de target y preprocesador
        # ASEGÚRATE QUE ESTAS COLUMNAS EXISTAN REALMENTE DESPUÉS DE TU UNIÓN
        if 'ingreso_mensual_estimado' not in df_final_ml_dataset.columns: df_final_ml_dataset['ingreso_mensual_estimado'] = np.random.rand(len(df_final_ml_dataset)) * 10000000
        if 'antiguedad_cliente' not in df_final_ml_dataset.columns: df_final_ml_dataset['antiguedad_cliente'] = np.random.randint(1, 10, len(df_final_ml_dataset))
        if 'estado_laboral' not in df_final_ml_dataset.columns: df_final_ml_dataset['estado_laboral'] = np.random.choice(['EMPLEADO', 'DESEMPLEADO', 'ESTUDIANTE'], len(df_final_ml_dataset))
        if 'num_total_prestamos' not in df_final_ml_dataset.columns: df_final_ml_dataset['num_total_prestamos'] = np.random.randint(0, 3, len(df_final_ml_dataset))
        if 'edad' not in df_final_ml_dataset.columns: df_final_ml_dataset['edad'] = np.random.randint(18, 70, len(df_final_ml_dataset))
        # ... añade otras columnas simuladas si tu lógica de abajo las necesita y no vienen de la unión ...
        logger.info(f"Dataset inicial para ML (posiblemente simulado/incompleto): {df_final_ml_dataset.shape}")
    else:
        logger.error("Datos de usuarios vacíos, no se puede continuar con el modelado.")
        exit() # Salir si no hay datos base

    # --- 1. CREACIÓN DE LA VARIABLE OBJETIVO (TARGET) ---
    # Mantén tu lógica de creación de 'riesgo' aquí
    if not df_final_ml_dataset.empty:
        condiciones_alto_riesgo = [
            (df_final_ml_dataset['ingreso_mensual_estimado'] < 1500000) & (df_final_ml_dataset['antiguedad_cliente'] < 2),
            (df_final_ml_dataset['estado_laboral'].isin(['DESEMPLEADO', 'ESTUDIANTE'])) & (df_final_ml_dataset['num_total_prestamos'] > 0),
            (df_final_ml_dataset['ingreso_mensual_estimado'] < 2000000) & (df_final_ml_dataset['num_total_prestamos'] > 1),
            (df_final_ml_dataset['edad'] < 22) & (df_final_ml_dataset['ingreso_mensual_estimado'] == 0)
        ]
        opciones_riesgo = [1] * len(condiciones_alto_riesgo)
        df_final_ml_dataset['riesgo'] = np.select(condiciones_alto_riesgo, opciones_riesgo, default=0)
        print("\nDistribución de la variable objetivo 'riesgo':")
        print(df_final_ml_dataset['riesgo'].value_counts(normalize=True))
    else:
        logger.error("df_final_ml_dataset está vacío antes de crear la variable objetivo.")
        exit()
    
    # --- Importaciones para el modelo ---
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.pipeline import Pipeline
    import joblib

    # --- 2. SELECCIÓN DE CARACTERÍSTICAS (FEATURES) Y TARGET ---
    columnas_a_excluir = ['user_id', 'cedula', 'nombre', 'apellido', 
                           'fecha_de_nacimiento', 'fecha_registro_banco'] 
    if 'riesgo' not in df_final_ml_dataset.columns:
        raise ValueError("La columna 'riesgo' (target) no existe. Por favor, créala primero.")
    TARGET_COLUMN = 'riesgo'
    
    # Filtrar columnas existentes antes de dropear
    existing_columns_to_exclude = [col for col in columnas_a_excluir if col in df_final_ml_dataset.columns]
    X = df_final_ml_dataset.drop(columns=[TARGET_COLUMN] + existing_columns_to_exclude, errors='ignore')
    y = df_final_ml_dataset[TARGET_COLUMN]

    # --- 3. PREPROCESAMIENTO (CODIFICACIÓN Y ESCALADO) ---
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Asegurarse que no haya features vacías
    if not categorical_features and not numerical_features:
        logger.error("No se identificaron features categóricas ni numéricas para el preprocesador.")
        exit()
    
    transformers_list = []
    if numerical_features:
        transformers_list.append(('num', StandardScaler(), numerical_features))
    if categorical_features:
        transformers_list.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features))

    if not transformers_list:
        logger.error("La lista de transformadores está vacía. Revisa la identificación de features.")
        # Si no hay features, no tiene sentido seguir, pero en tu caso sí las hay
        # Podrías decidir usar 'passthrough' para todas si este fuera el caso,
        # o simplemente no usar ColumnTransformer.
        # Para este ejemplo, asumimos que habrá features.
        preprocessor = 'passthrough' 
    else:
        preprocessor = ColumnTransformer(
            transformers=transformers_list,
            remainder='drop' # O 'passthrough' si quieres mantener otras columnas no especificadas
        )

    # --- 4. DIVISIÓN DE DATOS ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # --- 5. ENTRENAMIENTO DEL MODELO ---
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', RandomForestClassifier(n_estimators=100, 
                                                                          random_state=42, 
                                                                          class_weight='balanced_subsample'))])
    logger.info("\nEntrenando el modelo RandomForestClassifier...")
    model_pipeline.fit(X_train, y_train)
    logger.info("Modelo entrenado.")

    # --- 6. EVALUACIÓN DEL MODELO ---
    logger.info("\n--- Evaluación del Modelo en el Conjunto de Prueba ---")
    y_pred_test = model_pipeline.predict(X_test)
    
    # Para ROC AUC, necesitamos probabilidades de la clase positiva (1)
    if hasattr(model_pipeline, "predict_proba"):
        y_pred_proba_test = model_pipeline.predict_proba(X_test)[:, 1]
        logger.info(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba_test):.4f}")
    else:
        logger.warning("El modelo no tiene predict_proba, no se puede calcular ROC AUC.")

    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred_test):.4f}")
    logger.info("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred_test, zero_division=0)) # Añadir zero_division
    logger.info("\nMatriz de Confusión:")
    print(confusion_matrix(y_test, y_pred_test))

    # --- 7. GUARDAR EL MODELO LOCALMENTE ---
    model_filename_local = "perabank_risk_pipeline_v1.joblib"
    try:
        joblib.dump(model_pipeline, model_filename_local)
        logger.info(f"Pipeline de Modelo completo guardado localmente como: {model_filename_local}")

        # --- 8. SUBIR EL MODELO A S3 ---
        if ML_ARTIFACTS_BUCKET: # Solo intentar subir si el bucket está definido
            s3_model_key = f'models/{model_filename_local}' # Guardarlo en una "carpeta" models
            logger.info(f"Intentando subir el modelo a: s3://{ML_ARTIFACTS_BUCKET}/{s3_model_key}")
            try:
                s3.upload_file(model_filename_local, ML_ARTIFACTS_BUCKET, s3_model_key)
                logger.info(f"Pipeline de Modelo subido a S3: s3://{ML_ARTIFACTS_BUCKET}/{s3_model_key}")
            except Exception as e_s3_upload:
                logger.error(f"Error subiendo el modelo a S3: {e_s3_upload}", exc_info=True)
        else:
            logger.warning("La variable de entorno AWS_S3_BUCKET_ML (o ML_ARTIFACTS_BUCKET) no está definida. El modelo no se subirá a S3.")

    except Exception as e_save:
        logger.error(f"Error al guardar el modelo localmente: {e_save}", exc_info=True)
    
    logger.info("--- Fin del script de entrenamiento y guardado de modelo ---")