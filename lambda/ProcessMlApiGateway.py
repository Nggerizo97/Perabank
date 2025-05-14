import json
import os
import boto3
import joblib # Para cargar modelos scikit-learn
import pandas as pd
import numpy as np # A menudo necesario para dar forma a los datos para el modelo

# --- Configuración Global (se ejecuta una vez por instancia de Lambda, en cold start) ---
MODEL_BUCKET = os.environ.get('AWS_S3_BUCKET_ML') # Nombre de tu bucket S3 donde está el modelo
MODEL_KEY = os.environ.get('MODEL_KEY', 'models/perabank_risk_pipeline_v1.joblib')

LOCAL_MODEL_PATH = f'/tmp/{MODEL_KEY.split("/")[-1]}' # Descargar modelo a /tmp (único directorio escribible en Lambda)

s3_client = boto3.client('s3')

# Descargar y cargar el modelo solo si no está ya en /tmp (optimización para warm starts)
if not os.path.exists(LOCAL_MODEL_PATH):
    print(f"Modelo no encontrado localmente en {LOCAL_MODEL_PATH}. Descargando de s3://{MODEL_BUCKET}/{MODEL_KEY}...")
    try:
        s3_client.download_file(MODEL_BUCKET, MODEL_KEY, LOCAL_MODEL_PATH)
        print(f"Modelo descargado exitosamente a {LOCAL_MODEL_PATH}.")
    except Exception as e:
        print(f"Error descargando el modelo de S3: {e}")
        # Podrías querer lanzar un error aquí para que el cold start falle si el modelo no se puede descargar
        # raise e 
else:
    print(f"Modelo ya existe localmente en {LOCAL_MODEL_PATH}. No se descargará de nuevo.")

try:
    print(f"Cargando modelo desde {LOCAL_MODEL_PATH}...")
    model_pipeline = joblib.load(LOCAL_MODEL_PATH) # Carga el pipeline completo (preprocesador + clasificador)
    print("Pipeline de modelo cargado exitosamente en memoria.")
except Exception as e:
    print(f"Error cargando el modelo desde {LOCAL_MODEL_PATH}: {e}")
    model_pipeline = None # Marcar como no cargado
    # raise e

def lambda_handler(event, context):
    print(f"Evento recibido: {event}")

    if model_pipeline is None:
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            'body': json.dumps({'error': 'Modelo no pudo ser cargado.'})
        }

    try:
        # API Gateway con integración proxy Lambda pasa el cuerpo como un string JSON
        if isinstance(event.get('body'), str):
            body = json.loads(event['body'])
        else: # Si ya es un dict (ej. prueba directa en Lambda)
            body = event.get('body', event) # 'body' para API GW, o el evento mismo para test directo

        print(f"Cuerpo del request procesado: {body}")

        # --- PREPROCESAMIENTO DE LA ENTRADA ---
        # La entrada 'body' debe contener las características en el formato que tu pipeline espera.
        # Si tu pipeline (guardado como perabank_risk_pipeline_v1.joblib) incluye el ColumnTransformer
        # con OneHotEncoder y StandardScaler, entonces necesitas pasar un DataFrame de Pandas
        # con los nombres de columna originales (antes del preprocesamiento).

        # Ejemplo: Si el API recibe un diccionario de features:
        # { "ingreso_mensual_estimado": 5000000, "edad": 30, "profesion": "INGENIERO", ... }
        # Debes convertir esto a un DataFrame de Pandas con una fila.

        # Asegúrate de que 'body' sea un diccionario con las features
        if not isinstance(body, dict):
            raise ValueError("El cuerpo del request debe ser un objeto JSON (diccionario).")

        # Crear un DataFrame de Pandas con una sola fila a partir del 'body'
        # Las claves en 'body' deben coincidir con los nombres de las columnas
        # que tu preprocesador espera ANTES de la transformación.
        input_df = pd.DataFrame([body])

        # Asegúrate de que las columnas estén en el orden correcto si tu pipeline es sensible a ello,
        # aunque el ColumnTransformer de scikit-learn generalmente maneja esto por nombre de columna.
        # Si guardaste solo el modelo y no el pipeline de preprocesamiento, aquí tendrías que aplicar
        # el preprocesamiento (escalado, one-hot encoding) exactamente como en el entrenamiento.
        # ¡Pero como guardaste el pipeline completo, es más fácil!

        print(f"DataFrame de entrada para el pipeline: \n{input_df.to_string()}")

        # --- PREDICCIÓN ---
        prediction = model_pipeline.predict(input_df)
        # predict_proba() da las probabilidades de cada clase
        probabilities = model_pipeline.predict_proba(input_df)

        # La predicción será un array (ej. [0] o [1]), tomamos el primer elemento.
        # Las probabilidades serán un array de arrays (ej. [[0.8, 0.2]]), tomamos el primer elemento.
        riesgo_predicho = int(prediction[0]) # Convertir a int nativo de Python
        prob_clase_0 = float(probabilities[0][0])
        prob_clase_1 = float(probabilities[0][1])

        # Opcional: Mapear la predicción numérica a una etiqueta legible
        etiqueta_riesgo = "Bajo Riesgo" if riesgo_predicho == 0 else "Alto Riesgo"

        print(f"Predicción (numérica): {riesgo_predicho}, Etiqueta: {etiqueta_riesgo}")
        print(f"Probabilidades: Clase 0 (Bajo Riesgo) = {prob_clase_0:.4f}, Clase 1 (Alto Riesgo) = {prob_clase_1:.4f}")

        # --- RESPUESTA ---
        response_body = {
            'prediccion_numerica': riesgo_predicho,
            'prediccion_etiqueta': etiqueta_riesgo,
            'probabilidad_bajo_riesgo': prob_clase_0,
            'probabilidad_alto_riesgo': prob_clase_1
        }

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*' # Permite CORS para todas las fuentes (ajusta en producción)
            },
            'body': json.dumps(response_body)
        }

    except ValueError as ve:
        print(f"Error de Valor (posiblemente en la entrada): {str(ve)}")
        return {
            'statusCode': 400, # Bad Request
            'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            'body': json.dumps({'error': f"Error en los datos de entrada: {str(ve)}"})
        }
    except Exception as e:
        print(f"Error durante la predicción: {str(e)}")
        import traceback
        traceback.print_exc() # Imprime el traceback completo a CloudWatch Logs
        return {
            'statusCode': 500,
            'headers': { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' },
            'body': json.dumps({'error': f"Error interno del servidor al procesar la predicción: {str(e)}"})
        }