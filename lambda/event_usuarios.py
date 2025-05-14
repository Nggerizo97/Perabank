# Crea un archivo llamado, por ejemplo, test_locally.py en la misma carpeta que tu lambda_function.py

import json
from ProcessUsersBronzeToSilver_Lambda import lambda_handler # Asume que tu script se llama lambda_function.py

if __name__ == "__main__":
    # Opción 1: Simular el evento que pasaría Step Functions o una prueba manual
    mock_event = {
        "input_path": "s3://perabank-bronze-data-bank/usuarios/usuarios.csv"
        # Asegúrate de que este archivo exista en tu S3 para la prueba
    }

    # Opción 2: Simular un evento de trigger S3 (si quisieras probar esa lógica)
    # mock_event = {
    #     "Records": [
    #         {
    #             "s3": {
    #                 "bucket": {
    #                     "name": "perabank-bronze-data-bank"
    #                 },
    #                 "object": {
    #                     "key": "usuarios/usuarios.csv" # Asegúrate que este archivo exista
    #                 }
    #             }
    #         }
    #     ]
    # }

    mock_context = None # El contexto no suele ser crítico para pruebas locales de este tipo

    print("--- Iniciando prueba local de Lambda ---")
    try:
        result = lambda_handler(mock_event, mock_context)
        print("--- Resultado de la Lambda ---")
        print(json.dumps(result, indent=4, ensure_ascii=False)) # ensure_ascii para tildes
    except Exception as e:
        print(f"Error durante la ejecución local: {e}")
        import traceback
        traceback.print_exc()