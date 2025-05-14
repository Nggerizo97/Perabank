# test_transactions_locally.py
import json
from ProcessTransaccionesBronzeToSilver_Lambda import lambda_handler

if __name__ == "__main__":
    # Simular evento para pruebas
    mock_event = {
        "input_path": "s3://perabank-bronze-data-bank/transacciones/transacciones.csv"
    }

    print("--- Iniciando prueba local de Lambda para transacciones ---")
    try:
        result = lambda_handler(mock_event, None)
        print("--- Resultado de la Lambda ---")
        print(json.dumps(result, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"Error durante la ejecución local: {e}")
        import traceback
        traceback.print_exc()