import json
import random
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import uuid
import os
import boto3
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


# --- Configuración ---
# Usar un localizador específico para datos más realistas (ej: Colombia)
# fake = Faker('es_CO') # Puedes instalarlo con: pip install Faker[es_CO]
# Si no tienes el localizador, Faker usará el inglés por defecto
try:
    fake = Faker('es_CO')
    print("Usando localizador es_CO para Faker.")
except:
    fake = Faker()
    print("Localizador es_CO no encontrado, usando Faker por defecto (en).")

NUM_USERS = 150 # Aumentamos un poco
NUM_TRANSACTIONS_PER_USER_RANGE = (5, 50) # Rango de transacciones por usuario
NUM_ACCOUNTS_PER_USER_RANGE = (1, 3) # Rango de cuentas por usuario
NUM_LOANS_PERCENTAGE = 0.25 # Porcentaje de usuarios con préstamos
NUM_RECIPIENTS = 50 # Destinatarios registrados (para transferencias P2P simuladas)

OUTPUT_DIR = "output_enhanced"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Listas de Opciones (Más realistas) ---
ESTADOS_LABORALES = ['Empleado', 'Independiente', 'Desempleado', 'Estudiante', 'Pensionado']
NIVELES_EDUCATIVOS = ['Primaria', 'Secundaria', 'Técnico', 'Universitario', 'Postgrado', 'Ninguno']
ESTADOS_CIVILES = ['Soltero(a)', 'Casado(a)', 'Unión Libre', 'Divorciado(a)', 'Viudo(a)']
TIPOS_VIVIENDA = ['Propia', 'Alquilada', 'Familiar']
TIPOS_CUENTA = ['Ahorros', 'Corriente', 'Nómina']
ESTADOS_CUENTA = ['Activa', 'Inactiva', 'Bloqueada']
TIPOS_TRANSACCION = ['Compra', 'Transferencia Enviada', 'Transferencia Recibida', 'Pago Servicio', 'Retiro ATM', 'Depósito', 'Pago Préstamo', 'Salario']
CATEGORIAS_TRANSACCION = {
    'Compra': ['Alimentación', 'Transporte', 'Entretenimiento', 'Ropa y Accesorios', 'Hogar', 'Salud', 'Tecnología', 'Viajes'],
    'Pago Servicio': ['Agua', 'Luz', 'Gas', 'Internet', 'Telefonía', 'Televisión', 'Administración'],
    'Otros': ['Ajuste', 'Comisión Bancaria'] # Para tipos como Retiro, Depósito, etc.
}
CANALES = ['Online', 'POS', 'ATM', 'Sucursal', 'App Móvil']
ESTADOS_PRESTAMO = ['Activo', 'Pagado', 'En Mora']
BANCOS = ["Davivienda", "Bancolombia", "BBVA", "Banco de Bogotá", "Itaú", "Scotiabank Colpatria", "Banco Agrario", "Nequi", "Daviplata"] # Más bancos

# --- Funciones de Generación Mejoradas ---

def generate_users(num_records=NUM_USERS):
    users = []
    cedulas = set() # Para asegurar unicidad
    while len(users) < num_records:
        # cedula = fake.ssn() # fake.ssn() puede no generar formato colombiano
        # Generamos un número tipo cédula (simplificado)
        cedula = str(random.randint(10000000, 1100000000))
        if cedula in cedulas:
            continue
        cedulas.add(cedula)

        fecha_nacimiento = fake.date_of_birth(minimum_age=18, maximum_age=75)
        fecha_registro_banco = fake.date_time_between(start_date=fecha_nacimiento + timedelta(days=18*365), end_date='now')

        user = {
            # Cambiamos a user_id como PK y cedula como atributo único
            "user_id": str(uuid.uuid4()),
            "cedula": cedula,
            "nombre": fake.first_name(),
            "apellido": fake.last_name(),
            "ciudad_residencia": fake.city(), # Cambiado de nacimiento a residencia
            "fecha_de_nacimiento": fecha_nacimiento.isoformat(),
            "profesion": fake.job(),
            "ingreso_mensual_estimado": round(random.uniform(800000, 15000000), 2) if random.random() > 0.1 else 0, # COP, 10% sin ingreso reportado
            "estado_laboral": random.choice(ESTADOS_LABORALES),
            "nivel_educativo": random.choice(NIVELES_EDUCATIVOS),
            "estado_civil": random.choice(ESTADOS_CIVILES),
            "numero_dependientes": random.randint(0, 5),
            "tipo_vivienda": random.choice(TIPOS_VIVIENDA),
            "fecha_registro_banco": fecha_registro_banco.isoformat(),
        }
        users.append(user)
    return users

# Generamos Cuentas *después* de Usuarios
def generate_accounts(users_df):
    accounts = []
    for _, user in users_df.iterrows():
        num_accounts = random.randint(NUM_ACCOUNTS_PER_USER_RANGE[0], NUM_ACCOUNTS_PER_USER_RANGE[1])
        for _ in range(num_accounts):
            fecha_apertura = fake.date_time_between(start_date=pd.to_datetime(user['fecha_registro_banco']), end_date='now')
            account = {
                "account_id": str(uuid.uuid4()),
                "user_id": user['user_id'], # FK a Usuarios
                "numero_cuenta": fake.unique.bban(), # Genera un número de cuenta bancaria
                "tipo_cuenta": random.choice(TIPOS_CUENTA),
                "fecha_apertura": fecha_apertura.isoformat(),
                "saldo_actual": round(random.uniform(-50000, 25000000), 2), # Puede haber sobregiros pequeños
                "moneda": "COP",
                "estado_cuenta": random.choice(ESTADOS_CUENTA) if random.random() > 0.05 else "Activa", # Mayoría activas
            }
            accounts.append(account)
    return accounts

# Generamos Destinatarios Registrados (simulados, para transferencias P2P)
def generate_recipients(num_records=NUM_RECIPIENTS):
    recipients = []
    for _ in range(num_records):
        recipient = {
            "recipient_id": str(uuid.uuid4()), # PK
            "nombre_registrado": fake.name(), # Nombre con el que el usuario lo guardó
            "banco_destinatario": random.choice(BANCOS),
            "tipo_cuenta_dest": random.choice(TIPOS_CUENTA[:2]), # Ahorros o Corriente
            "numero_cuenta_dest": fake.unique.bban(), # Número ficticio
            # "pais": fake.country(), # Quizás menos relevante a menos que sean transacciones internacionales
        }
        recipients.append(recipient)
    return recipients

# Generamos Transacciones *después* de Usuarios y Cuentas
def generate_transactions(accounts_df, recipients_df):
    transactions = []
    user_accounts = accounts_df.groupby('user_id')['account_id'].apply(list).to_dict()
    recipient_ids = recipients_df['recipient_id'].tolist()

    for user_id, user_acc_list in user_accounts.items():
         num_transactions = random.randint(NUM_TRANSACTIONS_PER_USER_RANGE[0], NUM_TRANSACTIONS_PER_USER_RANGE[1])
         # Asegurar que las transacciones ocurran después de la apertura de la cuenta asociada
         account_open_dates = accounts_df[accounts_df['user_id'] == user_id].set_index('account_id')['fecha_apertura'].apply(pd.to_datetime).to_dict()

         for _ in range(num_transactions):
            account_id = random.choice(user_acc_list)
            min_tx_date = account_open_dates[account_id]

            tipo_transaccion = random.choice(TIPOS_TRANSACCION)
            monto = 0
            categoria = None
            destinatario_id_fk = None # FK a la tabla Recipients

            if tipo_transaccion == 'Compra':
                monto = round(random.uniform(5000, 500000), 2)
                categoria = random.choice(CATEGORIAS_TRANSACCION['Compra'])
            elif tipo_transaccion == 'Transferencia Enviada':
                monto = round(random.uniform(10000, 2000000), 2)
                categoria = 'Transferencia'
                if recipient_ids and random.random() > 0.3: # 70% de las transferencias van a destinatarios registrados
                    destinatario_id_fk = random.choice(recipient_ids)
            elif tipo_transaccion == 'Transferencia Recibida':
                 monto = round(random.uniform(10000, 3000000), 2)
                 categoria = 'Transferencia'
            elif tipo_transaccion == 'Pago Servicio':
                monto = round(random.uniform(15000, 300000), 2)
                categoria = random.choice(CATEGORIAS_TRANSACCION['Pago Servicio'])
            elif tipo_transaccion == 'Retiro ATM':
                monto = round(random.uniform(20000, 1000000), 2)
                categoria = 'Retiro'
            elif tipo_transaccion == 'Depósito':
                monto = round(random.uniform(50000, 5000000), 2)
                categoria = 'Depósito'
            elif tipo_transaccion == 'Pago Préstamo':
                 monto = round(random.uniform(50000, 1500000), 2)
                 categoria = 'Pago Obligación'
            elif tipo_transaccion == 'Salario':
                 monto = round(random.uniform(800000, 10000000), 2)
                 categoria = 'Ingreso Nómina'
            else: # Otros tipos
                monto = round(random.uniform(1000, 50000), 2)
                categoria = 'Otros'

            transaction = {
                "transaction_id": str(uuid.uuid4()), # PK
                "account_id": account_id, # FK a Cuentas
                "user_id": user_id, # FK a Usuarios (redundante si tenemos account_id, pero útil)
                "monto": monto,
                # Asegurar que fecha sea posterior a apertura de cuenta
                "fecha": fake.date_time_between(start_date=min_tx_date, end_date='now').isoformat(),
                "tipo_transaccion": tipo_transaccion,
                "categoria": categoria,
                "canal": random.choice(CANALES),
                "tipo_tarjeta_usada": random.choice(["Crédito", "Débito", "N/A"]), # N/A para transferencias, depósitos etc.
                "recipient_id_fk": destinatario_id_fk, # FK a la tabla Recipients (puede ser NULL)
                "descripcion": fake.sentence(nb_words=5) # Descripción corta opcional
            }
            transactions.append(transaction)
    return transactions

# Generamos Préstamos *después* de Usuarios
def generate_loans(users_df):
    loans = []
    eligible_users = users_df.sample(frac=NUM_LOANS_PERCENTAGE) # Tomamos un % de usuarios

    for _, user in eligible_users.iterrows():
        fecha_inicio_prestamo = fake.date_time_between(start_date=pd.to_datetime(user['fecha_registro_banco']) + timedelta(days=90), end_date='now')
        plazo = random.choice([12, 24, 36, 48, 60, 72])
        monto_prestamo = round(random.uniform(1000000, 50000000), 2)

        loan = {
            "loan_id": str(uuid.uuid4()),
            "user_id": user['user_id'], # FK a Usuarios
            "monto_otorgado": monto_prestamo,
            "tasa_interes_anual": round(random.uniform(15.0, 35.0), 2), # %
            "plazo_meses": plazo,
            "fecha_desembolso": fecha_inicio_prestamo.isoformat(),
            "cuota_mensual": round(monto_prestamo * ( (0.15/12) / (1-(1+(0.15/12))**(-plazo)) ), 2) , # Estimación simple (Tasa fija 15% anual para cálculo)
            "estado_prestamo": random.choice(ESTADOS_PRESTAMO),
            "proposito": random.choice(['Libre Inversión', 'Compra Vehículo', 'Educación', 'Vivienda', 'Deuda']),
        }
        loans.append(loan)
    return loans


# --- Función Principal para Generar y Guardar ---

def generate_and_save_all_data(upload_to_s3=True): # Añadimos un flag
    s3_client = None
    if upload_to_s3:
        s3_client = boto3.client(
                    's3',
                    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                    region_name=os.getenv('AWS_REGION')
                    )   
        bronze_bucket_name = 'perabank-bronze-data-bank' # Tu bucket

    print("Generando Usuarios...")
    users = generate_users()
    users_df = pd.DataFrame(users)
    # Guardar localmente (opcional si solo subes)
    local_usuarios_csv_path = os.path.join(OUTPUT_DIR, "usuarios.csv")
    users_df.to_csv(local_usuarios_csv_path, index=False)
    # users_df.to_json(os.path.join(OUTPUT_DIR, "usuarios.json"), orient='records', indent=4)
    print(f"Usuarios generados y guardados localmente: {len(users_df)}")
    if upload_to_s3:
        s3_key = "usuarios/usuarios.csv"
        s3_client.upload_file(local_usuarios_csv_path, bronze_bucket_name, s3_key)
        print(f"Usuarios subidos a S3: s3://{bronze_bucket_name}/{s3_key}")

    print("Generando Cuentas...")
    accounts = generate_accounts(users_df)
    accounts_df = pd.DataFrame(accounts)
    local_cuentas_csv_path = os.path.join(OUTPUT_DIR, "cuentas.csv")
    accounts_df.to_csv(local_cuentas_csv_path, index=False)
    print(f"Cuentas generadas y guardadas localmente: {len(accounts_df)}")
    if upload_to_s3:
        s3_key = "cuentas/cuentas.csv"
        s3_client.upload_file(local_cuentas_csv_path, bronze_bucket_name, s3_key)
        print(f"Cuentas subidas a S3: s3://{bronze_bucket_name}/{s3_key}")

    # Repite el patrón para Destinatarios, Transacciones y Préstamos
    # ... (código para destinatarios) ...
    print("Generando Destinatarios Registrados...")
    recipients = generate_recipients()
    recipients_df = pd.DataFrame(recipients)
    local_recipients_csv_path = os.path.join(OUTPUT_DIR, "destinatarios.csv")
    recipients_df.to_csv(local_recipients_csv_path, index=False)
    print(f"Destinatarios generados y guardados localmente: {len(recipients_df)}")
    if upload_to_s3:
        s3_key = "destinatarios/destinatarios.csv"
        s3_client.upload_file(local_recipients_csv_path, bronze_bucket_name, s3_key)
        print(f"Destinatarios subidos a S3: s3://{bronze_bucket_name}/{s3_key}")

    # ... (código para transacciones) ...
    print("Generando Transacciones...")
    transactions = generate_transactions(accounts_df, recipients_df)
    transactions_df = pd.DataFrame(transactions)
    local_transactions_csv_path = os.path.join(OUTPUT_DIR, "transacciones.csv")
    transactions_df.to_csv(local_transactions_csv_path, index=False)
    print(f"Transacciones generadas y guardadas localmente: {len(transactions_df)}")
    if upload_to_s3:
        s3_key = "transacciones/transacciones.csv"
        s3_client.upload_file(local_transactions_csv_path, bronze_bucket_name, s3_key)
        print(f"Transacciones subidas a S3: s3://{bronze_bucket_name}/{s3_key}")

    # ... (código para préstamos) ...
    print("Generando Préstamos...")
    loans = generate_loans(users_df)
    loans_df = pd.DataFrame(loans)
    local_loans_csv_path = os.path.join(OUTPUT_DIR, "prestamos.csv")
    loans_df.to_csv(local_loans_csv_path, index=False)
    print(f"Préstamos generados y guardados localmente: {len(loans_df)}")
    if upload_to_s3:
        s3_key = "prestamos/prestamos.csv"
        s3_client.upload_file(local_loans_csv_path, bronze_bucket_name, s3_key)
        print(f"Préstamos subidos a S3: s3://{bronze_bucket_name}/{s3_key}")

    print(f"\nDatos simulados generados y, si upload_to_s3=True, subidos a S3 Bronce.")

# --- Modifica la ejecución al final del script ---
if __name__ == "__main__":
    generate_and_save_all_data(upload_to_s3=True) # Cambia a False si solo quieres generar localmente