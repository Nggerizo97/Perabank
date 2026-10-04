"""Definición de contratos para la capa Bronze."""
from schemas.contracts import ColumnContract, TableContract

BRONZE_PAYSIM_CONTRACT = TableContract(
    table_name="paysim",
    layer="bronze",
    primary_keys=["step", "nameOrig", "nameDest"],
    columns=[
        ColumnContract("step", "int64", False, "Paso temporal (1 paso = 1 hora)"),
        ColumnContract("type", "string", False, "Tipo de transacción"),
        ColumnContract("amount", "float64", False, "Monto de la transacción"),
        ColumnContract("nameOrig", "string", False, "ID origen de la cuenta"),
        ColumnContract("oldbalanceOrg", "float64", True, "Saldo previo origen"),
        ColumnContract("newbalanceOrig", "float64", True, "Saldo posterior origen"),
        ColumnContract("nameDest", "string", False, "ID destino de la cuenta"),
        ColumnContract("oldbalanceDest", "float64", True, "Saldo previo destino"),
        ColumnContract("newbalanceDest", "float64", True, "Saldo posterior destino"),
        ColumnContract("isFraud", "int64", False, "Bandera de fraude real"),
        ColumnContract("isFlaggedFraud", "int64", False, "Bandera de fraude del sistema"),
        ColumnContract("source_system", "string", False, "Sistema origen (paysim)"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ]
)

BRONZE_BANK_TRANSACTIONS_CONTRACT = TableContract(
    table_name="bank_transactions",
    layer="bronze",
    primary_keys=["TransactionID"],
    columns=[
        ColumnContract("TransactionID", "string", False, "ID único de transacción"),
        ColumnContract("CustomerID", "string", False, "ID de cliente en banco indio"),
        ColumnContract("CustomerDOB", "string", True, "Fecha de nacimiento cliente"),
        ColumnContract("CustGender", "string", True, "Género cliente"),
        ColumnContract("CustLocation", "string", True, "Ubicación del cliente"),
        ColumnContract("CustAccountBalance", "float64", True, "Saldo en cuenta"),
        ColumnContract("TransactionDate", "string", False, "Fecha de la transacción"),
        ColumnContract("TransactionTime", "int64", True, "Hora de la transacción (HHMMSS)"),
        ColumnContract("TransactionAmount (INR)", "float64", False, "Monto de transacción en INR"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ]
)

BRONZE_BANK_MARKETING_CONTRACT = TableContract(
    table_name="bank_marketing",
    layer="bronze",
    primary_keys=[],
    columns=[
        ColumnContract("age", "int64", False, "Edad del cliente"),
        ColumnContract("job", "string", True, "Ocupación"),
        ColumnContract("marital", "string", True, "Estado civil"),
        ColumnContract("education", "string", True, "Nivel educativo"),
        ColumnContract("default", "string", True, "Crédito en mora"),
        ColumnContract("balance", "float64", False, "Saldo anual promedio (EUR)"),
        ColumnContract("housing", "string", True, "Tiene préstamo de vivienda"),
        ColumnContract("loan", "string", True, "Tiene préstamo personal"),
        ColumnContract("contact", "string", True, "Medio de contacto"),
        ColumnContract("day", "int64", False, "Último día de contacto"),
        ColumnContract("month", "string", False, "Último mes de contacto"),
        ColumnContract("duration", "int64", False, "Duración del contacto en seg"),
        ColumnContract("campaign", "int64", False, "Número de contactos en campaña"),
        ColumnContract("pdays", "int64", False, "Días desde última campaña"),
        ColumnContract("previous", "int64", False, "Contactos previos"),
        ColumnContract("poutcome", "string", True, "Resultado campaña previa"),
        ColumnContract("deposit", "string", True, "Suscripción a depósito a término"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ]
)

BRONZE_CREDITCARD_CONTRACT = TableContract(
    table_name="creditcard",
    layer="bronze",
    primary_keys=["Time"],
    columns=[
        ColumnContract("Time", "float64", False, "Segundos transcurridos desde 1ra transacción"),
        ColumnContract("Amount", "float64", False, "Monto de transacción (USD)"),
        ColumnContract("Class", "int64", False, "1 si es fraude, 0 de lo contrario"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ]
)

# Solo las columnas que consumen silver y gold: el archivo trae 142, y exigirlas
# todas haría fallar la ingesta por columnas que nadie usa (p.ej. hardship_*).
BRONZE_LENDINGCLUB_CONTRACT = TableContract(
    table_name="lendingclub",
    layer="bronze",
    primary_keys=["id"],
    columns=[
        ColumnContract(name, "string", True, "Columna cruda de LendingClub (ver datalake/LCDataDictionary.xlsx)")
        for name in [
            "id", "loan_amnt", "funded_amnt", "term", "int_rate", "installment", "grade", "sub_grade",
            "emp_length", "home_ownership", "annual_inc", "verification_status", "issue_d", "loan_status",
            "purpose", "addr_state", "zip_code", "dti", "delinq_2yrs", "earliest_cr_line",
            "fico_range_low", "fico_range_high", "inq_last_6mths", "mths_since_last_delinq",
            "mths_since_last_record", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
            "initial_list_status", "application_type", "mort_acc", "pub_rec_bankruptcies",
            "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim",
            "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee", "last_pymnt_d",
        ]
    ] + [
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ],
)
