"""Definición de contratos para la capa Silver."""
from schemas.contracts import ColumnContract, TableContract

SILVER_PAYSIM_CONTRACT = TableContract(
    table_name="paysim",
    layer="silver",
    primary_keys=["step", "nameOrig", "nameDest"],
    columns=[
        ColumnContract("step", "int64", False),
        ColumnContract("type", "string", False),
        ColumnContract("amount", "float64", False),
        ColumnContract("nameOrig", "string", False),
        ColumnContract("oldbalanceOrg", "float64", True),
        ColumnContract("newbalanceOrig", "float64", True),
        ColumnContract("nameDest", "string", False),
        ColumnContract("oldbalanceDest", "float64", True),
        ColumnContract("newbalanceDest", "float64", True),
        ColumnContract("isFraud", "bool", False),
        ColumnContract("isFlaggedFraud", "bool", False),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_BANK_TRANSACTIONS_CONTRACT = TableContract(
    table_name="bank_transactions",
    layer="silver",
    primary_keys=["TransactionID"],
    columns=[
        ColumnContract("TransactionID", "string", False),
        ColumnContract("CustomerID", "string", False),
        ColumnContract("CustomerDOB", "string", True),
        ColumnContract("CustGender", "string", True),
        ColumnContract("CustLocation", "string", True),
        ColumnContract("CustAccountBalance", "float64", True),
        ColumnContract("TransactionDate", "datetime64[ns]", False),
        ColumnContract("TransactionTime", "int64", True),
        ColumnContract("TransactionAmount (INR)", "float64", False),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_BANK_MARKETING_CONTRACT = TableContract(
    table_name="bank_marketing",
    layer="silver",
    primary_keys=[],
    columns=[
        ColumnContract("age", "int64", False),
        ColumnContract("job", "string", True),
        ColumnContract("marital", "string", True),
        ColumnContract("education", "string", True),
        ColumnContract("default", "bool", False),
        ColumnContract("balance", "float64", False),
        ColumnContract("housing", "bool", False),
        ColumnContract("loan", "bool", False),
        ColumnContract("contact", "string", True),
        ColumnContract("day", "int64", False),
        ColumnContract("month", "string", False),
        ColumnContract("duration", "int64", False),
        ColumnContract("campaign", "int64", False),
        ColumnContract("pdays", "int64", False),
        ColumnContract("previous", "int64", False),
        ColumnContract("poutcome", "string", True),
        ColumnContract("deposit", "bool", False),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_CREDITCARD_CONTRACT = TableContract(
    table_name="creditcard",
    layer="silver",
    primary_keys=["Time"],
    columns=[
        ColumnContract("Time", "float64", False),
        ColumnContract("Amount", "float64", False),
        ColumnContract("Class", "bool", False),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_MARKET_CONTRACT = TableContract(
    table_name="enrichment_market",
    layer="silver",
    primary_keys=["fecha"],
    columns=[
        ColumnContract("fecha", "string", False),
        ColumnContract("tasa_usd_inr", "float64", False),
        ColumnContract("tasa_usd_eur", "float64", False),
        ColumnContract("tasa_usd_cop", "float64", False),
        ColumnContract("tasa_ibr_overnight", "float64", False),
        ColumnContract("tasa_treasury_10y", "float64", True),
        ColumnContract("tasa_tbill_3m", "float64", True),
        ColumnContract("volatilidad_fx_30d", "float64", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_ECB_CONTRACT = TableContract(
    table_name="enrichment_ecb",
    layer="silver",
    primary_keys=["fecha"],
    columns=[
        ColumnContract("fecha", "string", False),
        ColumnContract("tasa_eur_usd", "float64", False),
        ColumnContract("tasa_eur_cop", "float64", True),
        ColumnContract("tasa_eur_inr", "float64", True),
        ColumnContract("tasa_eur_gbp", "float64", True),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_TRM_GOV_CONTRACT = TableContract(
    table_name="enrichment_trm_gov",
    layer="silver",
    primary_keys=["vigenciadesde"],
    columns=[
        ColumnContract("vigenciadesde", "string", False),
        ColumnContract("vigenciahasta", "string", True),
        ColumnContract("valor", "float64", False),
        ColumnContract("unidad", "string", True),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_TASAS_ACTIVAS_CONTRACT = TableContract(
    table_name="enrichment_tasas_activas",
    layer="silver",
    # (fecha_corte, nombre_entidad, tipo_de_cr_dito) no es único por sí solo: el mismo
    # trío tiene una fila por cada combinación de plazo/garantía/rango de monto.
    primary_keys=[],
    columns=[
        ColumnContract("fecha_corte", "string", False),
        ColumnContract("tipo_entidad", "string", True),
        ColumnContract("nombre_tipo_entidad", "string", True),
        ColumnContract("codigo_entidad", "string", True),
        ColumnContract("nombre_entidad", "string", False),
        ColumnContract("tipo_de_persona", "string", True),
        ColumnContract("sexo", "string", True),
        ColumnContract("tama_o_de_empresa", "string", True),
        ColumnContract("tipo_de_cr_dito", "string", False),
        ColumnContract("tipo_de_garant_a", "string", True),
        ColumnContract("producto_de_cr_dito", "string", True),
        ColumnContract("plazo_de_cr_dito", "string", True),
        ColumnContract("tasa_efectiva_promedio", "float64", False),
        ColumnContract("margen_adicional_a_la", "float64", True),
        ColumnContract("montos_desembolsados", "float64", True),
        ColumnContract("numero_de_creditos", "int64", True),
        ColumnContract("antiguedad_de_la_empresa", "string", True),
        ColumnContract("tipo_de_tasa", "string", True),
        ColumnContract("rango_monto_desembolsado", "string", True),
        ColumnContract("clase_deudor", "string", True),
        ColumnContract("codigo_ciiu", "string", True),
        ColumnContract("codigo_municipio", "string", True),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_TASAS_CAPTACION_CONTRACT = TableContract(
    table_name="enrichment_tasas_captacion",
    layer="silver",
    primary_keys=["fechacorte", "nombreentidad", "descripcion"],
    columns=[
        ColumnContract("fechacorte", "string", False),
        ColumnContract("tipoentidad", "string", True),
        ColumnContract("codigoentidad", "string", True),
        ColumnContract("nombreentidad", "string", False),
        ColumnContract("uca", "string", True),
        ColumnContract("nombre_unidad_de_captura", "string", True),
        ColumnContract("subcuenta", "string", True),
        ColumnContract("descripcion", "string", False),
        ColumnContract("tasa", "float64", True),
        ColumnContract("monto", "float64", True),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)

SILVER_ENRICHMENT_SECOP_CONTRATOS_CONTRACT = TableContract(
    table_name="enrichment_secop_contratos",
    layer="silver",
    primary_keys=["id_contrato"],
    columns=[
        ColumnContract("id_contrato", "string", False),
        ColumnContract("nombre_entidad", "string", True),
        ColumnContract("nit_entidad", "string", True),
        ColumnContract("departamento", "string", True),
        ColumnContract("ciudad", "string", True),
        ColumnContract("estado_contrato", "string", True),
        ColumnContract("tipo_de_contrato", "string", True),
        ColumnContract("modalidad_de_contratacion", "string", True),
        ColumnContract("fecha_de_fin_del_contrato", "string", True),
        ColumnContract("proveedor_adjudicado", "string", True),
        ColumnContract("documento_proveedor", "string", True),
        ColumnContract("es_pyme", "string", True),
        ColumnContract("valor_del_contrato", "float64", True),
        ColumnContract("valor_pagado", "float64", True),
        ColumnContract("source_system", "string", False),
        ColumnContract("ingested_at", "string", False),
    ]
)



SILVER_LENDINGCLUB_CONTRACT = TableContract(
    table_name="lendingclub",
    layer="silver",
    primary_keys=["id"],
    columns=[
        ColumnContract("id", "int64", False, "ID del préstamo en LendingClub"),
        ColumnContract("issue_d", "date", False, "Mes de originación"),
        ColumnContract("term", "int32", False, "Plazo en meses (36 o 60)"),
        ColumnContract("int_rate", "float64", True, "Tasa asignada por LendingClub (%)"),
        ColumnContract("installment", "float64", True, "Cuota mensual"),
        ColumnContract("grade", "string", True, "Grado de riesgo asignado por LendingClub"),
        ColumnContract("sub_grade", "string", True, "Subgrado de riesgo asignado por LendingClub"),
        ColumnContract("loan_amnt", "int64", True, "Monto solicitado"),
        ColumnContract("funded_amnt", "int64", True, "Monto desembolsado"),
        ColumnContract("emp_length", "int32", True, "Antigüedad laboral en años (0-10)"),
        ColumnContract("earliest_cr_line", "date", True, "Apertura de la primera línea de crédito"),
        ColumnContract("revol_util", "float64", True, "Utilización de crédito rotativo (%)"),
        ColumnContract("loan_status", "string", False, "Estado del préstamo al corte del dataset"),
        ColumnContract("last_pymnt_d", "date", True, "Mes del último pago recibido"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Timestamp UTC de ingesta"),
    ] + [
        ColumnContract(name, "as-is", True, "Columna de LendingClub sin transformar")
        for name in [
            "home_ownership", "annual_inc", "verification_status", "purpose", "addr_state", "zip_code",
            "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
            "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
            "total_acc", "initial_list_status", "application_type", "mort_acc", "pub_rec_bankruptcies",
            "acc_open_past_24mths", "bc_util", "num_actv_rev_tl", "tot_cur_bal", "total_rev_hi_lim",
            "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
        ]
    ],
)
