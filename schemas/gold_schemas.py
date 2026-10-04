"""Definición de contratos para la capa Gold (Star Schema)."""
from schemas.contracts import ColumnContract, TableContract

GOLD_DIM_ENTIDAD_FINANCIERA_CONTRACT = TableContract(
    table_name="dim_entidad_financiera",
    layer="gold",
    primary_keys=["sk_entidad"],
    columns=[
        ColumnContract("sk_entidad", "string", False, "Surrogate Key (MD5 de source_system + codigo_entidad)"),
        ColumnContract("codigo_entidad", "string", False, "Código de la entidad en Superfinanciera"),
        ColumnContract("nombre_entidad", "string", False, "Razón social de la entidad vigilada"),
        ColumnContract("tipo_entidad", "string", True, "Código del tipo de entidad"),
        ColumnContract("nombre_tipo_entidad", "string", True, "Descripción del tipo de entidad"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_DIM_PROVEEDOR_ESTATAL_CONTRACT = TableContract(
    table_name="dim_proveedor_estatal",
    layer="gold",
    primary_keys=["sk_proveedor"],
    columns=[
        ColumnContract("sk_proveedor", "string", False, "Surrogate Key (MD5 de source_system + documento_proveedor)"),
        ColumnContract("documento_proveedor", "string", False, "NIT o documento del proveedor"),
        ColumnContract("proveedor_adjudicado", "string", True, "Razón social del proveedor"),
        ColumnContract("es_pyme", "bool", False, "Indicador de micro, pequeña o mediana empresa"),
        ColumnContract("source_system", "string", False, "Sistema origen (secop)"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_DIM_TIPO_CREDITO_CONTRACT = TableContract(
    table_name="dim_tipo_credito",
    layer="gold",
    primary_keys=["sk_tipo_credito"],
    columns=[
        ColumnContract("sk_tipo_credito", "string", False, "Surrogate Key (MD5 de categoria + nombre)"),
        ColumnContract("categoria", "string", False, "COLOCACION (crédito) o CAPTACION (fondeo)"),
        ColumnContract("nombre_tipo", "string", False, "Modalidad de crédito o instrumento de captación"),
        ColumnContract("producto", "string", True, "Producto o unidad de captura asociada"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_DIM_CLIENTE_CONTRACT = TableContract(
    table_name="dim_cliente",
    layer="gold",
    primary_keys=["sk_cliente"],
    foreign_keys={
        "sk_entidad": "dim_entidad_financiera.sk_entidad",
    },
    columns=[
        ColumnContract("sk_cliente", "string", False, "Surrogate Key (MD5 hash of source_system + natural_id)"),
        ColumnContract("sk_entidad", "string", False, "FK a dim_entidad_financiera (miembro DESCONOCIDO si la fuente no reporta entidad)"),
        ColumnContract("source_system", "string", False, "Sistema fuente de origen"),
        ColumnContract("natural_id", "string", False, "ID natural en el sistema origen"),
        ColumnContract("genero", "string", True, "Género del cliente"),
        ColumnContract("ubicacion", "string", True, "Ubicación geográfica del cliente"),
        ColumnContract("edad", "int64", True, "Edad del cliente"),
        ColumnContract("ocupacion", "string", True, "Trabajo u ocupación"),
        ColumnContract("estado_civil", "string", True, "Estado civil"),
        ColumnContract("nivel_educativo", "string", True, "Nivel educativo"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_DIM_FECHA_CONTRACT = TableContract(
    table_name="dim_fecha",
    layer="gold",
    primary_keys=["sk_fecha"],
    columns=[
        ColumnContract("sk_fecha", "int64", False, "Surrogate Key de fecha (YYYYMMDD)"),
        ColumnContract("fecha", "string", False, "Fecha ISO (YYYY-MM-DD)"),
        ColumnContract("anio", "int64", False, "Año"),
        ColumnContract("mes", "int64", False, "Mes (1-12)"),
        ColumnContract("dia", "int64", False, "Día del mes (1-31)"),
        ColumnContract("trimestre", "int64", False, "Trimestre (1-4)"),
        ColumnContract("dia_semana", "string", False, "Día de la semana"),
        ColumnContract("es_fin_de_semana", "bool", False, "Verdadero si es sábado o domingo"),
    ]
)

GOLD_DIM_MONEDA_CONTRACT = TableContract(
    table_name="dim_moneda",
    layer="gold",
    primary_keys=["sk_moneda"],
    columns=[
        ColumnContract("sk_moneda", "string", False, "Surrogate key (codigo_iso)"),
        ColumnContract("codigo_iso", "string", False, "Código ISO 4217 (USD, EUR, INR, COP)"),
        ColumnContract("nombre_moneda", "string", False, "Nombre formal de la divisa"),
        ColumnContract("simbolo", "string", False, "Símbolo de la divisa ($, €, ₹)"),
    ]
)

GOLD_FACT_TRANSACCION_CONTRACT = TableContract(
    table_name="fact_transaccion",
    layer="gold",
    primary_keys=["id_transaccion"],
    foreign_keys={
        "sk_cliente": "dim_cliente.sk_cliente",
        "sk_fecha": "dim_fecha.sk_fecha",
        "sk_moneda": "dim_moneda.sk_moneda",
    },
    columns=[
        ColumnContract("id_transaccion", "string", False, "ID único o compuesto de transacción"),
        ColumnContract("sk_cliente", "string", False, "Clave foránea a dim_cliente"),
        ColumnContract("sk_fecha", "int64", False, "Clave foránea a dim_fecha"),
        ColumnContract("sk_moneda", "string", False, "Clave foránea a dim_moneda"),
        ColumnContract("tipo_transaccion", "string", False, "Tipo (TRANSFER, CASH_OUT, PAYMENT, etc.)"),
        ColumnContract("monto_original", "float64", False, "Monto en moneda de origen"),
        ColumnContract("tasa_cambio_usd", "float64", False, "Tasa de conversión a USD aplicada"),
        ColumnContract("monto_usd", "float64", False, "Monto homologado a USD"),
        ColumnContract("monto_cop", "float64", False, "Monto homologado a COP (tasa Yahoo Finance)"),
        ColumnContract("monto_cop_trm", "float64", True, "Monto homologado a COP con TRM oficial Superfinanciera"),
        ColumnContract("tasa_trm_oficial", "float64", True, "TRM oficial vigente en la fecha (COP/USD)"),
        ColumnContract("tasa_ecb_eur_usd", "float64", True, "Tasa de referencia BCE EUR/USD en la fecha"),
        ColumnContract("saldo_previo", "float64", True, "Saldo previo en cuenta"),
        ColumnContract("saldo_nuevo", "float64", True, "Saldo posterior a transacción"),
        ColumnContract("es_fraude", "bool", False, "Indicador de fraude"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_FACT_CAMPANA_MARCADO_CONTRACT = TableContract(
    table_name="fact_campana_marcado",
    layer="gold",
    primary_keys=["id_campana_contacto"],
    foreign_keys={
        "sk_cliente": "dim_cliente.sk_cliente",
        "sk_fecha": "dim_fecha.sk_fecha",
        "sk_tipo_credito": "dim_tipo_credito.sk_tipo_credito",
    },
    columns=[
        ColumnContract("id_campana_contacto", "string", False, "ID único de contacto de campaña"),
        ColumnContract("sk_cliente", "string", False, "Clave foránea a dim_cliente"),
        ColumnContract("sk_fecha", "int64", False, "Clave foránea a dim_fecha de contacto"),
        ColumnContract("sk_tipo_credito", "string", False, "Clave foránea a dim_tipo_credito"),
        ColumnContract("balance_eur", "float64", False, "Balance original del cliente (EUR)"),
        ColumnContract("balance_usd", "float64", False, "Balance homologado a USD"),
        ColumnContract("tiene_hipoteca", "bool", False, "Indicador de préstamo de vivienda"),
        ColumnContract("tiene_prestamo_personal", "bool", False, "Indicador de préstamo personal"),
        ColumnContract("tiene_mora", "bool", False, "Indicador de mora de crédito"),
        ColumnContract("tasa_ibr_referencia", "float64", False, "Tasa IBR benchmark del día de contacto"),
        ColumnContract("tasa_activa_mercado", "float64", True, "Tasa activa promedio del mercado colombiano (Superfinanciera)"),
        ColumnContract("tasa_captacion_mercado", "float64", True, "Tasa de captación promedio del mercado (costo de fondeo)"),
        ColumnContract("spread_tasa_credito", "float64", False, "Spread de riesgo aplicado sobre IBR (%)"),
        ColumnContract("tasa_oferta_estimada", "float64", False, "Tasa de interés final ofertada (%)"),
        ColumnContract("duracion_contacto_seg", "int64", False, "Duración de la llamada"),
        ColumnContract("resultado_previo", "string", True, "Resultado campaña previa"),
        ColumnContract("suscrito_deposito", "bool", False, "Resultado final (conversion)"),
        ColumnContract("source_system", "string", False, "Sistema origen (bank_marketing)"),
    ]
)

GOLD_FACT_FRAUDE_TARJETA_CONTRACT = TableContract(
    table_name="fact_fraude_tarjeta",
    layer="gold",
    primary_keys=["id_evento_tarjeta"],
    columns=[
        ColumnContract("id_evento_tarjeta", "string", False, "ID único del evento de tarjeta"),
        ColumnContract("segundos_transcurridos", "float64", False, "Tiempo transcurrido"),
        ColumnContract("monto_usd", "float64", False, "Monto de transacción (USD)"),
        ColumnContract("volatilidad_fx_30d", "float64", False, "Volatilidad del mercado FX"),
        ColumnContract("banda_riesgo_macro", "string", False, "Banda de riesgo macro (ALTO, MEDIO, BAJO)"),
        ColumnContract("es_fraude", "bool", False, "Target de fraude (ULB CreditCard)"),
        ColumnContract("source_system", "string", False, "Sistema origen (creditcard)"),
        # V1..V28: componentes PCA anonimizados del dataset ULB. Son la única señal
        # predictiva real de fraude; sin ellas un modelo solo ve monto y tiempo.
        *[ColumnContract(f"V{i}", "float64", True, f"Componente PCA anonimizado {i}") for i in range(1, 29)],
    ]
)

GOLD_FACT_CONTRATO_ESTATAL_CONTRACT = TableContract(
    table_name="fact_contrato_estatal",
    layer="gold",
    primary_keys=["id_contrato"],
    foreign_keys={
        "sk_proveedor": "dim_proveedor_estatal.sk_proveedor",
        "sk_fecha": "dim_fecha.sk_fecha",
    },
    columns=[
        ColumnContract("id_contrato", "string", False, "ID único del contrato en SECOP II"),
        ColumnContract("sk_proveedor", "string", False, "Clave foránea a dim_proveedor_estatal"),
        ColumnContract("sk_fecha", "int64", False, "FK a dim_fecha (fin del contrato; miembro DESCONOCIDO si no reportada)"),
        ColumnContract("nombre_entidad_contratante", "string", True, "Entidad estatal contratante"),
        ColumnContract("nit_entidad", "string", True, "NIT de la entidad contratante"),
        ColumnContract("departamento", "string", True, "Departamento de ejecución"),
        ColumnContract("ciudad", "string", True, "Ciudad de ejecución"),
        ColumnContract("estado_contrato", "string", True, "Estado de ejecución del contrato"),
        ColumnContract("tipo_de_contrato", "string", True, "Tipo de contrato"),
        ColumnContract("modalidad_de_contratacion", "string", True, "Modalidad de contratación"),
        ColumnContract("valor_del_contrato", "float64", True, "Valor total del contrato (COP)"),
        ColumnContract("valor_pagado", "float64", True, "Valor efectivamente pagado (COP)"),
        ColumnContract("valor_pendiente", "float64", True, "Valor pendiente de pago (COP), base para factoring estatal"),
        ColumnContract("source_system", "string", False, "Sistema origen (secop)"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

GOLD_FACT_TASAS_MERCADO_CONTRACT = TableContract(
    table_name="fact_tasas_mercado",
    layer="gold",
    primary_keys=["id_observacion_tasa"],
    foreign_keys={
        "sk_fecha": "dim_fecha.sk_fecha",
        "sk_entidad": "dim_entidad_financiera.sk_entidad",
        "sk_tipo_credito": "dim_tipo_credito.sk_tipo_credito",
    },
    columns=[
        ColumnContract("id_observacion_tasa", "string", False, "ID único de la observación de tasa"),
        ColumnContract("sk_fecha", "int64", False, "Clave foránea a dim_fecha"),
        ColumnContract("sk_entidad", "string", False, "FK a dim_entidad_financiera (DESCONOCIDO para tasas macro)"),
        ColumnContract("sk_tipo_credito", "string", False, "FK a dim_tipo_credito (DESCONOCIDO para tasas macro)"),
        ColumnContract("tipo_tasa", "string", False,
                       "IBR_OVERNIGHT, TREASURY_10Y, TBILL_3M, TASA_ACTIVA, TASA_CAPTACION, "
                       "FX_USDCOP, FX_USDEUR, FX_USDINR, TRM_OFICIAL, ECB_EURUSD"),
        ColumnContract("valor_tasa", "float64", False, "Valor de la tasa observada (%)"),
        ColumnContract("monto_asociado", "float64", True, "Monto desembolsado o captado asociado, si la fuente lo reporta"),
        ColumnContract("source_system", "string", False, "Sistema origen"),
        ColumnContract("ingested_at", "string", False, "Lineage timestamp"),
    ]
)

# Grano: un préstamo de LendingClub. Las columnas de origen conservan su nombre;
# las derivadas (sk_fecha, fico_promedio, meses_historial_credito, es_default,
# madurado, fecha_corte, estado_final) se documentan en etl/gold/lendingclub.py.
GOLD_FACT_PRESTAMO_MINORISTA_CONTRACT = TableContract(
    table_name="fact_prestamo_minorista",
    layer="gold",
    primary_keys=["id_prestamo"],
    foreign_keys={"sk_fecha": "dim_fecha.sk_fecha"},
    columns=[
        ColumnContract("id_prestamo", "int64", False, "ID del préstamo en LendingClub"),
        ColumnContract("sk_fecha", "int64", False, "FK a dim_fecha: mes de originación (YYYYMM01)"),
        ColumnContract("issue_d", "date", False, "Mes de originación"),
        ColumnContract("term", "int32", False, "Plazo en meses"),
        ColumnContract("fico_promedio", "float64", True, "Punto medio del rango FICO al originar"),
        ColumnContract("meses_historial_credito", "int64", True, "Meses entre la primera línea de crédito y la originación"),
        ColumnContract("estado_final", "string", False, "loan_status al corte del dataset"),
        ColumnContract("es_default", "bool", True, "TRUE castigado/default, FALSE pagado, NULL sin desenlace"),
        ColumnContract("madurado", "bool", False, "El plazo más 6 meses de gracia transcurrió antes del corte"),
        ColumnContract("fecha_corte", "date", False, "Último mes de originación presente en el dataset"),
    ] + [
        ColumnContract(name, "as-is", True, "Columna de LendingClub al originar (ver LCDataDictionary.xlsx)")
        for name in [
            "loan_amnt", "funded_amnt", "int_rate", "installment", "grade", "sub_grade", "emp_length",
            "home_ownership", "annual_inc", "verification_status", "purpose", "addr_state", "zip_code",
            "dti", "delinq_2yrs", "fico_range_low", "fico_range_high", "inq_last_6mths",
            "mths_since_last_delinq", "mths_since_last_record", "open_acc", "pub_rec", "revol_bal",
            "revol_util", "total_acc", "initial_list_status", "application_type", "mort_acc",
            "pub_rec_bankruptcies", "acc_open_past_24mths", "bc_util", "num_actv_rev_tl",
            "tot_cur_bal", "total_rev_hi_lim",
        ]
    ] + [
        ColumnContract(name, "as-is", True, "Desenlace del préstamo: conocido solo después de originar")
        for name in ["last_pymnt_d", "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee"]
    ] + [
        ColumnContract("ead_al_default", "float64", True, "Capital pendiente al castigo (solo castigados)"),
        ColumnContract("recuperacion_neta", "float64", True, "Recuperaciones menos costo de cobranza (solo castigados)"),
        ColumnContract("lgd_realizada", "float64", True, "1 - recuperacion_neta / ead_al_default, en [0, 1]"),
        ColumnContract("perdida_realizada", "float64", True, "Pérdida en dinero: castigados > 0, pagados 0, sin desenlace NULL"),
    ],
)
