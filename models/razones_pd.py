"""Razones principales de una PD, para cartas de acción adversa.

Pregunta que responde cada razón: ¿cuánto bajaría la PD de este solicitante si este
factor tuviera el valor de un buen pagador típico? La referencia de cada feature sale
de los préstamos pagados del entrenamiento: mediana (numéricas) o moda (categóricas).
Las features que más suben la PD frente a su referencia son las razones.

Por qué no SHAP: el boosting divide las categóricas por conjuntos de categorías, y el
TreeExplainer de shap no soporta esas divisiones. Este método funciona sobre el modelo
calibrado completo (el mismo que decide), es determinístico y no agrega dependencias.
No es aditivo como SHAP; para una carta de acción adversa basta el orden de las razones.
"""
import numpy as np
import pandas as pd

# Etiqueta legible y redacción de la razón cuando el valor del solicitante queda por
# encima / por debajo de la referencia.
DESCRIPCIONES = {
    "loan_amnt": ("Monto solicitado", "Monto solicitado alto", "Monto solicitado bajo"),
    "term": ("Plazo (meses)", "Plazo largo", "Plazo corto"),
    "emp_length": ("Antigüedad laboral (años)", "Antigüedad laboral alta", "Antigüedad laboral corta"),
    "annual_inc": ("Ingreso anual", "Ingreso anual alto", "Ingreso anual bajo"),
    "monto_sobre_ingreso": ("Monto / ingreso anual", "Monto alto en relación con el ingreso",
                            "Monto bajo en relación con el ingreso"),
    "dti": ("Relación deuda/ingreso (DTI, %)", "Endeudamiento alto frente al ingreso",
            "Endeudamiento bajo frente al ingreso"),
    "delinq_2yrs": ("Moras de 30+ días en 2 años", "Moras recientes en el historial", "Pocas moras recientes"),
    "fico_promedio": ("Puntaje FICO", "Puntaje FICO alto", "Puntaje FICO bajo"),
    "meses_historial_credito": ("Antigüedad del historial (meses)", "Historial de crédito largo",
                                "Historial de crédito corto"),
    "inq_last_6mths": ("Consultas de crédito en 6 meses", "Muchas consultas de crédito recientes",
                       "Pocas consultas de crédito recientes"),
    "mths_since_last_delinq": ("Meses desde la última mora", "Mora lejana en el tiempo", "Mora reciente"),
    "mths_since_last_record": ("Meses desde el último registro público", "Registro público lejano",
                               "Registro público reciente"),
    "open_acc": ("Cuentas abiertas", "Muchas cuentas de crédito abiertas", "Pocas cuentas de crédito abiertas"),
    "pub_rec": ("Registros públicos negativos", "Registros públicos negativos", "Sin registros públicos"),
    "revol_bal": ("Saldo rotativo", "Saldo rotativo alto", "Saldo rotativo bajo"),
    "revol_util": ("Uso del crédito rotativo (%)", "Uso alto del crédito rotativo", "Uso bajo del crédito rotativo"),
    "total_acc": ("Total de cuentas en el historial", "Muchas cuentas en el historial",
                  "Pocas cuentas en el historial"),
    "mort_acc": ("Créditos hipotecarios", "Varios créditos hipotecarios", "Sin créditos hipotecarios"),
    "pub_rec_bankruptcies": ("Quiebras registradas", "Quiebras registradas", "Sin quiebras registradas"),
    "acc_open_past_24mths": ("Cuentas abiertas en 24 meses", "Muchas cuentas abiertas recientemente",
                             "Pocas cuentas abiertas recientemente"),
    "bc_util": ("Uso de tarjetas bancarias (%)", "Uso alto de tarjetas de crédito", "Uso bajo de tarjetas de crédito"),
    "num_actv_rev_tl": ("Líneas rotativas activas", "Muchas líneas rotativas activas",
                        "Pocas líneas rotativas activas"),
    "tot_cur_bal": ("Saldo total vigente", "Saldo total vigente alto", "Saldo total vigente bajo"),
    "total_rev_hi_lim": ("Cupo rotativo total", "Cupo rotativo total alto", "Cupo rotativo total bajo"),
    "home_ownership": ("Tenencia de vivienda", None, None),
    "verification_status": ("Verificación de ingresos", None, None),
    "purpose": ("Propósito del préstamo", None, None),
    "application_type": ("Tipo de solicitud", None, None),
}


def referencias(train: pd.DataFrame, numericas: list[str], categoricas: list[str]) -> dict:
    """Valor de un buen pagador típico por feature, sobre los préstamos pagados.

    Si la mayoría de los buenos pagadores no tiene el dato, la referencia es "sin
    dato": en campos como mths_since_last_delinq el vacío significa "nunca tuvo
    mora", y ese es el perfil típico, no la mediana de quienes sí la tuvieron."""
    pagados = train[~train["es_default"].astype(bool)]
    ref = {
        c: float("nan") if pagados[c].isna().mean() > 0.5 else float(pagados[c].median())
        for c in numericas
    }
    return ref | {c: pagados[c].mode().iloc[0] for c in categoricas}


def contribuciones(predecir, df: pd.DataFrame, refs: dict) -> pd.DataFrame:
    """PD(solicitante) - PD(solicitante con la feature en su referencia), por feature.
    Positivo: la feature sube la PD frente a un buen pagador típico.

    `predecir` recibe un DataFrame con las features crudas y devuelve un array de PD;
    todas las variantes se evalúan en una sola llamada al modelo."""
    features = list(refs)
    base = df[features].reset_index(drop=True)
    n = len(base)
    # Bloque i = todos los solicitantes con la feature i en su referencia. Las numéricas
    # pasan a float para admitir una referencia "sin dato" (NaN) en columnas enteras.
    variantes = base.iloc[np.tile(np.arange(n), len(features))].reset_index(drop=True)
    numericas = [c for c in features if not isinstance(refs[c], str)]
    variantes[numericas] = variantes[numericas].astype(float)
    for i, col in enumerate(features):
        variantes.loc[i * n:(i + 1) * n - 1, col] = refs[col]
    pd_base = predecir(base)
    pd_variantes = predecir(variantes).reshape(len(features), n)
    return pd.DataFrame((pd_base[None, :] - pd_variantes).T, columns=features)


def razones(fila: pd.Series, contrib: pd.Series, refs: dict, top: int = 4) -> list[dict]:
    """Las `top` features que más suben la PD, redactadas para una carta de acción adversa."""
    salida = []
    for col, aporte in contrib.sort_values(ascending=False).items():
        if aporte <= 0 or len(salida) == top:
            break
        etiqueta, si_alto, si_bajo = DESCRIPCIONES.get(col, (col, None, None))
        valor, ref = fila[col], refs[col]
        if isinstance(ref, str):
            texto = f"{etiqueta}: {valor}"
        elif pd.isna(valor):
            texto = f"Sin información de {etiqueta.lower()}"
        elif pd.isna(ref):
            texto = f"{etiqueta}: {valor:g}"           # p.ej. "Meses desde la última mora: 7"
        else:
            texto = si_alto if valor > ref else si_bajo
        salida.append({
            "feature": col,
            "razon": texto,
            "valor": None if pd.isna(valor) else valor,
            "referencia": None if not isinstance(ref, str) and pd.isna(ref) else ref,
            "aumento_pd": float(aporte),
        })
    return salida


def frecuencia_razones(contrib: pd.DataFrame, top: int = 4) -> pd.Series:
    """Fracción de una población en la que cada feature aparece entre sus `top` razones."""
    rangos = contrib.where(contrib > 0).rank(axis=1, ascending=False, method="first")
    return ((rangos <= top).sum() / len(contrib)).sort_values(ascending=False)
