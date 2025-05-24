import streamlit as st
import pandas as pd
import joblib
import os
from dotenv import load_dotenv
import numpy as np
import json
import requests
import traceback

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="PeraBank Risk Analyzer")

# Cargar variables de entorno
load_dotenv()

# --- Configuración de Gemini ---
GEMINI_API_KEY = os.getenv("API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")  # Modelo actualizado
GEMINI_API_ENDPOINT = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent"

# Verificar configuración de Gemini
gemini_api_configured = bool(GEMINI_API_KEY)

# --- Carga del Modelo y Datos ---
MODEL_PATH = "perabank_risk_pipeline_v1.joblib"
model_pipeline = None
if os.path.exists(MODEL_PATH):
    try:
        model_pipeline = joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error cargando el modelo: {str(e)}")

DATA_PATH = "output_enhanced/usuarios.csv" 
df_display_data = pd.DataFrame()
if os.path.exists(DATA_PATH):
    try:
        df_display_data = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.warning(f"Error cargando datos: {str(e)}")

# --- Funciones Principales ---
def generate_gemini_explanation(prompt_text, generation_config=None):
    """
    Función mejorada para llamar a la API de Gemini
    """
    if not gemini_api_configured:
        return False, "API Key no configurada"
    
    headers = {'Content-Type': 'application/json'}
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }]
    }
    
    if generation_config:
        payload["generationConfig"] = generation_config
    
    try:
        response = requests.post(
            f"{GEMINI_API_ENDPOINT}?key={GEMINI_API_KEY}",
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        response_data = response.json()
        
        if response_data.get("candidates"):
            text_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
            return True, text_response
        elif response_data.get("promptFeedback"):
            return False, f"Error en el prompt: {response_data['promptFeedback']}"
        else:
            return False, "Respuesta inesperada de la API"
            
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {str(e)}"
    except Exception as e:
        return False, f"Error inesperado: {str(e)}"

# --- Interfaz de Usuario ---
LOGO_PATH = "logo_perabank.png"
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=200)
st.title("PeraBank 🏦 - Analizador de Riesgo Crediticio (Beta)")

# Definir pestañas
tab1, tab2, tab3 = st.tabs(["📊 Dashboard Clientes", "🔍 Predicción de Riesgo Individual", "💡 Explicación IA (Gemini)"])

# --- Pestaña 1: Dashboard Clientes ---
with tab1:
    st.header("📊 Dashboard Analítico de Clientes PeraBank")

    if not df_display_data.empty:
        st.markdown("### Métricas Clave de la Base de Clientes")
        
        # Asegúrate de que la columna 'riesgo' exista si la vas a usar aquí.
        # Si 'riesgo' se crea solo para el entrenamiento del modelo y no está en df_display_data,
        # necesitarías cargarla o calcularla aquí también, o basar los KPIs en otras métricas.
        # Por ahora, asumiré que 'riesgo' (0 o 1) está en df_display_data para algunos KPIs.
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Clientes", f"{len(df_display_data)}")
        
        if 'ingreso_mensual_estimado' in df_display_data.columns:
            ingreso_promedio = df_display_data['ingreso_mensual_estimado'].mean()
            col2.metric("Ingreso Promedio", f"${ingreso_promedio:,.0f} COP")
        
        if 'edad' in df_display_data.columns:
            edad_promedio = df_display_data['edad'].mean()
            col3.metric("Edad Promedio", f"{edad_promedio:.1f} años")

        if 'riesgo' in df_display_data.columns: # Asumiendo que tienes una columna 'riesgo' (0 o 1)
            riesgo_promedio_pct = df_display_data['riesgo'].mean() * 100
            col4.metric("Tasa de 'Alto Riesgo'", f"{riesgo_promedio_pct:.2f}%")
        
        st.markdown("---")

        # --- Visualizaciones Detalladas ---
        st.subheader("Análisis Demográfico y Financiero")

        c1, c2 = st.columns(2)

        with c1:
            # Distribución de Edad
            if 'edad' in df_display_data.columns:
                st.markdown("#### Distribución de Edad")
                # Usar un histograma real con matplotlib o plotly para mejor visualización
                # Aquí un ejemplo simple con st.bar_chart si 'rango_edad' ya existe
                if 'rango_edad' in df_display_data.columns:
                    rango_edad_counts = df_display_data['rango_edad'].value_counts().sort_index()
                    st.bar_chart(rango_edad_counts)
                else: # Fallback a histograma simple si no hay rango_edad
                    try:
                        # Importar matplotlib.pyplot solo cuando se necesita
                        import matplotlib.pyplot as plt
                        fig_edad, ax_edad = plt.subplots()
                        df_display_data['edad'].plot(kind='hist', ax=ax_edad, bins=10, edgecolor='black')
                        ax_edad.set_title("Histograma de Edad")
                        ax_edad.set_xlabel("Edad")
                        ax_edad.set_ylabel("Frecuencia")
                        st.pyplot(fig_edad)
                    except ImportError:
                        st.warning("Matplotlib no instalado. No se puede mostrar el histograma de edad.")
                    except Exception as e:
                        st.warning(f"No se pudo generar el histograma de edad: {e}")
            
            # Estado Laboral
            if 'estado_laboral' in df_display_data.columns:
                st.markdown("#### Estado Laboral de Clientes")
                estado_laboral_counts = df_display_data['estado_laboral'].value_counts()
                st.bar_chart(estado_laboral_counts)

            # Nivel Educativo
            if 'nivel_educativo' in df_display_data.columns:
                st.markdown("#### Nivel Educativo")
                nivel_educativo_counts = df_display_data['nivel_educativo'].value_counts()
                st.bar_chart(nivel_educativo_counts)

        with c2:
            # Distribución de Ingresos
            if 'ingreso_mensual_estimado' in df_display_data.columns:
                st.markdown("#### Distribución de Ingresos Mensuales Estimados")
                if 'categoria_ingresos' in df_display_data.columns: # Si ya tienes categorías
                    categoria_ingresos_counts = df_display_data['categoria_ingresos'].value_counts().sort_index()
                    st.bar_chart(categoria_ingresos_counts)
                else: # Histograma para ingresos continuos
                    try:
                        import matplotlib.pyplot as plt
                        fig_ingreso, ax_ingreso = plt.subplots()
                        # Podrías querer filtrar outliers o usar escala logarítmica si la distribución es muy sesgada
                        ingresos_validos = df_display_data['ingreso_mensual_estimado'].dropna()
                        if not ingresos_validos.empty:
                            ax_ingreso.hist(ingresos_validos, bins=15, edgecolor='black')
                            ax_ingreso.set_title("Histograma de Ingresos")
                            ax_ingreso.set_xlabel("Ingreso Mensual Estimado (COP)")
                            ax_ingreso.set_ylabel("Frecuencia")
                            st.pyplot(fig_ingreso)
                        else:
                            st.write("No hay datos de ingresos válidos para el histograma.")
                    except ImportError:
                        st.warning("Matplotlib no instalado. No se puede mostrar el histograma de ingresos.")
                    except Exception as e:
                        st.warning(f"No se pudo generar el histograma de ingresos: {e}")
            
            # Tipo de Vivienda
            if 'tipo_vivienda' in df_display_data.columns:
                st.markdown("#### Tipo de Vivienda")
                tipo_vivienda_counts = df_display_data['tipo_vivienda'].value_counts()
                st.bar_chart(tipo_vivienda_counts)

            # Antigüedad del Cliente
            if 'antiguedad_cliente' in df_display_data.columns:
                st.markdown("#### Antigüedad del Cliente (Años)")
                try:
                    import matplotlib.pyplot as plt
                    fig_ant, ax_ant = plt.subplots()
                    antiguedad_validos = df_display_data['antiguedad_cliente'].dropna()
                    if not antiguedad_validos.empty:
                        ax_ant.hist(antiguedad_validos, bins=10, edgecolor='black')
                        ax_ant.set_title("Histograma de Antigüedad del Cliente")
                        ax_ant.set_xlabel("Años como Cliente")
                        ax_ant.set_ylabel("Frecuencia")
                        st.pyplot(fig_ant)
                    else:
                        st.write("No hay datos de antigüedad válidos para el histograma.")
                except ImportError:
                    st.warning("Matplotlib no instalado. No se puede mostrar el histograma de antigüedad.")
                except Exception as e:
                    st.warning(f"No se pudo generar el histograma de antigüedad: {e}")

        st.markdown("---")
        st.subheader("Análisis de Préstamos y Riesgo (Ejemplos)")

        # Asegúrate de que la columna 'riesgo' esté disponible y sea numérica (0 o 1)
        if 'riesgo' in df_display_data.columns and pd.api.types.is_numeric_dtype(df_display_data['riesgo']):
            # Convertir 'riesgo' a etiquetas para los gráficos si es numérico 0/1
            df_display_data['riesgo_etiqueta'] = df_display_data['riesgo'].apply(lambda x: 'Alto Riesgo' if x == 1 else 'Bajo Riesgo')
            
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### Distribución de Riesgo General")
                riesgo_counts = df_display_data['riesgo_etiqueta'].value_counts()
                st.bar_chart(riesgo_counts)

            with c4:
                # Riesgo por Categoría de Ingresos (si ambas columnas existen)
                if 'categoria_ingresos' in df_display_data.columns:
                    st.markdown("#### Riesgo por Categoría de Ingresos")
                    riesgo_por_ingreso = df_display_data.groupby('categoria_ingresos')['riesgo_etiqueta'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
                    st.bar_chart(riesgo_por_ingreso) # Muestra porcentajes de Alto/Bajo Riesgo por categoría
            
            # Podrías añadir más: Riesgo por Rango de Edad, Riesgo por Nivel Educativo, etc.
            if 'rango_edad' in df_display_data.columns:
                 st.markdown("#### Riesgo por Rango de Edad")
                 riesgo_por_edad = df_display_data.groupby('rango_edad')['riesgo_etiqueta'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
                 st.bar_chart(riesgo_por_edad)


        # Mostrar una porción de la tabla con más detalles (puede ser filtrable en el futuro)
        st.markdown("---")
        st.subheader("Vista Detallada de Clientes (Muestra)")
        cols_to_show_detailed = ['user_id', 'edad', 'rango_edad', 'profesion', 'ingreso_mensual_estimado', 
                                 'categoria_ingresos', 'estado_laboral', 'antiguedad_cliente', 
                                 'num_total_cuentas', 'saldo_total_en_cuentas', 'num_total_prestamos', 
                                 'monto_total_prestado']
        if 'riesgo_etiqueta' in df_display_data.columns: # Añadir la etiqueta de riesgo si existe
            cols_to_show_detailed.append('riesgo_etiqueta')
        
        cols_to_show_detailed = [col for col in cols_to_show_detailed if col in df_display_data.columns]
        if cols_to_show_detailed:
            st.dataframe(df_display_data[cols_to_show_detailed].head(20))
        
    else:
        st.warning("No se pudieron cargar los datos para el dashboard. Verifica la ruta y el archivo.")

# ... (Importaciones y código de configuración inicial, carga de modelo PeraBank, carga de df_display_data) ...
# ... (Definición de la Pestaña 1) ...

# --- Pestaña 2: Predicción de Riesgo Individual ---
with tab2:
    st.header("Evaluar Riesgo de Nuevo Solicitante")
    
    # Estas listas deben coincidir con las columnas que tu pipeline espera ANTES del preprocesamiento
    # Es CRUCIAL que las definas correctamente según tu script de entrenamiento
    DEFAULT_CATEGORICAL_FEATURES = st.session_state.get('categorical_features_list', ['ciudad_residencia', 'rango_edad', 'profesion', 'categoria_ingresos', 
                                'estado_laboral', 'nivel_educativo', 'estado_civil', 'tipo_vivienda'])
    DEFAULT_NUMERICAL_FEATURES = st.session_state.get('numerical_features_list', ['edad', 'ingreso_mensual_estimado', 'numero_dependientes', 
                                'tiene_dependientes', 'vivienda_propia', 'antiguedad_cliente', 
                                'num_total_cuentas', 'saldo_total_en_cuentas', 
                                'promedio_transacciones_por_cuenta', 'monto_total_movido_por_usuario', 
                                'num_total_prestamos', 'monto_total_prestado', 'promedio_monto_prestamo'])
    
    ALL_MODEL_FEATURES = DEFAULT_NUMERICAL_FEATURES + DEFAULT_CATEGORICAL_FEATURES

    st.subheader("Por favor, ingresa los datos del solicitante:")
    
    with st.form(key='prediction_form'):
        feature_input_values = {} # Diccionario para guardar los inputs del formulario
        
        # Crear inputs dinámicamente o definirlos explícitamente
        # Es VITAL que crees un input para CADA feature en ALL_MODEL_FEATURES
        # o tengas una forma de asignarles un valor por defecto ANTES de crear input_data_df
        st.write("--- Datos Demográficos y Laborales ---")
        col1, col2, col3 = st.columns(3)
        with col1:
            feature_input_values['edad'] = st.number_input("Edad", min_value=18, max_value=100, value=35, step=1, key="edad_input")
            
            ESTADOS_LABORALES_OPTIONS = ['EMPLEADO', 'INDEPENDIENTE', 'DESEMPLEADO', 'ESTUDIANTE', 'PENSIONADO', 'OTRO', 'DESCONOCIDO']
            if 'estado_laboral' in df_display_data.columns and not df_display_data.empty:
                unique_options = sorted(list(set(df_display_data['estado_laboral'].dropna().astype(str).tolist() + ESTADOS_LABORALES_OPTIONS)))
                ESTADOS_LABORALES_OPTIONS = unique_options
            feature_input_values['estado_laboral'] = st.selectbox("Estado Laboral", ESTADOS_LABORALES_OPTIONS, index=ESTADOS_LABORALES_OPTIONS.index('EMPLEADO') if 'EMPLEADO' in ESTADOS_LABORALES_OPTIONS else 0, key="estado_lab_input")

        with col2:
            feature_input_values['numero_dependientes'] = st.number_input("Nº Dependientes", min_value=0, value=0, step=1, key="num_dep_input")
            
            NIVEL_EDUCATIVO_OPTIONS = ['NINGUNO', 'PRIMARIA', 'SECUNDARIA', 'TECNICO', 'UNIVERSITARIO', 'POSTGRADO', 'OTRO', 'DESCONOCIDO']
            if 'nivel_educativo' in df_display_data.columns and not df_display_data.empty:
                unique_options = sorted(list(set(df_display_data['nivel_educativo'].dropna().astype(str).tolist() + NIVEL_EDUCATIVO_OPTIONS)))
                NIVEL_EDUCATIVO_OPTIONS = unique_options
            feature_input_values['nivel_educativo'] = st.selectbox("Nivel Educativo", NIVEL_EDUCATIVO_OPTIONS, key="nivel_edu_input")

        with col3:
            feature_input_values['profesion'] = st.text_input("Profesión", "INGENIERO", key="profesion_input")
            
            ESTADOS_CIVILES_OPTIONS = ['SOLTERO(A)', 'CASADO(A)', 'UNION LIBRE', 'DIVORCIADO(A)', 'VIUDO(A)', 'DESCONOCIDO']
            if 'estado_civil' in df_display_data.columns and not df_display_data.empty:
                 unique_options = sorted(list(set(df_display_data['estado_civil'].dropna().astype(str).tolist() + ESTADOS_CIVILES_OPTIONS)))
                 ESTADOS_CIVILES_OPTIONS = unique_options
            feature_input_values['estado_civil'] = st.selectbox("Estado Civil", ESTADOS_CIVILES_OPTIONS, key="estado_civil_input")


        st.write("--- Datos Financieros y de Antigüedad ---")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            feature_input_values['ingreso_mensual_estimado'] = st.number_input("Ingreso Mensual (COP)", min_value=0, value=5000000, step=100000, key="ingreso_input")
            feature_input_values['antiguedad_cliente'] = st.number_input("Antigüedad cliente (años)", min_value=0, value=5, step=1, key="ant_cli_input")

        with col_f2:
            feature_input_values['num_total_prestamos'] = st.number_input("Nº Préstamos Históricos Totales", min_value=0, value=1, step=1, key="num_prest_hist_input")
            feature_input_values['saldo_total_en_cuentas'] = st.number_input("Saldo Total en Cuentas (COP)", min_value=-1000000, value=10000000, step=500000, key="saldo_cuentas_input")

        with col_f3:
            feature_input_values['num_total_cuentas'] = st.number_input("Nº Total de Cuentas", min_value=0, value=2, step=1, key="num_cuentas_input")
            # ... Añade el resto de inputs numéricos y categóricos que necesites ...
            # Por ejemplo:
            feature_input_values['ciudad_residencia'] = st.text_input("Ciudad de Residencia", "MEDELLIN", key="ciudad_res_input")
            TIPOS_VIVIENDA_OPTIONS = ['PROPIA', 'ALQUILADA', 'FAMILIAR', 'OTRO', 'DESCONOCIDO']
            if 'tipo_vivienda' in df_display_data.columns and not df_display_data.empty:
                unique_options = sorted(list(set(df_display_data['tipo_vivienda'].dropna().astype(str).tolist() + TIPOS_VIVIENDA_OPTIONS)))
                TIPOS_VIVIENDA_OPTIONS = unique_options
            feature_input_values['tipo_vivienda'] = st.selectbox("Tipo de Vivienda", TIPOS_VIVIENDA_OPTIONS, key="tipo_viv_input")

        # Asegúrate de tener inputs para TODAS las features en ALL_MODEL_FEATURES
        # o una lógica para asignarles defaults si no tienen un widget.

        st.markdown("---")
        submit_button = st.form_submit_button(label='Predecir Riesgo del Solicitante')

    if submit_button:
        print("DEBUG Pestaña 2: Botón 'Predecir Riesgo' presionado.") # DEBUG
        if model_pipeline:
            try:
                # Construir el DataFrame con todas las features que tu modelo espera
                input_data_for_model = {}
                print("DEBUG Pestaña 2: Construyendo input_data_for_model. Valores del formulario:") # DEBUG
                print(feature_input_values) # DEBUG

                for feature_name in ALL_MODEL_FEATURES:
                    # Usar un default si la feature no fue explícitamente ingresada en el formulario
                    # (esto es por si no creaste un widget para cada una de ALL_MODEL_FEATURES)
                    default_value = np.nan 
                    if feature_name in DEFAULT_NUMERICAL_FEATURES: 
                        default_value = 0.0 # Default para numéricas
                    elif feature_name in DEFAULT_CATEGORICAL_FEATURES: 
                        default_value = "DESCONOCIDO" # Default para categóricas
                    
                    input_data_for_model[feature_name] = feature_input_values.get(feature_name, default_value)
                
                # Lógica para features derivadas (rango_edad, categoria_ingresos)
                # SI Y SOLO SI no son generadas por tu pipeline de preprocesamiento
                # Si tu pipeline guardado (ColumnTransformer) espera 'edad' y luego crea 'rango_edad',
                # entonces 'rango_edad' NO debería estar en ALL_MODEL_FEATURES como input directo.
                # Asumiré por ahora que SÍ son inputs directos o que el pipeline los espera y los crea a partir de otros.
                # Si son features que tu pipeline genera internamente, entonces NO las definas aquí.
                # Solo debes proveer las features CRUDAS que el pipeline necesita para EMPEZAR su preprocesamiento.

                if 'rango_edad' in ALL_MODEL_FEATURES and 'edad' in input_data_for_model:
                    edad_val = input_data_for_model.get('edad', 0)
                    bins_edad=[0, 25, 35, 45, 55, 65, float('inf')]
                    labels_edad=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
                    if pd.notna(edad_val) and edad_val > 0: # Asegurar que edad_val sea válido
                         input_data_for_model['rango_edad'] = pd.cut(pd.Series([edad_val]), bins=bins_edad, labels=labels_edad, right=False).astype(str)[0]
                    else:
                         input_data_for_model['rango_edad'] = "DESCONOCIDO"

                if 'categoria_ingresos' in ALL_MODEL_FEATURES and 'ingreso_mensual_estimado' in input_data_for_model:
                    ingreso_val = input_data_for_model.get('ingreso_mensual_estimado', 0)
                    bins_ingreso=[-float('inf'), 1000000, 3000000, 6000000, 10000000, float('inf')]
                    labels_ingreso=['MUY BAJOS', 'BAJOS', 'MEDIOS', 'ALTOS', 'MUY ALTOS']
                    if pd.notna(ingreso_val): # Asegurar que ingreso_val sea válido
                        input_data_for_model['categoria_ingresos'] = pd.cut(pd.Series([ingreso_val]), bins=bins_ingreso, labels=labels_ingreso, right=False).astype(str)[0]
                    else:
                        input_data_for_model['categoria_ingresos'] = "DESCONOCIDO"
                
                # Crear el DataFrame para la predicción
                input_data_df = pd.DataFrame([input_data_for_model])
                
                # Reordenar columnas para que coincidan con el orden de entrenamiento si es necesario
                # Aunque ColumnTransformer usualmente maneja esto por nombre, ser explícito no daña.
                # Obtener las features que el preprocesador del pipeline realmente conoce:
                try:
                    # Si tu pipeline tiene un paso 'preprocessor' y este tiene 'feature_names_in_'
                    pipeline_feature_names = model_pipeline.named_steps['preprocessor'].feature_names_in_
                    # Asegurar que input_data_df solo tenga estas columnas y en este orden
                    input_data_df = input_data_df[pipeline_feature_names]
                except Exception as e_feat_names:
                    print(f"DEBUG Pestaña 2: No se pudieron obtener feature_names_in_ del preprocesador. Usando ALL_MODEL_FEATURES. Error: {e_feat_names}")
                    # Fallback si no se puede obtener del pipeline (menos robusto)
                    cols_para_pipeline = [col for col in ALL_MODEL_FEATURES if col in input_data_df.columns]
                    input_data_df = input_data_df[cols_para_pipeline]


                st.write("DEBUG Pestaña 2: Datos de entrada para el pipeline (DataFrame):") # DEBUG
                st.dataframe(input_data_df) # DEBUG

                prediction_numeric = model_pipeline.predict(input_data_df)[0]
                prediction_proba = model_pipeline.predict_proba(input_data_df)[0]
                riesgo_etiqueta = "BAJO RIESGO" if prediction_numeric == 0 else "ALTO RIESGO"
                
                print(f"DEBUG Pestaña 2: Predicción Numérica: {prediction_numeric}, Etiqueta: {riesgo_etiqueta}") # DEBUG
                print(f"DEBUG Pestaña 2: Probabilidades: {prediction_proba}") # DEBUG

                st.subheader("Resultado de la Predicción PeraBank:")
                if riesgo_etiqueta == "ALTO RIESGO": st.error(f"Predicción: {riesgo_etiqueta}")
                else: st.success(f"Predicción: {riesgo_etiqueta}")
                st.write(f"Probabilidad de Bajo Riesgo (Clase 0): {prediction_proba[0]:.2%}")
                st.write(f"Probabilidad de Alto Riesgo (Clase 1): {prediction_proba[1]:.2%}")
                
                # --- Guardado en st.session_state ---
                st.session_state.last_prediction_details = {
                    'input_features': input_data_for_model.copy(), # Guardar el diccionario de inputs
                    'label': riesgo_etiqueta,
                    'proba_low': float(prediction_proba[0]), # Convertir a float nativo
                    'proba_high': float(prediction_proba[1]), # Convertir a float nativo
                    'raw_numeric_prediction': int(prediction_numeric) # Convertir a int nativo
                }
                print("DEBUG Pestaña 2: Guardado en st.session_state.last_prediction_details:", st.session_state.last_prediction_details) # DEBUG
                st.success("Predicción completada y guardada para explicación.") # Feedback visual

            except Exception as e:
                st.error(f"Error durante la predicción PeraBank: {e}")
                st.text(traceback.format_exc()) # Imprime el traceback completo en la UI
                print(f"ERROR Pestaña 2: {traceback.format_exc()}") # También en la consola
        else:
            st.error("El modelo PeraBank no está cargado. No se puede predecir.")

# ... (Definición de la Pestaña 3 y la sidebar se mantienen igual que en mi respuesta anterior) ...

# --- Pestaña 3: Explicación con Gemini ---
with tab3:
    st.header("Explicación de la Predicción con IA")
    
    if not gemini_api_configured:
        st.warning("API de Gemini no configurada. Agrega tu API Key en un archivo .env")
    # CAMBIO IMPORTANTE AQUÍ: Cómo se guardan/leen los datos de la última predicción
    elif 'last_prediction_details' not in st.session_state: # Usaremos una clave diferente y más descriptiva
        st.info("Realiza una predicción primero en la pestaña anterior")
    else:
        # Mostrar datos de la última predicción
        pred_details = st.session_state.last_prediction_details # Leer de la nueva clave
        
        # features_text se construye a partir de pred_details['input_features']
        features_text = "\n".join([f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in pred_details['input_features'].items()])
        
        st.text_area("Datos del Solicitante", value=features_text, height=200, disabled=True)
        
        if pred_details['label'] == "ALTO RIESGO":
            st.error(f"Predicción: {pred_details['label']} (Probabilidad: {pred_details['proba_high']:.2%})")
        else:
            st.success(f"Predicción: {pred_details['label']} (Probabilidad: {pred_details['proba_low']:.2%})")
        
        with st.expander("⚙️ Configuración de Generación de Texto (Gemini)"):
            temperature = st.slider("Creatividad (Temperatura)", 0.0, 1.0, 0.7, key="gemini_temp")
            max_tokens = st.number_input("Máximo de Tokens de Salida", 100, 2048, 1000, key="gemini_tokens") # Ajustado max a 2048 para Flash

        if st.button("Generar Explicación con Gemini"):
            with st.spinner("Generando explicación..."):
                # Crear prompt
                prompt = f"""
                Eres un asistente experto en análisis de riesgo crediticio para PeraBank.
                Un solicitante de crédito presenta las siguientes características (estas fueron las usadas por el modelo de predicción):
                {json.dumps(pred_details['input_features'], indent=2, ensure_ascii=False, default=str)}

                Nuestro modelo interno de Machine Learning ha predicho que este solicitante es de: **{pred_details['label']}**.
                La probabilidad calculada por el modelo para que sea 'ALTO RIESGO' es {pred_details['proba_high']:.2%} y para 'BAJO RIESGO' es {pred_details['proba_low']:.2%}.

                Por favor, proporciona un análisis detallado para un evaluador de crédito:
                1. Factores clave (2-3) de los datos del solicitante que probablemente justifican esta predicción. Explica brevemente.
                2. ¿Características contradictorias o que merecerían una segunda revisión humana?
                3. Recomendación general (aprobar, rechazar, más info) y por qué.
                4. ¿Cómo ayudaría comparar este perfil con datos agregados de otros clientes de PeraBank?

                Presenta tu respuesta de forma clara, estructurada y profesional.
                """
                
                generation_config_payload = {
                    "temperature": temperature,
                    "maxOutputTokens": max_tokens
                    # Puedes añadir "topK", "topP" si quieres más control
                }
                
                # Llamar a Gemini
                success, response_text = generate_gemini_explanation(prompt, generation_config_payload)
                
                if success:
                    st.markdown("### Explicación Generada por Gemini")
                    st.markdown(response_text)
                    
                    # Guardar en historial (opcional)
                    if 'explanations' not in st.session_state:
                        st.session_state.explanations = []
                    st.session_state.explanations.append({
                        "features": pred_details['input_features'],
                        "prediction": pred_details['label'],
                        "explanation": response_text
                    })
                else:
                    st.error(f"Error generando explicación con Gemini: {response_text}")

# --- Barra Lateral ---
st.sidebar.header("Sobre PeraBank Beta")
st.sidebar.info("""
Esta aplicación es un prototipo para análisis de riesgo crediticio.
Usa modelos de ML y IA con fines demostrativos.
""")