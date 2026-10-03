import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza una predicción utilizando un modelo de Bagging pre-entrenado.")

# 1. Inputs del usuario
st.header("Datos de Entrada")

# Opciones para la variable Felder obtenidas de los datos reales o previstos
opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']

# Entradas del formulario
felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder)
examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01)

# Crear un DataFrame temporal con los datos del usuario
data_dict = {
    'Felder': [felder_input],
    'Examen_admisión': [examen_input]
}
df_input = pd.DataFrame(data_dict)

if st.button("Realizar Predicción"):
    try:
        # Copia de trabajo
        df_procesado = df_input.copy()
        
        # 2. Cargar y aplicar el transformador/columnas de One-Hot para la variable Felder
        one_hot_transformer = joblib.load('one_hot_columns.joblib')
        
        if isinstance(one_hot_transformer, list):
            si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
            for col_name in si_columnas_one_hot:
                valor_esperado = col_name.replace('Felder_', '')
                df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
        else:
            # Fallback en caso de que sea un transformador de sklearn u otro objeto
            df_encoded = pd.get_dummies(df_procesado[['Felder']])
            df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
            si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]
            
        # Eliminar la variable original Felder
        df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
        
        # Asegurar que existan todas las columnas que el modelo espera
        if isinstance(one_hot_transformer, list):
            for col in si_columnas_one_hot:
                if col not in df_procesado.columns:
                    df_procesado[col] = 0
                    
        # 3. Normalizar la variable Examen_admisión con 'min_max_scaler.joblib'
        scaler = joblib.load('min_max_scaler.joblib')
        df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
        df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
        
        # Reordenar las columnas conforme lo espera el modelo
        columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
        df_procesado = df_procesado[columnas_ordenadas]
        
        st.subheader("Datos Procesados para el Modelo")
        st.dataframe(df_procesado)
        
        # 4. Predicción con 'bagging_model.joblib'
        model = joblib.load('bagging_model.joblib')
        prediccion = model.predict(df_procesado)
        
        st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
        
    except Exception as e:
        st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")
