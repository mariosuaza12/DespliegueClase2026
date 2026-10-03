import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Configurar la página de Streamlit
st.set_page_config(page_title="Predicción de Nota Final", layout="centered")

st.title("Predicción de Nota Final - Curso")
st.write(
    "Puedes ingresar un estudiante manualmente o cargar un archivo Excel "
    "con varios estudiantes para realizar las predicciones."
)

# Carpeta donde se encuentra este archivo
BASE_DIR = Path(__file__).resolve().parent


# 1. Cargar artefactos necesarios
@st.cache_resource
def load_artifacts():
    try:
        columnas_one_hot = joblib.load(BASE_DIR / "one_hot_columns.joblib")
        scaler = joblib.load(BASE_DIR / "min_max_scaler.joblib")
        model = joblib.load(BASE_DIR / "bagging_optimizado.joblib")

        return columnas_one_hot, scaler, model

    except Exception as e:
        st.error(f"Error al cargar los archivos .joblib: {e}")
        return None, None, None


columnas_one_hot, scaler, model = load_artifacts()


def preparar_datos(df):
    """
    Prepara los datos de entrada utilizando exactamente el mismo
    proceso de One-Hot Encoding y escalado utilizado por el modelo.
    """

    df_input = df.copy()

    # Verificar columnas necesarias
    columnas_requeridas = ["Felder", "Examen_admisión"]
    faltantes = [
        columna for columna in columnas_requeridas
        if columna not in df_input.columns
    ]

    if faltantes:
        raise ValueError(
            "Faltan las siguientes columnas obligatorias: "
            + ", ".join(faltantes)
        )

    # Convertir examen de admisión a número
    df_input["Examen_admisión"] = pd.to_numeric(
        df_input["Examen_admisión"],
        errors="coerce"
    )

    if df_input["Examen_admisión"].isna().any():
        raise ValueError(
            "La columna 'Examen_admisión' contiene valores vacíos "
            "o que no son numéricos."
        )

    # Aplicar One-Hot Encoding manual
    for col in columnas_one_hot:
        if col.startswith("Felder_"):
            categoria = col.replace("Felder_", "")
            df_input[col] = (
                df_input["Felder"] == categoria
            ).astype(float)

    # Aplicar el Min-Max Scaler
    df_input["Examen_admision_scaled"] = scaler.transform(
        df_input[["Examen_admisión"]]
    ).ravel()

    # Seleccionar y ordenar exactamente las columnas que espera el modelo
    columnas_finales = [
        col for col in columnas_one_hot
        if col in df_input.columns
    ]

    return df_input[columnas_finales]


if columnas_one_hot is not None and scaler is not None and model is not None:

    # ---------------------------------------------------------
    # OPCIÓN 1: PREDICCIÓN INDIVIDUAL
    # ---------------------------------------------------------
    st.header("1. Predicción individual")

    categorias_felder = [
        col.replace("Felder_", "")
        for col in columnas_one_hot
        if col.startswith("Felder_")
    ]

    felder_selected = st.selectbox(
        "Estilo de Aprendizaje (Felder)",
        options=categorias_felder
    )

    examen_admision = st.slider(
        "Nota de Examen de Admisión",
        min_value=0.0,
        max_value=5.0,
        value=3.8,
        step=0.05
    )

    if st.button("Calcular Predicción"):
        try:
            df_individual = pd.DataFrame([
                {
                    "Felder": felder_selected,
                    "Examen_admisión": examen_admision
                }
            ])

            df_procesado = preparar_datos(df_individual)
            prediccion = model.predict(df_procesado)[0]

            st.success(f"### Nota Final Estimada: {prediccion:.3f}")

            with st.expander("Ver variables procesadas"):
                st.dataframe(df_procesado)

        except Exception as e:
            st.error(f"Error al realizar la predicción: {e}")

    # ---------------------------------------------------------
    # OPCIÓN 2: PREDICCIÓN DESDE EXCEL
    # ---------------------------------------------------------
    st.divider()
    st.header("2. Predicción mediante archivo Excel")

    st.write(
        "Carga un archivo `.xlsx` que contenga las columnas "
        "`Felder` y `Examen_admisión`."
    )

    st.info(
        "Ejemplo de estructura: una fila por estudiante y las columnas "
        "'Felder' y 'Examen_admisión'. Puedes incluir otras columnas "
        "(por ejemplo, ID o nombre); se conservarán en el resultado."
    )

    archivo_excel = st.file_uploader(
        "Selecciona el archivo Excel con los datos de prueba",
        type=["xlsx", "xls"]
    )

    if archivo_excel is not None:
        try:
            # Leer Excel
            df_excel = pd.read_excel(archivo_excel)

            st.subheader("Datos cargados")
            st.dataframe(df_excel, use_container_width=True)

            columnas_requeridas = ["Felder", "Examen_admisión"]
            faltantes = [
                columna
                for columna in columnas_requeridas
                if columna not in df_excel.columns
            ]

            if faltantes:
                st.error(
                    "El archivo no tiene las columnas obligatorias: "
                    + ", ".join(faltantes)
                )

                st.write("Columnas encontradas:")
                st.code(", ".join(df_excel.columns.astype(str)))

            elif df_excel.empty:
                st.warning("El archivo Excel no contiene registros.")

            else:
                if st.button("Procesar Excel y calcular predicciones"):
                    try:
                        # Preparar datos para el modelo
                        df_procesado = preparar_datos(df_excel)

                        # Realizar predicciones
                        predicciones = model.predict(df_procesado)

                        # Crear resultado conservando las columnas originales
                        df_resultado = df_excel.copy()
                        df_resultado["Prediccion_Nota_Final"] = predicciones

                        st.success(
                            f"Se realizaron {len(df_resultado)} predicciones correctamente."
                        )

                        st.subheader("Resultados")
                        st.dataframe(
                            df_resultado,
                            use_container_width=True
                        )

                        # Crear Excel para descargar
                        output = pd.io.common.BytesIO()

                        with pd.ExcelWriter(
                            output,
                            engine="openpyxl"
                        ) as writer:
                            df_resultado.to_excel(
                                writer,
                                index=False,
                                sheet_name="Predicciones"
                            )

                        output.seek(0)

                        st.download_button(
                            label="Descargar resultados en Excel",
                            data=output,
                            file_name="predicciones_resultado.xlsx",
                            mime=(
                                "application/vnd.openxmlformats-officedocument."
                                "spreadsheetml.sheet"
                            )
                        )

                        with st.expander(
                            "Ver variables procesadas enviadas al modelo"
                        ):
                            st.dataframe(
                                df_procesado,
                                use_container_width=True
                            )

                    except Exception as e:
                        st.error(
                            f"Error al procesar el archivo Excel: {e}"
                        )

else:
    st.warning(
        "Por favor, asegúrate de que los archivos "
        "'one_hot_columns.joblib', 'min_max_scaler.joblib' y "
        "'bagging_optimizado.joblib' se encuentren junto a app.py."
    )
