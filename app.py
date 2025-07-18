import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# —————————————————————————————
# 1) Carga y entrenamiento del modelo
# —————————————————————————————
@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # Carga y limpieza de columnas
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # Definición de columnas
    feature_cols = [
        "Tipo de placa",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + \
      [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = ["SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3"]

    # Verificación de columnas presentes
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        st.error("Faltan columnas en el Excel:")
        for c in missing:
            st.write(f"- {c}")
        st.stop()

    # Convertir todas las columnas numéricas a float, forzando errores a NaN
    num_cols = [c for c in feature_cols + target_cols if c != "Tipo de placa"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Codificar la variable categórica
    le = LabelEncoder()
    df["Tipo_Code"] = le.fit_transform(df["Tipo de placa"])

    # Preparar X e Y
    X = df[["Tipo_Code"] + [c for c in feature_cols if c != "Tipo de placa"]]
    Y = df[target_cols]

    # Eliminar filas con datos faltantes
    df_clean = pd.concat([X, Y], axis=1).dropna()
    X_clean = df_clean[X.columns]
    Y_clean = df_clean[Y.columns]

    # Entrenar modelo multisalida
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_clean, Y_clean)

    # Máximos históricos de SP para capar predicciones
    max_sp = {
        i: int(Y_clean[f"SP_actual zona {i}"].max())
        for i in range(1, 4)
    }

    return model, le, max_sp

# Cargamos modelo, encoder y máximos
model, le, max_sp = load_and_train()

# —————————————————————————————
# 2) Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.titl

