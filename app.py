import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Secadero IA - Recomendador SP", layout="centered")
st.title("Recomendador Inteligente de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Cargar datos
    df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Definir columnas exactas según tu Excel
    entrada_cols = [
        "Tipo de placa", "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    # Verificar columnas
    required = entrada_cols + humedad_cols
    missing = [c for c in required + target_cols if c not in df.columns]
    if missing:
        st.error("Faltan columnas en el Excel:")
        for col in missing:
            st.write(f"- {col}")
        st.stop()

    # Dropna en las columnas de entrada y target
    df = df.dropna(subset=entrada_cols + humedad_cols + target_cols)

    # Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo de placa"] = le.fit_transform(df["Tipo de placa"])

    # Preparar matrices
    X = df[entrada_cols + humedad_cols]
    Y = df[target_cols]

    # Entrenar modelo
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    # Máximos históricos de SP
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # Media histórica de humedades
    hist_means = {i: df[f"Humedad piso {i}"].mean() for i in range(1, 11)}

    return model, le, max_sp, hist_means, entrada_cols, humedad_cols

# Carga modelo
model, le, max_sp, hist_means, entrada_cols, humedad_cols = load_and_train()

# UI
st.header("Introduce los datos actuales")

input_data = {}
# Selector de tipo
placa = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = placa  # mantiene string

# Inputs de entrada
for col in entrada_cols:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, step=1.0, format="%.2f")
# Humedades actuales
for col in humedad_cols:
    input_data[col] = st.number_input(col, step=0.1, format="%.2f")

if st.button("Calcular Temperaturas SP Recomendadas"):
    # Transformar input
    df_in = pd.DataFrame([input_data])
    df_in["Tipo de placa"] = le.transform(df_in["Tipo de placa"])
    df_in = df_in[entrada_cols + humedad_cols]

    # Diferencia humedad media
    hum_act = df_in[humedad_cols].values.flatten()
    hum_obj = np.array([hist_means[i] for i in range(1, 11)])
    diff_mean = (hum_act - hum_obj).mean()

    preds = model.predict(df_in)[0]
    ajuste = diff_mean * 1.5  # peso suave
    resultados = {
        i: min(int(round(preds[i-1] - ajuste)), max_sp[i])
        for i in range(1, 4)
    }

    # Mostrar resultados
    st.subheader("Temperaturas SP Recomendadas")
    for zona, sp in resultados.items():
        st.write(f"Zona {zona}: {sp} °C")
    st.info(
        "Las recomendaciones ajustan la SP según la media de desviación de humedad, "
        "sin superar los máximos históricos.")
