import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de la página
st.set_page_config(page_title="Secadero IA - SP Recomendada", layout="centered")
st.title("Recomendador de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Carga de datos del Excel
    df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Columnas de entrada y target según el Excel
    entrada_cols = [
        "Tipo de placa",
        "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    # Verificar columnas
    required = entrada_cols + humedad_cols + hist_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan las siguientes columnas en el Excel:")
        for col in missing:
            st.write(f"- {col}")
        st.stop()

    # Dropna en columnas críticas
    df = df.dropna(subset=entrada_cols + humedad_cols)

    # Calcular diferencias humedad actual vs histórica
    for i in range(1, 11):
        df[f"diff_humedad_{i}"] = df[f"Humedad piso {i}"] - df[f"Humedad historica piso {i}"]

    diff_cols = [f"diff_humedad_{i}" for i in range(1, 11)]
    feature_cols = entrada_cols + diff_cols

    # Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo de placa"] = le.fit_transform(df["Tipo de placa"].astype(str))

    # Preparar X e Y
    X = df[feature_cols]
    Y = df[target_cols]

    # Entrenar modelo
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    )
    model.fit(X, Y)

    # Máximos históricos de SP
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # Medias de humedad histórica
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() for i in range(1, 11)}

    return model, le, max_sp, hist_means, feature_cols, humedad_cols

# Cargar modelo y parámetros\ nmodel, le, max_sp, hist_means, feature_cols, humedad_cols = load_and_train()

st.header("Introduce los datos actuales")

# Recopilar inputs\ ninput_data = {}

# Tipo de placa\ nplaca = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = placa

# Resto de entradas excepto diffs y hum
for col in entrada_cols:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, format="%.2f", step=0.1)

# Humedades actuales
humedades = []
for i in range(1, 11):
    h = st.number_input(f"Humedad piso {i}", format="%.2f", step=0.1)
    input_data[f"Humedad piso {i}"] = h
    humedades.append(h)

if st.button("Calcular SP Recomendadas"):
    # Preparar DataFrame de entrada\ n    df_in = pd.DataFrame([input_data])

    # Codificar placa\ n    df_in["Tipo de placa"] = le.transform(df_in["Tipo de placa"])

    # Añadir diffs\ n    for i in range(1, 11):
        df_in[f"diff_humedad_{i}"] = humedades[i-1] - hist_means[i]

    # Reordenar columnas\ n    df_in = df_in[feature_cols]

    # Predecir\ n    preds = model.predict(df_in)[0]

    # Mostrar resultados y diferencia vs actual\ n    st.subheader("Resultados SP Recomendadas y Diferencias")
    for idx in range(1, 4):
        sp_act = input_data[f"SP_actual zona {idx}"]
        sp_pred = min(int(round(preds[idx-1])), max_sp[idx])
        diff = sp_pred - sp_act
        sign = "+" if diff >= 0 else ""
        st.write(f"Zona {idx}: Recomendada {sp_pred} °C ({sign}{diff} °C respecto actual {sp_act} °C)")

    st.info(
        "La SP recomendada ajusta según la desviación media de humedad vs histórica, "
        "manteniendo SP dentro de límites históricos y mostrando la variación respecto al valor actual."
    )
