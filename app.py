import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Secadero IA - SP Recomendada", layout="centered")
st.title("Recomendador Inteligente de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Carga de datos
    df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Definición de columnas
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

    # Verificar existencia de columnas
    required = entrada_cols + humedad_cols + target_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan las siguientes columnas en el Excel:")
        for col in missing:
            st.write(f"- {col}")
        st.stop()

    # Eliminar filas incompletas en entradas y targets
    df = df.dropna(subset=entrada_cols + humedad_cols + target_cols)

    # Calcular diferencias de humedad
    for i in range(1, 11):
        hist_col = f"Humedad historica piso {i}"
        if hist_col in df.columns:
            df[f"diff_humedad_{i}"] = df[f"Humedad piso {i}"] - df[hist_col]
        else:
            df[f"diff_humedad_{i}"] = 0.0  # fallback si no existe histórica

    diff_cols = [f"diff_humedad_{i}" for i in range(1, 11)]
    feature_cols = [c for c in entrada_cols if c not in target_cols] + diff_cols

    # Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo de placa"] = le.fit_transform(df["Tipo de placa"].astype(str))

    # Preparar X e Y
    X = df[feature_cols]
    Y = df[target_cols]

    # Entrenar modelo suave
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    )
    model.fit(X, Y)

    # Máximos históricos de SP
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # Medias históricas de humedad (para explicación)
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() if f"Humedad historica piso {i}" in df.columns else 0.0
                  for i in range(1, 11)}

    return model, le, max_sp, hist_means, feature_cols, humedad_cols

model, le, max_sp, hist_means, feature_cols, humedad_cols = load_and_train()

st.header("Introduce los datos actuales")

# Recogida de inputs
input_data = {}

# Tipo de placa
placa = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = placa

# Resto de entradas
for col in feature_cols:
    if col == "Tipo de placa" or col.startswith("diff_humedad_"):
        continue
    input_data[col] = st.number_input(col, format="%.2f", step=0.1)

# Humedades actuales
humedades = []
for i in range(1, 11):
    h = st.number_input(f"Humedad piso {i}", format="%.2f", step=0.1)
    input_data[f"Humedad piso {i}"] = h
    humedades.append(h)

if st.button("Calcular Temperaturas SP Recomendadas"):
    # Construir DataFrame de entrada
    df_in = pd.DataFrame([input_data])
    df_in["Tipo de placa"] = le.transform(df_in["Tipo de placa"])
    # Añadir diffs
    for i in range(1, 11):
        df_in[f"diff_humedad_{i}"] = humedades[i-1] - hist_means[i]

    # Seleccionar columnas en orden
    df_in = df_in[feature_cols]

    # Predecir
    preds = model.predict(df_in)[0]

    # Ajuste inverso y límite
    diff_mean = np.mean([df_in[f"diff_humedad_{i}"].iloc[0] for i in range(1, 11)])
    ajuste = diff_mean * 1.5
    resultados = {
        i: min(int(round(preds[i-1] - ajuste)), max_sp[i]) for i in range(1, 4)
    }

    # Mostrar resultados y explicación
    st.subheader("🔧 SP Recomendadas por Zona")
    for i in range(1, 4):
        st.write(f"Zona {i}: {resultados[i]} °C")
    st.info(
        "La recomendación ajusta las temperaturas SP en función de la desviación media de humedad "
        "respecto al histórico, elevando SP si la humedad es alta y reduciéndola si es baja, "
        "sin superar los valores máximos históricos."
    )

