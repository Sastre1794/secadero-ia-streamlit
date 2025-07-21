import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de la página\ st.set_page_config(page_title="Secadero IA - SP Recomendada", layout="centered")
st.title("Recomendador de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Cargar datos
    df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Definir columnas
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

    # Verificar columnas requeridas
    required = entrada_cols + humedad_cols + hist_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan las siguientes columnas en el Excel:")
        for c in missing:
            st.write(f"- {c}")
        st.stop()

    # Eliminar filas incompletas
    df = df.dropna(subset=entrada_cols + humedad_cols + target_cols)

    # Calcular diferencias de humedad
    diff_cols = []
    for i in range(1, 11):
        col_act = f"Humedad piso {i}"
        col_hist = f"Humedad historica piso {i}"
        df[f"diff_humedad_{i}"] = df[col_act] - df[col_hist]
        diff_cols.append(f"diff_humedad_{i}")

    # Features y target
    feature_cols = entrada_cols + diff_cols
    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    # Codificar tipo de placa
    le = LabelEncoder()
    X["Tipo de placa"] = le.fit_transform(X["Tipo de placa"].astype(str))

    # Entrenar modelo
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    )
    model.fit(X, Y)

    # Obtener máximos históricos y medias de humedad histórica
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() for i in range(1, 11)}

    return model, le, max_sp, hist_means, feature_cols, humedad_cols

# Cargar modelo y parámetros\ nmodel, le, max_sp, hist_means, feature_cols, humedad_cols = load_and_train()

# Interfaz de usuario\ nst.header("Introduce los datos actuales del secadero")

# Inputs\ ninput_data = {}
# Tipo de placa\ nplaca = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = placa

# Datos adicionales\ nfor col in entrada_cols:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, format="%.2f", step=0.1)

# Humedades actuales\ nhumedades = []
for i in range(1, 11):
    h = st.number_input(f"Humedad piso {i}", format="%.2f", step=0.1)
    input_data[f"Humedad piso {i}"] = h
    humedades.append(h)

if st.button("Calcular SP Recomendadas"):
    # Construir DataFrame de entrada\ n    df_in = pd.DataFrame([input_data])
    df_in["Tipo de placa"] = le.transform(df_in["Tipo de placa"])

    # Añadir diffs\ n    for i in range(1, 11):
        df_in[f"diff_humedad_{i}"] = humedades[i-1] - hist_means[i]

    # Seleccionar columnas y predecir\ n    df_in = df_in[feature_cols]
    preds = model.predict(df_in)[0]

    # Mostrar resultados\ n    st.subheader("Resultados SP Recomendadas y Diferencias")
    for i in range(1, 4):
        sp_act = input_data[f"SP_actual zona {i}"]
        sp_rec = min(int(round(preds[i-1])), max_sp[i])
        diff = sp_rec - sp_act
        sign = "+" if diff >= 0 else ""
        st.write(f"Zona {i}: Recomendada {sp_rec} °C ({sign}{diff}°C respecto actual {sp_act}°C)")

    st.info(
        "La SP recomendada ajusta según la desviación de humedad vs histórica, "
        "limitada a los máximos históricos y mostrando la variación respecto al valor actual."
    )
```
