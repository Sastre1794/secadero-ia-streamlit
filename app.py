import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Secadero IA - Recomendador SP Por Héctor Sastre", layout="centered")

st.title("Recomendador Inteligente de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    df = pd.read_excel("/mnt/data/Secadero 1 automático.xlsx")

    columnas_entrada = [
        "Tipo de placa", "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea",
        "Humedad piso 1", "Humedad piso 2", "Humedad piso 3", "Humedad piso 4", "Humedad piso 5",
        "Humedad piso 6", "Humedad piso 7", "Humedad piso 8", "Humedad piso 9", "Humedad piso 10"
    ]

    target_cols = ["SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3"]
    hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]

    other_feats = [col for col in columnas_entrada if col not in target_cols and not col.startswith("Humedad piso")]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]

    all_required = other_feats + humedad_cols + hist_cols + target_cols
    df = df.dropna(subset=all_required)

    X = df[other_feats + humedad_cols]
    Y = df[target_cols]

    le = LabelEncoder()
    X.loc[:, "Tipo de placa"] = le.fit_transform(X["Tipo de placa"])

    hist_means = df[hist_cols].mean()
    max_sp = {i: int(df[f"SP_actual zona {i}"].max()) for i in range(1,4)}

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    return model, le, max_sp, hist_means, other_feats, humedad_cols

model, le, max_sp, hist_means, other_feats, humedad_cols = load_and_train()

st.header("Introduce los datos actuales")

input_data = {}

# Tipo placa
placa = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = le.transform([placa])[0]

# Datos numéricos
for col in other_feats:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, step=1.0, format="%.2f")

# Humedades actuales
for col in humedad_cols:
    input_data[col] = st.number_input(col, step=0.1, format="%.2f")

if st.button("Calcular Temperaturas SP Recomendadas"):
    df_in = pd.DataFrame([input_data])

    # Calcular diferencia humedad
    humedad_actual = df_in[humedad_cols].values.flatten()
    humedad_objetivo = hist_means.values.flatten()
    dif_humedad = (humedad_actual - humedad_objetivo).mean()

    preds = model.predict(df_in)[0]

    ajuste = dif_humedad * 1.5  # más suave
    preds = [min(int(pred - ajuste), max_sp[i+1]) for i, pred in enumerate(preds)]

    st.subheader("Temperaturas SP Recomendadas")
    for i, sp in enumerate(preds, start=1):
        st.write(f"Zona {i}: {sp} ºC")

    st.info("""
    Explicación: La recomendación ajusta ligeramente las temperaturas según la diferencia promedio de humedad respecto a los históricos, asegurando no superar los máximos históricos.
    """)
