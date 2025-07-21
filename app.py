import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx", alpha=5.0):
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # Columnas
    hist_cols = [f"Humedad historica piso {i}" for i in range(1,11)]
    actual_cols = [f"Humedad piso {i}" for i in range(1,11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1,4)]
    other_feats = [
        "Tipo de placa","Peso Húmedo","Agua ","Yeso","Agua Evaporada",
        "Temperatura entrega 1","Temperatura entrega 2","Temperatura entrega 3",
        "Temperatura retorno 1","Temperatura retorno 2","Temperatura retorno 3",
        "Velocidad línea"
    ]

    # Drop filas incompletas en estas columnas
    df = df.dropna(subset=hist_cols + actual_cols + target_cols + other_feats)

    # Codificar placa
    le = LabelEncoder()
    df['Tipo de placa'] = le.fit_transform(df['Tipo de placa'].astype(str))

    # Calcular diffs y escalar
    for i in range(1,11):
        diff = df[f"Humedad piso {i}"] - df[f"Humedad historica piso {i}"]
        df[f"Diff piso {i}"] = diff * alpha

    # Features finales: otros + diffs
    feature_cols = other_feats + [f"Diff piso {i}" for i in range(1,11)]
    X = df[feature_cols]
    Y = df[target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1,4)}
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() for i in range(1,11)}

    return model, le, max_sp, hist_means, feature_cols

model, le, max_sp, hist_means, feat_cols = load_and_train()

st.title("🔧 Secadero IA - SP Recomendada con Humedad Potenciada")

# Inputs
tipo = st.selectbox("Tipo de placa", le.classes_)
code = le.transform([tipo])[0]
peso = st.number_input("Peso Húmedo", min_value=0.0)
agua = st.number_input("Agua", min_value=0.0)
yeso = st.number_input("Yeso", min_value=0.0)
evap = st.number_input("Agua Evaporada", min_value=0.0)
entrega = [st.number_input(f"Temp entrega {i}", format="%.1f") for i in (1,2,3)]
retorno = [st.number_input(f"Temp retorno {i}", format="%.1f") for i in (1,2,3)]
vel = st.number_input("Velocidad línea", format="%.1f")
hum = [st.number_input(f"Humedad piso {i}", format="%.3f") for i in range(1,11)]

if st.button("Calcular SP recomendada"):
    # Montar DataFrame de entrada
    data = {
        "Tipo de placa": code,
        "Peso Húmedo": peso,
        "Agua ": agua,
        "Yeso": yeso,
        "Agua Evaporada": evap,
        "Temperatura entrega 1": entrega[0],
        "Temperatura entrega 2": entrega[1],
        "Temperatura entrega 3": entrega[2],
        "Temperatura retorno 1": retorno[0],
        "Temperatura retorno 2": retorno[1],
        "Temperatura retorno 3": retorno[2],
        "Velocidad línea": vel
    }
    # Añadir diffs
    for i in range(1,11):
        diff = hum[i-1] - hist_means[i]
        data[f"Diff piso {i}"] = diff * 5.0  # mismo alpha

    df_in = pd.DataFrame([data])[feat_cols]
    preds = model.predict(df_in)[0]

    # Limitar al máximo histórico
    recs = {i: min(int(round(preds[i-1])), max_sp[i]) for i in range(1,4)}

    st.subheader("🌡️ SP recomendadas")
    for i in recs:
        st.metric(f"Zona {i}", recs[i])

    st.markdown("---")
    st.write("**Cómo afecta la humedad:**")
    for i in range(1,4):
        # Muestreamos la media de diffs para explicar
        diff_med = np.mean([hum[j-1]-hist_means[j] for j in range(1,11)])
        st.write(f"- Zona {i}: Ajuste basado en un error medio de humedad de {diff_med:.2f} pts.")

