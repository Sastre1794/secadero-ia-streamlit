import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

@st.cache_data
def load_and_train():
    df = pd.read_excel('Secadero 1 automático.xlsx')

    # Definir columnas
    hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]
    actual_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]
    other_feats = [
        "Peso Húmedo", "Agua ", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "Velocidad línea"
    ]

    df = df.dropna(subset=["Tipo de placa"] + hist_cols + actual_cols + target_cols + other_feats)

    # Label encoding para tipo de placa
    le = LabelEncoder()
    df["Tipo de placa"] = le.fit_transform(df["Tipo de placa"])

    X = df[["Tipo de placa"] + other_feats + actual_cols]
    Y = df[target_cols]

    # Guardar máximos históricos
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # Media histórica por piso
    hist_means = df[hist_cols].mean()

    # Modelo más suave
    model = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42))
    model.fit(X, Y)

    return model, le, max_sp, hist_means, X.columns.tolist()

model, le, max_sp, hist_means, feature_cols = load_and_train()

st.title("Recomendador de Temperaturas SP para Secadero Por Héctor Sastre")

placa = st.selectbox("Tipo de placa", le.classes_)

inputs = {}
for col in ["Peso Húmedo", "Agua ", "Yeso", "Agua Evaporada",
             "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
             "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
             "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
             "Velocidad línea"]:
    inputs[col] = st.number_input(col, step=1.0)

humedades = []
for i in range(1, 11):
    humedades.append(st.number_input(f"Humedad piso {i}", step=0.1))

if st.button("Calcular Temperaturas SP Recomendadas"):
    df_in = pd.DataFrame([{**inputs, "Tipo de placa": le.transform([placa])[0], **{f"Humedad piso {i+1}": humedades[i] for i in range(10)}}])

    # Ajuste moderado por diferencia de humedades
    diferencia_humedad = sum([humedades[i] - hist_means.iloc[i] for i in range(10)]) / 10
    ajuste_global = max(min(diferencia_humedad * 2, 10), -10)

    preds = model.predict(df_in)[0]
    recomendaciones = {}
    explicaciones = []

    for i in range(1, 4):
        sp_actual = inputs[f"SP_actual zona {i}"]
        base_pred = preds[i-1] + ajuste_global
        recomendado = min(int(round(base_pred)), max_sp[i])
        recomendaciones[i] = recomendado

        diff = recomendado - sp_actual
        razon = ""
        if diff > 2:
            razon = f"Aumenta +{diff} grados porque la humedad actual es significativamente mayor que la histórica."
        elif diff < -2:
            razon = f"Reduce {abs(diff)} grados porque la humedad actual es bastante inferior a la histórica."
        else:
            razon = "Sin cambios bruscos; la humedad es estable respecto a la histórica."

        explicaciones.append(f"Zona {i}: {razon}")

    st.subheader("Resultados")
    for i in range(1, 4):
        st.write(f"Zona {i} - SP Recomendado: {recomendaciones[i]} °C")
        st.caption(explicaciones[i-1])
