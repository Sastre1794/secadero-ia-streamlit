import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_and_train():
    df = pd.read_excel('Secadero 1 automático.xlsx')

    # Columnas
    target_cols = [f'SP_actual zona {i}' for i in range(1, 4)]
    hist_cols = [f'Humedad historica piso {i}' for i in range(1, 11)]
    pisos_cols = [f'Humedad piso {i}' for i in range(1, 11)]

    feature_cols = [
        'Tipo de placa', 'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada',
        'Temperatura entrega 1', 'Temperatura entrega 2', 'Temperatura entrega 3',
        'Temperatura retorno 1', 'Temperatura retorno 2', 'Temperatura retorno 3',
        'Velocidad línea'
    ] + pisos_cols

    # Preprocesado
    df = df.dropna(subset=feature_cols + target_cols + hist_cols).reset_index(drop=True)

    # Encoding tipo de placa
    le = LabelEncoder()
    df['Tipo de placa'] = le.fit_transform(df['Tipo de placa'].astype(str))

    # Diferencia humedades actuales vs históricas
    for i in range(1, 11):
        df[f'Diff piso {i}'] = df[f'Humedad piso {i}'] - df[f'Humedad historica piso {i}']

    X = df[feature_cols + [f'Diff piso {i}' for i in range(1, 11)]]
    Y = df[target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    # Máximos SP históricos
    max_sp = {i: int(df[f'SP_actual zona {i}'].max()) for i in range(1, 4)}
    hist_means = df[hist_cols].mean().to_dict()

    return model, le, max_sp, hist_means

model, le, max_sp, hist_means = load_and_train()

st.title("Calculadora Temperatura SP Recomendada Secadero")

# Input usuario
placa = st.text_input("Tipo de placa")
peso_humedo = st.number_input("Peso húmedo", min_value=0.0)
agua = st.number_input("Agua", min_value=0.0)
yeso = st.number_input("Yeso", min_value=0.0)
agua_evap = st.number_input("Agua evaporada", min_value=0.0)
entregas = [st.number_input(f"Temperatura entrega {i}", value=0.0) for i in range(1, 4)]
retornos = [st.number_input(f"Temperatura retorno {i}", value=0.0) for i in range(1, 4)]
sp_actual = [st.number_input(f"SP actual zona {i}", value=0.0) for i in range(1, 4)]
vel_linea = st.number_input("Velocidad línea", min_value=0.0)
humedades = [st.number_input(f"Humedad piso {i}", min_value=0.0) for i in range(1, 11)]

if st.button("Calcular SP recomendado"):
    # Encoding tipo de placa
    placa_encoded = le.transform([placa])[0] if placa in le.classes_ else 0

    # Diferencia humedades vs históricas
    diffs = []
    for i in range(1, 11):
        media_hist = hist_means[f'Humedad historica piso {i}']
        diffs.append(humedades[i-1] - media_hist)

    input_data = np.array([[
        placa_encoded, peso_humedo, agua, yeso, agua_evap,
        *entregas, *retornos, vel_linea, *humedades, *diffs
    ]])

    preds = model.predict(input_data)[0]

    # Limitar a máximos históricos sin decimales
    sp_recomendados = [min(int(round(pred)), max_sp[i+1]) for i, pred in enumerate(preds)]

    st.success(f"Temperaturas SP recomendadas (Zona 1-2-3): {sp_recomendados}")

