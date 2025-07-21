import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

@st.cache_data

def load_and_train():
    df = pd.read_excel('Secadero 1 automático.xlsx')

    # Columnas
    hum_actual = [f'Humedad piso {i}' for i in range(1, 11)]
    hum_hist = [f'Humedad historica piso {i}' for i in range(1, 11)]
    target_cols = [f'SP_actual zona {i}' for i in range(1, 4)]
    temp_entrega = [f'Temperatura entrega {i}' for i in range(1, 4)]
    temp_retorno = [f'Temperatura retorno {i}' for i in range(1, 4)]
    otras = ['Tipo de placa', 'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada', 'Velocidad línea']

    df = df.dropna(subset=hum_actual + hum_hist + target_cols + temp_entrega + temp_retorno + otras)

    hist_means = df[hum_hist].mean().values

    le = LabelEncoder()
    df['Tipo_encoded'] = le.fit_transform(df['Tipo de placa'])

    X = df[otras + temp_entrega + temp_retorno + hum_actual + hum_hist].copy()
    X['Tipo de placa'] = df['Tipo_encoded']

    Y = df[target_cols]

    model = MultiOutputRegressor(RandomForestRegressor(random_state=42, n_estimators=200))
    model.fit(X, Y)

    max_sp = {i: int(df[f'SP_actual zona {i}'].max()) for i in range(1, 4)}

    return model, le, max_sp, hist_means

model, le, max_sp, hist_means = load_and_train()

st.title("Recomendador de Temperaturas Secadero Por Héctor Sastre")

placa = st.selectbox("Tipo de placa", le.classes_)
peso = st.number_input("Peso Húmedo", min_value=0.000)
agua = st.number_input("Agua", min_value=0)
yeso = st.number_input("Yeso", min_value=0)
evaporada = st.number_input("Agua Evaporada", min_value=0.000)
velocidad = st.number_input("Velocidad Línea", min_value=0)

entregas = [st.number_input(f"Temperatura entrega {i}", min_value=0) for i in range(1, 4)]
retornos = [st.number_input(f"Temperatura retorno {i}", min_value=0) for i in range(1, 4)]
humedades = [st.number_input(f"Humedad piso {i}", min_value=0.0) for i in range(1, 11)]

if st.button("Calcular Temperaturas Recomendadas"):
    hist_media = hist_means
    placa_encoded = le.transform([placa])[0]
    
    input_data = [[placa_encoded, peso, agua, yeso, evaporada, velocidad] + entregas + retornos + humedades + hist_media.tolist()]
    df_input = pd.DataFrame(input_data, columns=['Tipo de placa','Peso Húmedo','Agua ','Yeso','Agua Evaporada','Velocidad línea'] + \
                                        [f'Temperatura entrega {i}' for i in range(1, 4)] + \
                                        [f'Temperatura retorno {i}' for i in range(1, 4)] + \
                                        [f'Humedad piso {i}' for i in range(1, 11)] + \
                                        [f'Humedad historica piso {i}' for i in range(1, 11)])
    
    preds = model.predict(df_input)[0]
    preds = [min(round(p), max_sp[i+1]) for i, p in enumerate(preds)]
    
    st.subheader("Temperaturas SP Recomendadas")
    for i, sp in enumerate(preds, 1):
        diferencia = np.mean([humedades[j] - hist_media[j] for j in range(10)])
        if diferencia > 0.5:
            explicacion = "Recomendamos subir ligeramente la temperatura para acelerar el secado, ya que la humedad actual es alta."
        elif diferencia < -0.5:
            explicacion = "Recomendamos bajar la temperatura para evitar sobresecar, ya que la humedad actual es baja."
        else:
            explicacion = "La humedad actual es similar a la histórica, mantenemos temperaturas optimizadas."
        st.write(f"Zona {i}: {sp} ºC. {explicacion}")
