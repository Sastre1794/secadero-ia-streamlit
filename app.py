import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

@st.cache_data
def load_and_train():
    df = pd.read_excel('/mnt/data/Secadero 1 automático.xlsx')

    # Definir columnas
    target_cols = [f'SP_actual zona {i}' for i in range(1, 4)]
    humedad_cols = [f'Humedad piso {i}' for i in range(1, 11)]
    humedad_hist_cols = [f'Humedad historica piso {i}' for i in range(1, 11)]

    entrada_cols = [
        'Tipo de placa', 'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada',
        'Temperatura entrega 1', 'Temperatura entrega 2', 'Temperatura entrega 3',
        'Temperatura retorno 1', 'Temperatura retorno 2', 'Temperatura retorno 3',
        'Velocidad línea'
    ] + humedad_cols

    # Guardar medias históricas para memoria
    hist_means = df[humedad_hist_cols].mean()

    # Calcular diferencia de humedades y añadir al DataFrame
    for i in range(1, 11):
        df[f'diff_humedad_piso_{i}'] = df[f'Humedad piso {i}'] - df[f'Humedad historica piso {i}']

    diff_cols = [f'diff_humedad_piso_{i}' for i in range(1, 11)]
    feature_cols = [
        'Tipo de placa', 'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada',
        'Temperatura entrega 1', 'Temperatura entrega 2', 'Temperatura entrega 3',
        'Temperatura retorno 1', 'Temperatura retorno 2', 'Temperatura retorno 3',
        'Velocidad línea'
    ] + diff_cols

    df = df.dropna(subset=feature_cols + target_cols)

    # Codificar placa
    le = LabelEncoder()
    df['Tipo de placa'] = le.fit_transform(df['Tipo de placa'])

    X = df[feature_cols]
    Y = df[target_cols]

    model = MultiOutputRegressor(GradientBoostingRegressor())
    model.fit(X, Y)

    max_sp = {i: int(Y[f'SP_actual zona {i}'].max()) for i in range(1, 4)}

    return model, le, max_sp, hist_means, feature_cols, humedad_cols

model, le, max_sp, hist_means, feature_cols, humedad_cols = load_and_train()

st.title('Predicción Temperaturas SP Secadero')

# Entrada de datos
tipo_placa = st.selectbox('Tipo de placa', le.classes_)
peso_humedo = st.number_input('Peso Húmedo')
ag = st.number_input('Agua ')
yes = st.number_input('Yeso')
agua_evaporada = st.number_input('Agua Evaporada')
temp_entregas = [st.number_input(f'Temperatura entrega {i}') for i in range(1, 4)]
temp_retornos = [st.number_input(f'Temperatura retorno {i}') for i in range(1, 4)]
vel_linea = st.number_input('Velocidad línea')
humedades = [st.number_input(f'Humedad piso {i}') for i in range(1, 11)]

# Calcular diferencias respecto a histórico
diffs = [humedades[i] - hist_means.iloc[i] for i in range(10)]

# Preparar dataframe de entrada
entrada = pd.DataFrame([[
    le.transform([tipo_placa])[0], peso_humedo, ag, yes, agua_evaporada,
    *temp_entregas, *temp_retornos, vel_linea, *diffs
]], columns=feature_cols)

preds = model.predict(entrada)[0]
preds_final = [min(round(pred), max_sp[i+1]) for i, pred in enumerate(preds)]

st.subheader('Resultados recomendados de SP por zona:')
for i, sp in enumerate(preds_final):
    st.write(f"Zona {i+1}: {sp} ºC")

st.caption('La predicción ajusta las temperaturas SP buscando reducir la diferencia entre la humedad actual y la histórica, sin superar el máximo SP histórico.')
