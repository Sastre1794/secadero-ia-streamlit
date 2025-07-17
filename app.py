import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# —————————————————————————————
# 1) Carga y limpieza de datos
# —————————————————————————————
@st.cache_data
def load_and_train(path='Secadero 1 automático.xlsx'):
    # Lee tu histórico
    df = pd.read_excel(path, sheet_name='Sheet1')
    df.columns = df.columns.str.strip()
    # Codifica tipo de placa
    le = LabelEncoder()
    df['Tipo_Placa_Code'] = le.fit_transform(df['Tipo de placa'])
    # Elimina filas con NaN en X o Y
    feature_cols = ['Tipo_Placa_Code', 'Peso Húmedo', 'Agua', 'Yeso', 'Agua Evaporada', 'Velocidad línea'] + \
                   [f'Humedad piso {i}' for i in range(1,11)]
    target_cols = ['Tempretura zona 1', 'Tempretura zona 2', 'Tempretura zona 3']
    df_clean = df.dropna(subset=feature_cols + target_cols)
    X = df_clean[feature_cols]
    Y = df_clean[target_cols]
    # Entrena modelo multisalida
    base = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, Y)
    return model, le

multi_model, le = load_and_train()

# —————————————————————————————
# 2) Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador Secadero", layout="wide")
st.title("🔧 Recomendador de Temperaturas en Secadero")

st.write("Introduce los parámetros del día para calcular las **temperaturas recomendadas** en las zonas 1, 2 y 3.")

# Columnas de inputs
col1, col2, col3 = st.columns(3)

with col1:
    tipo = st.selectbox("Tipo de placa", le.classes_)
    code = le.transform([tipo])[0]
    peso = st.number_input("Peso húmedo (kg/m²)", min_value=0.0, format="%.3f")
    agua = st.number_input("Agua (l/m²)", min_value=0.0, format="%.3f")
    yeso = st.number_input("Yeso (kg/m²)", min_value=0.0, format="%.3f")
    evap = st.number_input("Agua evaporada (l/m²)", min_value=0.0, format="%.3f")

with col2:
    velo = st.number_input("Velocidad línea (m/min)", min_value=0.0, format="%.3f")
    st.markdown("**Humedad superficial (unidad equipo)**")
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(1, 6)}

with col3:
    hum.update({i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(6, 11)})

if st.button("Calcular temperaturas"):
    # Prepara entrada
    data = {
        'Tipo_Placa_Code': code,
        'Peso Húmedo': peso,
        'Agua': agua,
        'Yeso': yeso,
        'Agua Evaporada': evap,
        'Velocidad línea': velo
    }
    for i in range(1, 11):
        data[f'Humedad piso {i}'] = hum[i]
    df_in = pd.DataFrame([data])
    # Predecir
    z1, z2, z3 = multi_model.predict(df_in)[0]
    # Mostrar
    st.subheader("🌡️ Temperaturas recomendadas")
    st.metric("Zona 1 (°C)", f"{z1:.1f}")
    st.metric("Zona 2 (°C)", f"{z2:.1f}")
    st.metric("Zona 3 (°C)", f"{z3:.1f}")
    st.write("---")
    st.write(f"- Zona 1: {z1:.1f}°C")
    st.write(f"- Zona 2: {z2:.1f}°C")
    st.write(f"- Zona 3: {z3:.1f}°C")

