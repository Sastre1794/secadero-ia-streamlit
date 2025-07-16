import streamlit as st
import pandas as pd
import joblib

# —————————————————————————
# 1) Carga del modelo y del encoder
# —————————————————————————
multi_model = joblib.load('multi_model_secadero.pkl')
le = joblib.load('label_encoder_secadero.pkl')

# —————————————————————————
# 2) Título y descripción
# —————————————————————————
st.set_page_config(page_title="Recomendador de Temperaturas Secadero", layout="wide")
st.title("🔧 Recomendador de Temperaturas en Secadero")
st.write("""
Introduce los parámetros del día para obtener las **temperaturas recomendadas** 
en las zonas 1, 2 y 3, con el fin de acercar la humedad superficial a su valor histórico óptimo.
""")

# —————————————————————————
# 3) Entradas del usuario
# —————————————————————————
# Codificamos el tipo de placa mediante el encoder
tipo_placa = st.selectbox("Tipo de placa", le.classes_)
placa_code = le.transform([tipo_placa])[0]

col1, col2, col3 = st.columns(3)

with col1:
    peso_humedo = st.number_input("Peso húmedo (kg/m²)", min_value=0.0, format="%.3f")
    agua = st.number_input("Cantidad de agua (l/m²)", min_value=0.0, format="%.3f")
    yeso = st.number_input("Cantidad de yeso (kg/m²)", min_value=0.0, format="%.3f")
    agua_evap = st.number_input("Agua evaporada (l/m²)", min_value=0.0, format="%.3f")

with col2:
    velocidad = st.number_input("Velocidad línea (m/min)", min_value=0.0, format="%.3f")
    st.markdown("**Humedad superficial (unidades equipo)**")
    humedades = {}
    for i in range(1, 6):
        humedades[f"piso_{i}"] = st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")
with col3:
    for i in range(6, 11):
        humedades[f"piso_{i}"] = st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")

# —————————————————————————
# 4) Predicción al pulsar botón
# —————————————————————————
if st.button("Calcular temperaturas"):
    # Preparar DataFrame de entrada con el orden correcto
    entrada = pd.DataFrame([{
        'Tipo_Placa_Code': placa_code,
        'Peso Húmedo': peso_humedo,
        'Agua': agua,
        'Yeso': yeso,
        'Agua Evaporada': agua_evap,
        'Velocidad línea': velocidad,
        **{f'Humedad piso {i}': humedades[f'piso_{i}'] for i in range(1, 11)}
    }])

    # Predecir
    z1, z2, z3 = multi_model.predict(entrada)[0]

    # Mostrar resultados
    st.subheader("🌡️ Temperaturas recomendadas")
    st.metric("Zona 1 (°C)", f"{z1:.1f}")
    st.metric("Zona 2 (°C)", f"{z2:.1f}")
    st.metric("Zona 3 (°C)", f"{z3:.1f}")

    # Sugerencias adicionales
    st.write("---")
    st.write("**Recomendaciones rápidas:**")
    if z1 > entrada['Temperatura zona 1'].iloc[0] if 'Temperatura zona 1' in entrada.columns else False:
        st.write("- Aumentar temperatura en Zona 1.")
    if z2 > entrada.get('Temperatura zona 2', [0])[0]:
        st.write("- Aumentar temperatura en Zona 2.")
    if z3 > entrada.get('Temperatura zona 3', [0])[0]:
        st.write("- Aumentar temperatura en Zona 3.")
