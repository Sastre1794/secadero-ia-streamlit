import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# —————————————————————————————
# 1) Cargar y entrenar modelo en caliente
# —————————————————————————————
@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # Leemos el histórico limpio
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # Codificamos el tipo de placa
    le = LabelEncoder()
    df["Tipo_Placa_Code"] = le.fit_transform(df["Tipo de placa"])

    # Definimos columnas de entrada
    feature_cols = [
        "Tipo_Placa_Code",
        "Peso Húmedo",
        "Agua",
        "Yeso",
        "Agua Evaporada",
        "Velocidad línea",
    ]
    # Humedades pisos 1–10
    feature_cols += [f"Humedad piso {i}" for i in range(1, 11)]
    # Temperaturas SP actuales zonas 1–3
    feature_cols += [f"Temperatura SP zona {i}" for i in range(1, 4)]
    # Temperatura de entrega 1–3 y retorno 1–3
    feature_cols += [f"Temperatura de entrega {i}" for i in range(1, 4)]
    feature_cols += [f"Temperatura de retorno {i}" for i in range(1, 4)]

    # Definimos las columnas a predecir (targets)
    target_cols = [f"Temperatura SP zona {i}" for i in range(1, 4)]

    # Eliminamos filas con datos faltantes
    df_clean = df.dropna(subset=feature_cols + target_cols)

    X = df_clean[feature_cols]
    y = df_clean[target_cols]

    # Entrenamos un RandomForest multisalida
    base = RandomForestRegressor(n_estimators=100, random_state=42)
    model = MultiOutputRegressor(base)
    model.fit(X, y)

    # Calculamos máximos históricos de SP por zona
    max_sp = {
        i: int(df_clean[f"Temperatura SP zona {i}"].max())
        for i in range(1, 4)
    }

    return model, le, max_sp

multi_model, le, max_sp = load_and_train()

# —————————————————————————————
# 2) Interfaz de usuario
# —————————————————————————————
st.set_page_config(page_title="Recomendador de SP Secadero", layout="wide")
st.title("🔧 Recomendador de Temperatura SP en Secadero")
st.write("Introduce los datos del día para obtener las **Temperaturas SP** idealmente ajustadas.")

# Disposición en tres columnas
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
    st.markdown("**Humedad superficial (unidades equipo)**")
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")
           for i in range(1, 6)}

with col3:
    for i in range(6, 11):
        hum[i] = st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")

# SP actuales, entrega y retorno
st.markdown("### Temperaturas actuales y de sistema")
tsc = {}
tec = {}
trl = {}
for i in range(1, 4):
    tsc[i] = st.number_input(f"Temperatura SP zona {i}", min_value=0.0, format="%.1f")
    tec[i] = st.number_input(f"Temperatura de entrega {i}", min_value=0.0, format="%.1f")
    trl[i] = st.number_input(f"Temperatura de retorno {i}", min_value=0.0, format="%.1f")

# Botón de predicción
if st.button("Calcular SP recomendada"):
    # Construimos el DataFrame de entrada
    data = {
        "Tipo_Placa_Code": code,
        "Peso Húmedo": peso,
        "Agua": agua,
        "Yeso": yeso,
        "Agua Evaporada": evap,
        "Velocidad línea": velo,
    }
    for i in range(1, 11):
        data[f"Humedad piso {i}"] = hum[i]
    for i in range(1, 4):
        data[f"Temperatura SP zona {i}"] = tsc[i]
        data[f"Temperatura de entrega {i}"] = tec[i]
        data[f"Temperatura de retorno {i}"] = trl[i]

    df_in = pd.DataFrame([data])
    preds = multi_model.predict(df_in)[0]

    # Redondeamos, convertimos a entero y capamos por el máximo histórico
    recs = {}
    for i, pred in enumerate(preds, start=1):
        val = int(round(pred))
        recs[i] = min(val, max_sp[i])

    # Mostramos resultados
    st.subheader("🌡️ Temperaturas SP recomendadas")
    st.metric("Zona 1 (°C)", recs[1])
    st.metric("Zona 2 (°C)", recs[2])
    st.metric("Zona 3 (°C)", recs[3])

    st.write("---")
    st.write("**Valores finales (enteros y capados a histórico máximo):**")
    st.write(f"- Zona 1: {recs[1]}°C (máx. hist.: {max_sp[1]}°C)")
    st.write(f"- Zona 2: {recs[2]}°C (máx. hist.: {max_sp[2]}°C)")
    st.write(f"- Zona 3: {recs[3]}°C (máx. hist.: {max_sp[3]}°C)")

