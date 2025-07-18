import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

# —————————————————————————————
# 1) Carga y entrenamiento del modelo
# —————————————————————————————
@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # Leemos el histórico desde la hoja limpia
    df = pd.read_excel(path, sheet_name="Sheet1")
    # Limpiamos espacios en nombres
    df.columns = df.columns.str.strip()

    # Codificamos la variable categórica
    le = LabelEncoder()
    df["Tipo_Placa_Code"] = le.fit_transform(df["Tipo de placa"])

    # Definimos columnas de entrada (features)
    feature_cols = [
        "Tipo_Placa_Code",
        "Peso Húmedo",
        "Agua",
        "Yeso",
        "Agua Evaporada",
        "Velocidad línea",
    ]
    feature_cols += [f"Humedad piso {i}" for i in range(1, 11)]
    feature_cols += [f"Temperatura SP zona {i}" for i in range(1, 4)]
    feature_cols += [f"Temperatura de entrega {i}" for i in range(1, 4)]
    feature_cols += [f"Temperatura de retorno {i}" for i in range(1, 4)]

    # Definimos columnas objetivo (SP recomendada)
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
    max_sp = {i: int(df_clean[f"Temperatura SP zona {i}"].max()) for i in range(1, 4)}

    return model, le, max_sp

multi_model, le, max_sp = load_and_train()

# —————————————————————————————
# 2) Interfaz de usuario en Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador de Temperatura SP en Secadero")
st.write("Ingresa los datos del día para obtener las **Temperaturas SP** recomendadas para cada zona.")

# Disposición en columnas
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
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")
           for i in range(1, 6)}

with col3:
    for i in range(6, 11):
        hum[i] = st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")

st.markdown("### Temperaturas actuales y de sistema")
tsc = {}
tec = {}
trl = {}
for i in range(1, 4):
    tsc[i] = st.number_input(f"Temperatura SP zona {i}", min_value=0.0, format="%.1f")
    tec[i] = st.number_input(f"Temperatura de entrega {i}", min_value=0.0, format="%.1f")
    trl[i] = st.number_input(f"Temperatura de retorno {i}", min_value=0.0, format="%.1f")

# Botón de cálculo
if st.button("Calcular SP recomendada"):
    # Preparamos el dataframe de entrada
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

    # Redondeamos al entero y capamos por el histórico
    recs = {}
    for idx, pred in enumerate(preds, start=1):
        val = int(round(pred))
        recs[idx] = min(val, max_sp[idx])

    # Mostramos resultados
    st.subheader("🌡️ Temperaturas SP recomendadas")
    st.metric("Zona 1 (°C)", recs[1])
    st.metric("Zona 2 (°C)", recs[2])
    st.metric("Zona 3 (°C)", recs[3])

    st.write("---")
    st.write("**Detalle de valores (enteros y capados):**")
    for i in range(1, 4):
        st.write(f"- Zona {i}: {recs[i]}°C (máx. hist.: {max_sp[i]}°C)")

