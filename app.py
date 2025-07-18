import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# —————————————————————————————
# 1) Carga y validación de datos
# —————————————————————————————
@st.cache_data
def load_data(path="Secadero 1 automático.xlsx"):
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Columnas que necesitamos
feature_cols = [
    "Tipo de placa",
    "Peso Húmedo",
    "Agua",
    "Yeso",
    "Agua Evaporada",
    "Velocidad línea"
] + [f"Humedad piso {i}" for i in range(1, 11)] + [
    f"Temperatura SP zona {i}" for i in range(1, 4)
] + [
    f"Temperatura entrega {i}" for i in range(1, 4)
] + [
    f"Temperatura retorno {i}" for i in range(1, 4)
]

target_cols = [f"Temperatura SP zona {i}" for i in range(1, 4)]

# Detectar columnas faltantes
missing = [col for col in feature_cols + target_cols if col not in df.columns]
if missing:
    st.error("Faltan las siguientes columnas en el Excel:")
    for col in missing:
        st.write(f"- {col}")
    st.stop()

# —————————————————————————————
# 2) Entrenamiento del modelo
# —————————————————————————————
# Codificar Tipo de placa
le = LabelEncoder()
df["Tipo_Placa_Code"] = le.fit_transform(df["Tipo de placa"])

# Preparar X e Y
X = df[["Tipo_Placa_Code"] + feature_cols[1:]]  # sustituye la columna categórica por el código
Y = df[target_cols]

# Eliminar filas con NaN
data = pd.concat([X, Y], axis=1).dropna()
X_clean = data[X.columns]
Y_clean = data[Y.columns]

# Entrenar modelo multisalida
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_clean, Y_clean)

# Máximos históricos de SP
max_sp = {
    i: int(X_clean[f"Temperatura SP zona {i}"].max())
    for i in range(1, 4)
}

# —————————————————————————————
# 3) Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador de Temperatura SP en Secadero")
st.write("Introduce los datos del día para obtener las SP recomendadas por zona.")

# Inputs organizados en tres columnas
col1, col2, col3 = st.columns(3)
with col1:
    tipo = st.selectbox("Tipo de placa", le.classes_)
    code = le.transform([tipo])[0]
    peso = st.number_input("Peso húmedo (kg/m²)", min_value=0.0, format="%.3f")
    agua = st.number_input("Agua (l/m²)", min_value=0.0, format="%.3f")
    yeso = st.number_input("Yeso (kg/m²)", min_value=0.0, format="%.3f")
    evap = st.number_input("Agua Evaporada (l/m²)", min_value=0.0, format="%.3f")
with col2:
    velo = st.number_input("Velocidad línea (m/min)", min_value=0.0, format="%.3f")
    st.markdown("**Humedad superficial (unidad equipo)**")
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(1, 6)}
with col3:
    for i in range(6, 11):
        hum[i] = st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f")

st.markdown("### Temperaturas actuales y de sistema")
tsc = {}
tec = {}
trl = {}
for i in range(1, 4):
    tsc[i] = st.number_input(f"Temperatura SP zona {i}", min_value=0.0, format="%.1f")
    tec[i] = st.number_input(f"Temperatura entrega {i}", min_value=0.0, format="%.1f")
    trl[i] = st.number_input(f"Temperatura retorno {i}", min_value=0.0, format="%.1f")

# Botón de cálculo
if st.button("Calcular SP recomendada"):
    # Construir DataFrame de entrada
    entrada = {
        "Tipo_Placa_Code": code,
        "Peso Húmedo": peso,
        "Agua": agua,
        "Yeso": yeso,
        "Agua Evaporada": evap,
        "Velocidad línea": velo
    }
    for i in range(1, 11):
        entrada[f"Humedad piso {i}"] = hum[i]
    for i in range(1, 4):
        entrada[f"Temperatura SP zona {i}"] = tsc[i]
        entrada[f"Temperatura entrega {i}"] = tec[i]
        entrada[f"Temperatura retorno {i}"] = trl[i]

    df_in = pd.DataFrame([entrada])
    preds = model.predict(df_in)[0]

    # Redondear y capar por histórico
    recs = {}
    for idx, pred in enumerate(preds, start=1):
        val = int(round(pred))
        recs[idx] = min(val, max_sp[idx])

    # Mostrar resultados
    st.subheader("🌡️ Temperaturas SP recomendadas")
    st.metric("Zona 1 (°C)", recs[1])
    st.metric("Zona 2 (°C)", recs[2])
    st.metric("Zona 3 (°C)", recs[3])
    st.write("---")
    for i in range(1, 4):
        st.write(f"- Zona {i}: {recs[i]}°C (máx hist: {max_sp[i]}°C)")

