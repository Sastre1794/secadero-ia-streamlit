import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # 1) Carga de datos
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # 2) Columnas a usar
    feature_cols = [
        "Tipo de placa",
        "Velocidad línea",
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + [
        f"Humedad historica piso {i}" for i in range(1, 11)
    ] + [
        f"Temperatura entrega {i}" for i in range(1, 4)
    ] + [
        f"Temperatura retorno {i}" for i in range(1, 4)
    ]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    # 3) Validación columnas
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en Excel: {missing}")

    # 4) Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo_Code"] = le.fit_transform(df["Tipo de placa"])

    # 5) Construir X, Y
    X = df[["Tipo_Code"] + [c for c in feature_cols if c != "Tipo de placa"]]
    Y = df[target_cols]

    # 6) Filtrar NaN
    data = pd.concat([X, Y], axis=1).dropna()
    X_clean = data[X.columns]
    Y_clean = data[Y.columns]

    # 7) Entrenar modelo multisalida
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42)
    )
    model.fit(X_clean, Y_clean)

    # 8) Máximos históricos de SP
    max_sp = {
        i: int(Y_clean[f"SP_actual zona {i}"].max())
        for i in range(1, 4)
    }
    return model, le, max_sp

model, le, max_sp = load_and_train()

# ---- Streamlit UI ----
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador SP en Secadero")

# Inputs
col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo de placa", le.classes_)
    code = le.transform([tipo])[0]
    velocidad = st.number_input("Velocidad línea (m/min)", min_value=0.0, format="%.1f")
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(1, 6)}
with col2:
    hum.update({i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(6, 11)})
    # Columnas de humedad histórica
    hist = {i: st.number_input(f"Histórica piso {i}", min_value=0.0, format="%.3f") for i in range(1, 11)}

st.markdown("### Temperaturas de sistema")
entrega = {i: st.number_input(f"Temperatura entrega {i}", min_value=0.0, format="%.1f") for i in range(1, 4)}
retorno = {i: st.number_input(f"Temperatura retorno {i}", min_value=0.0, format="%.1f") for i in range(1, 4)}

if st.button("Calcular SP recomendada"):
    # Preparar entrada
    data = {
        "Tipo_Code": code,
        "Velocidad línea": velocidad,
    }
    for i in range(1, 11):
        data[f"Humedad piso {i}"] = hum[i]
        dat
