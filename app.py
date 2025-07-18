import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # 1) Carga y limpieza
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # 2) Definir columnas de entrada y target
    feature_cols = [
        "Tipo de placa",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + \
      [f"Humedad historica piso {i}" for i in range(1, 11)]

    target_cols = ["SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3"]

    # 3) Validar que están todas
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    # 4) Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo_Code"] = le.fit_transform(df["Tipo de placa"])

    # 5) Construir X e Y
    X = df[["Tipo_Code"]
           + feature_cols[1:]]  # sustituimos “Tipo de placa” por su código
    Y = df[target_cols]

    # 6) Dropna
    dt = pd.concat([X, Y], axis=1).dropna()
    X_train = dt[X.columns]
    Y_train = dt[Y.columns]

    # 7) Entrenar
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_train, Y_train)

    # 8) Máximos históricos para capar
    max_sp = {i: int(Y_train[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    return model, le, max_sp

model, le, max_sp = load_and_train()

# —————————————————————————————
# Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador de SP Ajustada en Secadero")
st.write("Introduce los datos actuales para obtener las **SP recomendadas** por zona.")

# Input fields en dos columnas
col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo de placa", le.classes_)
    code = le.transform([tipo])[0]
    entrega = {i: st.number_input(f"Temperatura entrega {i}", min_value=0.0, format="%.1f") for i in (1,2,3)}
    retorno = {i: st.number_input(f"Temperatura retorno {i}", min_value=0.0, format="%.1f") for i in (1,2,3)}
    sp_act = {i: st.number_input(f"SP_actual zona {i}", min_value=0.0, format="%.1f") for i in (1,2,3)}
with col2:
    velocidad = st.number_input("Velocidad línea (m/min)", min_value=0.0, format="%.1f")
    hum = {i: st.number_input(f"Humedad piso {i}", min_value=0.0, format="%.3f") for i in range(1, 11)}
    hist = {i: st.number_input(f"Humedad historica piso {i}", min_value=0.0, format="%.3f") for i in range(1, 11)}

if st.button("Calcular SP recomendada"):
    # Construir DataFrame de entrada
    data = {
        "Tipo_Code": code,
        "Temperatura entrega 1": entrega[1],
        "Temperatura entrega 2": entrega[2],
        "Temperatura entrega 3": entrega[3],
        "Temperatura retorno 1": retorno[1],
        "Temperatura retorno 2": retorno[2],
        "Temperatura retorno 3": retorno[3],
        "SP_actual zona 1": sp_act[1],
        "SP_actual zona 2": sp_act[2],
        "SP_actual zona 3": sp_act[3],
        "Velocidad línea": velocidad,
    }
    for i in range(1, 11):
        data[f"Humedad piso {i}"] = hum[i]
        data[f"Humedad historica piso {i}"] = hist[i]

    df_in = pd.DataFrame([data])
    preds = model.predict(df_in)[0]

    # Redondear y capar a máximos históricos
    recs = {i: min(int(round(preds[i-1])), max_sp[i]) for i in range(1, 4)}

    # Mostrar métricas
    st.subheader("🌡️ SP recomendadas")
    st.metric("Zona 1 (°C)", recs[1])
    st.metric("Zona 2 (°C)", recs[2])
    st.metric("Zona 3 (°C)", recs[3])
    st.write("---")
    for i in range(1, 4):
        st.write(f"- Zona {i}: {recs[i]}°C (máx hist.: {max_sp[i]}°C)")
