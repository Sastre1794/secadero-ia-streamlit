import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # 1) Carga y limpieza de columnas
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # 2) Columnas de entrada y target
    feature_cols = [
        "Tipo de placa",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + \
      [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = ["SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3"]

    # 3) Verificación
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        st.error("Faltan columnas en el Excel:")
        for c in missing:
            st.write(f"- {c}")
        st.stop()

    # 4) Codificar tipo de placa
    le = LabelEncoder()
    df["Tipo_Code"] = le.fit_transform(df["Tipo de placa"])

    # 5) Montar X e Y
    X = df[["Tipo_Code"] + [c for c in feature_cols if c != "Tipo de placa"]]
    Y = df[target_cols]

    # 6) Eliminar filas incompletas
    df_clean = pd.concat([X, Y], axis=1).dropna()
    X_clean = df_clean[X.columns]
    Y_clean = df_clean[Y.columns]

    # 7) Entrenar modelo multisalida
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_clean, Y_clean)

    # 8) Máximos históricos de SP a partir de Y_clean
    max_sp = {
        i: int(Y_clean[f"SP_actual zona {i}"].max())
        for i in range(1, 4)
    }

    return model, le, max_sp

model, le, max_sp = load_and_train()

# —————————————————————————————
# Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador de SP Ajustada en Secadero")
st.write("Introduce los datos actuales para obtener las SP recomendadas por zona.")

# Inputs
col1, col2 = st.columns(2)
with col1:
    tipo = st.selectbox("Tipo de placa", le.classes_)
    code = le.transform([tipo])[0]
    entrega = {i: st.number_input(f"Temperatura entrega {i}", format="%.1f") for i in (1,2,3)}
    retorno = {i: st.number_input(f"Temperatura retorno {i}", format="%.1f") for i in (1,2,3)}
    sp_act = {i: st.number_input(f"SP_actual zona {i}", format="%.1f") for i in (1,2,3)}
with col2:
    velocidad = st.number_input("Velocidad línea (m/min)", format="%.1f")
    hum = {i: st.number_input(f"Humedad piso {i}", format="%.3f") for i in range(1, 11)}
    hist = {i: st.number_input(f"Humedad historica piso {i}", format="%.3f") for i in range(1, 11)}

if st.button("Calcular SP recomendada"):
    data = {
        "Tipo_Code": code,
        **{f"Temperatura entrega {i}": entrega[i] for i in (1,2,3)},
        **{f"Temperatura retorno {i}": retorno[i] for i in (1,2,3)},
        **{f"SP_actual zona {i}": sp_act[i] for i in (1,2,3)},
        "Velocidad línea": velocidad,
        **{f"Humedad piso {i}": hum[i] for i in range(1,11)},
        **{f"Humedad historica piso {i}": hist[i] for i in range(1,11)},
    }
    df_in = pd.DataFrame([data])
    preds = model.predict(df_in)[0]

    recs = {
        i: min(int(round(preds[i-1])), max_sp[i])
        for i in range(1, 4)
    }

    st.subheader("🌡️ SP recomendadas")
    for i in range(1,4):
        st.metric(f"Zona {i} (°C)", recs[i])

    st.write("---")
    for i in range(1,4):
        st.write(f"- Zona {i}: {recs[i]}°C (máx hist.: {max_sp[i]}°C)")
