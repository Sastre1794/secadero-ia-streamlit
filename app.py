import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

# —————————————————————————————
# 1) Carga y entrenamiento del modelo
# —————————————————————————————
@st.cache_data
def load_and_train(path="Secadero 1 automático.xlsx"):
    # Carga y limpieza
    df = pd.read_excel(path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()

    # Definición de columnas
    feature_cols = [
        "Tipo de placa",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + \
      [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = ["SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3"]

    # Validar columnas
    missing = [c for c in feature_cols + target_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    # Forzar numéricos
    num_cols = [c for c in feature_cols + target_cols if c != "Tipo de placa"]
    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")

    # Codificar categoría
    le = LabelEncoder()
    df["Tipo_Code"] = le.fit_transform(df["Tipo de placa"])

    # Preparar X e Y
    X = df[["Tipo_Code"] + [c for c in feature_cols if c != "Tipo de placa"]]
    Y = df[target_cols]

    # Limpiar filas incompletas
    df_clean = pd.concat([X, Y], axis=1).dropna()
    X_clean = df_clean[X.columns]
    Y_clean = df_clean[Y.columns]

    # Entrenar modelo multisalida
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X_clean, Y_clean)

    # Máximos históricos de SP y medias de humedad histórica
    max_sp = {i: int(Y_clean[f"SP_actual zona {i}"].max()) for i in range(1, 4)}
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() for i in range(1, 11)}

    # Guardar orden de características
    feature_names = list(X_clean.columns)

    return model, le, max_sp, hist_means, feature_names

# Cargar modelo y datos auxiliares
model, le, max_sp, hist_means, feature_names = load_and_train()

# —————————————————————————————
# 2) Interfaz Streamlit
# —————————————————————————————
st.set_page_config(page_title="Recomendador SP Secadero", layout="wide")
st.title("🔧 Recomendador de SP Ajustada en Secadero")
st.write("Introduce los datos actuales; la humedad histórica se aplicará automáticamente.")

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

if st.button("Calcular SP recomendada"):
    # Crear diccionario de entrada
    data = {
        "Tipo_Code": code,
        **{f"Temperatura entrega {i}": entrega[i] for i in (1,2,3)},
        **{f"Temperatura retorno {i}": retorno[i] for i in (1,2,3)},
        **{f"SP_actual zona {i}": sp_act[i] for i in (1,2,3)},
        "Velocidad línea": velocidad,
        **{f"Humedad piso {i}": hum[i] for i in range(1, 11)},
        **{f"Humedad historica piso {i}": hist_means[i] for i in range(1, 11)},
    }
    # DataFrame y reindex para coincidir con entrenamiento
    df_in = pd.DataFrame([data])[feature_names]

    # Predicción y capado
    preds = model.predict(df_in)[0]
    recs = {i: min(int(round(preds[i-1])), max_sp[i]) for i in range(1,4)}

    # Mostrar resultados
    st.subheader("🌡️ SP recomendadas")
    for i in range(1,4):
        st.metric(f"Zona {i} (°C)", recs[i])
    st.write("---")
    for i in range(1,4):
        st.write(f"- Zona {i}: {recs[i]}°C (Máx hist.: {max_sp[i]}°C)")
