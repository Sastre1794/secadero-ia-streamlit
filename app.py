import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Secadero IA - Recomendador SP Por Héctor Sastre", layout="centered")
st.title("Recomendador Inteligente de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # 1) Carga correcta del Excel
    df = pd.read_excel("Secadero 1 automático.xlsx")

    # 2) Columnas según tu Excel
    entrada_cols = [
        "Tipo de placa", "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    # 3) Elimina filas vacías solo en las columnas de entrada + targets
    df = df.dropna(subset=entrada_cols + target_cols)

    # 4) Label encode para tipo de placa
    le = LabelEncoder()
    df["Tipo de placa"] = le.fit_transform(df["Tipo de placa"].astype(str))

    # 5) Construye X,Y
    X = df[entrada_cols + humedad_cols].copy()
    Y = df[target_cols].copy()

    # 6) Entrena modelo
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    # 7) Guarda máximos históricos
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # 8) Media histórica por piso (si no existe, se ignora)
    hist_means = {}
    for i in range(1, 11):
        col = f"Humedad historica piso {i}"
        hist_means[i] = df[col].mean() if col in df.columns else X[f"Humedad piso {i}"].mean()

    return model, le, max_sp, hist_means, entrada_cols, humedad_cols

model, le, max_sp, hist_means, entrada_cols, humedad_cols = load_and_train()

st.header("Introduce los datos actuales")

# 9) Inputs estrictos según entrada_cols
input_data = {}
# Tipo de placa
placa = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = le.transform([placa])[0]
# Otros campos
for col in entrada_cols:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, step=1.0, format="%.2f")

# Humedades actuales
for col in humedad_cols:
    input_data[col] = st.number_input(col, step=0.1, format="%.2f")

if st.button("Calcular Temperaturas SP Recomendadas"):
    # DataFrame de entrada
    df_in = pd.DataFrame([input_data])[entrada_cols + humedad_cols]

    # Difierecia media de humedad
    hum_act = df_in[humedad_cols].values.flatten()
    hum_obj = np.array([hist_means[i] for i in range(1, 11)])
    dif_media = (hum_act - hum_obj).mean()

    # Predicción y limitación
    preds = model.predict(df_in)[0]
    ajuste = dif_media * 1.5  # suave
    resultados = {
        i: min(int(round(preds[i - 1] - ajuste)), max_sp[i])
        for i in range(1, 4)
    }

    # Mostrar
    st.subheader("Temperaturas SP Recomendadas")
    for zona, sp in resultados.items():
        st.write(f"Zona {zona}: {sp} °C")

    st.info(
        "La recomendación ajusta las temperaturas según la diferencia media de humedad "
        "frente a histórico, sin superar los máximos registrados."
    )
