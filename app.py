import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de la página
st.set_page_config(page_title="Secadero IA - SP Recomendada", layout="centered")
st.title("Recomendador de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Carga de datos
df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Columnas de entrada y target\entrada_cols = [
        "Tipo de placa",
        "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    # Verificar existencia
equired = entrada_cols + humedad_cols + hist_cols + target_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan columnas en el Excel:")
        for col in missing:
            st.write(f"- {col}")
        st.stop()

    # Eliminar filas con NaN en columnas críticas
df = df.dropna(subset=entrada_cols + humedad_cols)

    # Calcular diffs de humedad
    diff_cols = []
    for i in range(1, 11):
        col_act = f"Humedad piso {i}"
        col_hist = f"Humedad historica piso {i}"
        diff = df[col_act] - df[col_hist]
        df[f"diff_humedad_{i}"] = diff
        diff_cols.append(f"diff_humedad_{i}")

    # Preparar matrices
    feature_cols = entrada_cols + diff_cols
    df["Tipo de placa"] = LabelEncoder().fit_transform(df["Tipo de placa"].astype(str))
    X = df[feature_cols]
    Y = df[target_cols]

    # Entrenar modelo suave
    model = MultiOutputRegressor(
        GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42)
    )
    model.fit(X, Y)

    # Máximos SP históricos
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    # Medias históricas de humedad para explicación
    hist_means = {i: df[f"Humedad historica piso {i}"].mean() for i in range(1, 11)}

    return model, LabelEncoder().fit(df["Tipo de placa"]), max_sp, hist_means, feature_cols, humedad_cols

# Carga del modelo
model, le, max_sp, hist_means, feature_cols, humedad_cols = load_and_train()

# Inputs usuario
input_data = {}

# Tipo de placa
tipo = st.selectbox("Tipo de placa", le.classes_)
input_data["Tipo de placa"] = tipo

# Resto de entradas excepto diffsor col in [c for c in feature_cols if not c.startswith("diff_humedad_")]:
    if col != "Tipo de placa":
        input_data[col] = st.number_input(col, format="%.2f")

# Humedades actuales
humedades = []
for i in range(1, 11):
    h = st.number_input(f"Humedad piso {i}", format="%.2f")
    input_data[f"Humedad piso {i}"] = h
    humedades.append(h)

if st.button("Calcular SP Recomendadas"):
    # DataFrame de entrada
df_in = pd.DataFrame([input_data])
    df_in["Tipo de placa"] = le.transform(df_in["Tipo de placa"])

    # Añadir diffsor i in range(1, 11):
        df_in[f"diff_humedad_{i}"] = humedades[i-1] - hist_means[i]

    # Reordenar para predict
    df_in = df_in[feature_cols]

    preds = model.predict(df_in)[0]

    st.subheader("Resultados SP Recomendadas y Diferencias")
    for i in range(1, 4):
        sp_act = input_data[f"SP_actual zona {i}"]
        sp_rec = min(int(round(preds[i-1])), max_sp[i])
        diff = sp_rec - sp_act
        st.write(
            f"Zona {i}: Recomendada {sp_rec} °C ({'+' if diff>=0 else ''}{diff}°C respecto actual {sp_act}°C)"
        )

    st.info(
        "La SP recomendada ajusta de acuerdo a la desviación de humedad vs histórico, "
        "limitada a los máximos registrados y mostrando la variación respecto al valor actual."
    )
```
