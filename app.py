```python
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import LabelEncoder

# Configuración de página
st.set_page_config(page_title="Secadero IA - SP Recomendada", layout="centered")

# Título
st.title("Recomendador Inteligente de Temperaturas SP por Zona")

@st.cache_data
def load_and_train():
    # Carga de datos desde archivo en la raíz del repo
df = pd.read_excel("Secadero 1 automático.xlsx")
    df.columns = df.columns.str.strip()

    # Columnas de interés
    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]
    humedad_cols = [f"Humedad piso {i}" for i in range(1, 11)]
    humedad_hist_cols = [f"Humedad historica piso {i}" for i in range(1, 11)]
    entrada_cols = [
        "Tipo de placa",
        "Peso Húmedo", "Agua", "Yeso", "Agua Evaporada",
        "Temperatura entrega 1", "Temperatura entrega 2", "Temperatura entrega 3",
        "Temperatura retorno 1", "Temperatura retorno 2", "Temperatura retorno 3",
        "SP_actual zona 1", "SP_actual zona 2", "SP_actual zona 3",
        "Velocidad línea"
    ]

    # Verificar existencia de columnas
    required = entrada_cols + humedad_cols + target_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error("Faltan columnas en el Excel:")
        for col in missing:
            st.write(f"- {col}")
        st.stop()

    # Limpiar filas con datos faltantes en columnas críticas
    df = df.dropna(subset=required)

    # Calcular diferencias de humedad
    for i in range(1, 11):
        df[f"diff_humedad_{i}"] = (
            df[f"Humedad piso {i}"] -
            (df[f"Humedad historica piso {i}"] if f"Humedad historica piso {i}" in df.columns else df[f"Humedad piso {i}"])
        )

    diff_cols = [f"diff_humedad_{i}" for i in range(1, 11)]
    feature_cols = [c for c in entrada_cols if c != "SP_actual zona 1" and c != "SP_actual zona 2" and c != "SP_actual zona 3"] + diff_cols

    # Codificar tipo de placa
df["Tipo de placa"] = LabelEncoder().fit_transform(df["Tipo de placa"].astype(str))

    X = df[feature_cols]
    Y = df[target_cols]

    # Entrenar modelo
    model = MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        random_state=42
    ))
    model.fit(X, Y)

    # Máximos históricos\max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}
    return model, max_sp, feature_cols

# Carga del modelo
model, max_sp, feature_cols = load_and_train()

# Interfaz de usuario
tipo = st.selectbox("Tipo de placa", LabelEncoder().fit(pd.read_excel("Secadero 1 automático.xlsx")["Tipo de placa"]))
peso = st.number_input("Peso Húmedo", min_value=0.0)
agua = st.number_input("Agua", min_value=0.0)
yeso = st.number_input("Yeso", min_value=0.0)
evap = st.number_input("Agua Evaporada", min_value=0.0)
ent1, ent2, ent3 = [st.number_input(f"Temperatura entrega {i}", min_value=0.0) for i in range(1, 4)]
ret1, ret2, ret3 = [st.number_input(f"Temperatura retorno {i}", min_value=0.0) for i in range(1, 4)]
sp1, sp2, sp3 = [st.number_input(f"SP_actual zona {i}", min_value=0.0) for i in range(1, 4)]
vel = st.number_input("Velocidad línea", min_value=0.0)
humedades = [st.number_input(f"Humedad piso {i}", min_value=0.0) for i in range(1, 11)]

if st.button("Calcular SP Recomendadas"):
    # Preparar input
    df_in = pd.DataFrame([{
        "Tipo de placa": tipo,
        "Peso Húmedo": peso,
        "Agua": agua,
        "Yeso": yeso,
        "Agua Evaporada": evap,
        "Temperatura entrega 1": ent1,
        "Temperatura entrega 2": ent2,
        "Temperatura entrega 3": ent3,
        "Temperatura retorno 1": ret1,
        "Temperatura retorno 2": ret2,
        "Temperatura retorno 3": ret3,
        "SP_actual zona 1": sp1,
        "SP_actual zona 2": sp2,
        "SP_actual zona 3": sp3,
        "Velocidad línea": vel,
        **{f"Humedad piso {i+1}": humedades[i] for i in range(10)}
    }])

    # Codificar y calcular difs
    df_in["Tipo de placa"] = LabelEncoder().fit(pd.read_excel("Secadero 1 automático.xlsx")["Tipo de placa"]).transform(df_in["Tipo de placa"])
    for i in range(1, 11):
        hist = pd.read_excel("Secadero 1 automático.xlsx")[f"Humedad historica piso {i}"].mean()
        df_in[f"diff_humedad_{i}"] = df_in[f"Humedad piso {i}"] - hist

    df_in = df_in[feature_cols]
    preds = model.predict(df_in)[0]

    # Ajuste según humedad y límite máximo
    dif_media = np.mean([df_in[f"diff_humedad_{i}"].values[0] for i in range(1, 11)])
    ajuste = dif_media * 1.5
    results = {i: min(int(round(preds[i-1] - ajuste)), max_sp[i]) for i in range(1, 4)}

    st.subheader("SP Recomendadas por Zona")
    for z, val in results.items():
        st.write(f"Zona {z}: {val} °C")
    st.info("Ajuste final suavizado y limitado al máximo histórico de SP.")
```
