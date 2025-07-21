import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.multioutput import MultiOutputRegressor

@st.cache_data
def load_and_train():
    df = pd.read_excel("Secadero 1 automático.xlsx")

    # Limpiamos datos numéricos
    num_cols = [
        'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada',
        'Temperatura entrega 1', 'Temperatura entrega 2', 'Temperatura entrega 3',
        'Temperatura retorno 1', 'Temperatura retorno 2', 'Temperatura retorno 3',
        'SP_actual zona 1', 'SP_actual zona 2', 'SP_actual zona 3', 'Velocidad línea'
    ] + [f"Humedad piso {i}" for i in range(1, 11)] + [f"Humedad historica piso {i}" for i in range(1, 11)]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    feature_cols = [
        'Tipo de placa', 'Peso Húmedo', 'Agua ', 'Yeso', 'Agua Evaporada',
        'Temperatura entrega 1', 'Temperatura entrega 2', 'Temperatura entrega 3',
        'Temperatura retorno 1', 'Temperatura retorno 2', 'Temperatura retorno 3',
        'SP_actual zona 1', 'SP_actual zona 2', 'SP_actual zona 3', 'Velocidad línea'
    ] + [f"Humedad piso {i}" for i in range(1, 11)]

    target_cols = [f"SP_actual zona {i}" for i in range(1, 4)]

    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    le = LabelEncoder()
    X["Tipo de placa"] = le.fit_transform(X["Tipo de placa"])

    # Guardamos humedades históricas medias para referencia
    hist_means = df[[f"Humedad historica piso {i}" for i in range(1, 11)]].mean().to_dict()

    # Limitar SP al máximo histórico
    max_sp = {i: int(Y[f"SP_actual zona {i}"].max()) for i in range(1, 4)}

    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
    model.fit(X, Y)

    return model, le, max_sp, hist_means

model, le, max_sp, hist_means = load_and_train()

# Explicación: SP recomendado se ajusta para optimizar la humedad de la placa según histórico.
# A continuación, pedimos los datos actuales del día.

st.title("Secadero IA - Recomendación Temperaturas SP")

placas_disponibles = list(le.classes_)
placa = st.selectbox("Tipo de placa", placas_disponibles)

peso_humedo = st.number_input("Peso húmedo", min_value=0.0)
agua = st.number_input("Agua", min_value=0.0)
yeso = st.number_input("Yeso", min_value=0.0)
agua_evap = st.number_input("Agua evaporada", min_value=0.0)
entregas = [st.number_input(f"Temperatura entrega {i}", value=0.0) for i in range(1, 4)]
retornos = [st.number_input(f"Temperatura retorno {i}", value=0.0) for i in range(1, 4)]
sp_actual = [st.number_input(f"SP actual zona {i}", value=0.0) for i in range(1, 4)]
vel_linea = st.number_input("Velocidad línea", min_value=0.0)
humedades = [st.number_input(f"Humedad piso {i}", min_value=0.0) for i in range(1, 11)]

if st.button("Calcular Temperaturas Recomendadas"):
    df_in = pd.DataFrame({
        'Tipo de placa': [le.transform([placa])[0]],
        'Peso Húmedo': [peso_humedo],
        'Agua ': [agua],
        'Yeso': [yeso],
        'Agua Evaporada': [agua_evap],
        'Temperatura entrega 1': [entregas[0]],
        'Temperatura entrega 2': [entregas[1]],
        'Temperatura entrega 3': [entregas[2]],
        'Temperatura retorno 1': [retornos[0]],
        'Temperatura retorno 2': [retornos[1]],
        'Temperatura retorno 3': [retornos[2]],
        'SP_actual zona 1': [sp_actual[0]],
        'SP_actual zona 2': [sp_actual[1]],
        'SP_actual zona 3': [sp_actual[2]],
        'Velocidad línea': [vel_linea],
        **{f"Humedad piso {i}": [humedades[i - 1]] for i in range(1, 11)}
    })

    preds = model.predict(df_in)[0]

    resultados = {}
    for i in range(1, 4):
        sp_pred = round(preds[i - 1])
        if sp_pred > max_sp[i]:
            sp_pred = max_sp[i]
        resultados[f"Zona {i}"] = sp_pred

    st.subheader("Temperaturas SP recomendadas (sin superar máximos históricos):")
    for zona, valor in resultados.items():
        st.write(f"{zona}: {valor} °C")

        # Explicación del resultado
    st.markdown("---")
    st.subheader("Explicación detallada por zona:")
    for i in range(1, 4):
        actual = sp_actual[i-1]
        reco = resultados[f"Zona {i}"]
        diff = reco - actual
        if diff > 0:
            reason = (
                "La SP recomendada es más alta que la actual, lo que ayudará a incrementar la evaporación "
                "en la Zona " + str(i) + ", reduciendo la humedad superficial más rápido y evitando posibles arquements de la placa."
            )
        elif diff < 0:
            reason = (
                "La SP recomendada es menor que la actual en la Zona " + str(i) + ", para evitar sobresecado "
                "y daños en los bordes, reduciendo el riesgo de desprendimiento del cartón."
            )
        else:
            reason = (
                "La SP recomendada coincide con la actual en la Zona " + str(i) + ", indica que las condiciones actuales "
                "son óptimas según el histórico y no requieren ajuste."
            )
        st.write(f"**Zona {i}:** {reason}")

            )
        st.write(f"**Zona {i}:** {reason}")
