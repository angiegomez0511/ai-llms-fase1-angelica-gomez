import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_predecir_conversion_clientes():
    """
    Genera un caso de uso aleatorio para la función predecir_conversion_clientes(df).
    Devuelve:
        input_data: dict con la clave 'df'
        output_data: float con accuracy esperado
    """
    n = random.randint(80, 140)

    df = pd.DataFrame({
        "edad": np.random.randint(18, 65, size=n),
        "ingresos": np.random.randint(1000, 15000, size=n),
        "visitas_web": np.random.randint(1, 25, size=n),
        "tiempo_en_pagina": np.random.randint(10, 600, size=n),
        "conversion": np.random.randint(0, 2, size=n)
    })

    input_data = {"df": df.copy()}

    X = df.drop(columns=["conversion"])
    y = df["conversion"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo = LogisticRegression(max_iter=500)
    modelo.fit(X_train, y_train)
    pred = modelo.predict(X_test)
    output_data = float(accuracy_score(y_test, pred))

    return input_data, output_data
