import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

def generar_caso_de_uso_segmentar_clientes_kmeans():
    """
    Genera un caso de uso aleatorio para la función segmentar_clientes_kmeans(df, k).
    Devuelve:
        input_data: dict con claves 'df' y 'k'
        output_data: DataFrame esperado
    """
    n = random.randint(30, 80)
    k = random.randint(2, 5)

    df = pd.DataFrame({
        "edad": np.random.randint(18, 70, size=n),
        "ingresos": np.random.randint(1000, 20000, size=n),
        "gasto": np.random.randint(100, 5000, size=n)
    })

    input_data = {"df": df.copy(), "k": k}

    modelo = KMeans(n_clusters=k, n_init=10, random_state=42)
    output_df = df.copy()
    output_df["cluster"] = modelo.fit_predict(output_df)

    return input_data, output_df
