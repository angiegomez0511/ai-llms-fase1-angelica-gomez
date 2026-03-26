import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_limpiar_leads_marketing():
    """
    Genera un caso de uso aleatorio para la función limpiar_leads_marketing(df).
    Devuelve:
        input_data: dict con la clave 'df'
        output_data: DataFrame esperado
    """
    n_filas = random.randint(12, 25)

    data = {
        "nombre": [f"lead_{i}" for i in range(n_filas)],
        "email": [f"lead_{i}@mail.com" for i in range(n_filas)],
        "edad": np.random.randint(18, 60, size=n_filas),
        "canal": np.random.choice(["facebook", "instagram", "google", "tiktok"], size=n_filas),
        "col_irrelevante": np.random.choice([np.nan, "x"], size=n_filas, p=[0.7, 0.3])
    }

    df = pd.DataFrame(data)

    # Introducir algunos nulos aleatorios
    for _ in range(random.randint(2, 5)):
        fila = random.randint(0, len(df) - 1)
        col = random.choice(["nombre", "email", "edad", "canal"])
        df.loc[fila, col] = np.nan

    # Agregar duplicados aleatorios
    duplicados = df.sample(random.randint(2, 4), replace=True, random_state=None)
    df = pd.concat([df, duplicados], ignore_index=True)

    input_data = {"df": df.copy()}

    # Output esperado
    umbral = len(df) * 0.5
    cols_validas = df.columns[df.isnull().sum() <= umbral]
    output_data = df[cols_validas].drop_duplicates().dropna().reset_index(drop=True)

    return input_data, output_data
