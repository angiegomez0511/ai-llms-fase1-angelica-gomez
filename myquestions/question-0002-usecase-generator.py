import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_metricas_campana():
    """
    Genera un caso de uso aleatorio para la función calcular_metricas_campana(df).
    Devuelve:
        input_data: dict con la clave 'df'
        output_data: DataFrame esperado
    """
    n = random.randint(12, 30)

    df = pd.DataFrame({
        "campaña": np.random.choice(["Lanzamiento", "Remarketing", "LeadGen", "Branding"], size=n),
        "clicks": np.random.randint(1, 200, size=n),
        "impresiones": np.random.randint(100, 5000, size=n),
        "costo": np.round(np.random.uniform(10, 800, size=n), 2)
    })

    input_data = {"df": df.copy()}

    df_calc = df.copy()
    df_calc["ctr"] = df_calc["clicks"] / df_calc["impresiones"]
    df_calc["cpc"] = df_calc["costo"] / df_calc["clicks"]

    output_data = (
        df_calc.groupby("campaña")[["ctr", "cpc"]]
        .mean()
        .reset_index()
        .rename(columns={"ctr": "ctr_promedio", "cpc": "cpc_promedio"})
    )

    return input_data, output_data
