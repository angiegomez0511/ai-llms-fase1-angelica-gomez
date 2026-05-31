import numpy as np

def analizar_canciones(df):
    df_copy = df.copy()
    df_copy["eficiencia_popularidad"] = df_copy["popularidad"] / df_copy["duracion_segundos"]

    promedio_global = df_copy["reproducciones"].mean()
    filtrado = df_copy[df_copy["reproducciones"] > promedio_global]

    resultado = (
        filtrado
        .groupby("genero")
        .agg({
            "eficiencia_popularidad": "mean",
            "popularidad": "var",
            "reproducciones": lambda x: np.percentile(x, 75)
        })
        .rename(columns={
            "eficiencia_popularidad": "promedio_eficiencia",
            "popularidad": "varianza_popularidad",
            "reproducciones": "p75_reproducciones"
        })
        .reset_index()
        .sort_values(by="p75_reproducciones", ascending=False)
        .reset_index(drop=True)
    )

    return resultado
