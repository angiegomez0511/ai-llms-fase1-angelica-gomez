import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def detectar_drift_ks(df_base, df_nuevo, columnas, alpha=0.05):
    rows = []
    cols_drift = []

    for c in columnas:
        x = df_base[c].dropna().to_numpy()
        y = df_nuevo[c].dropna().to_numpy()

        if len(x) == 0 or len(y) == 0:
            ks_stat, p_value = np.nan, np.nan
            hay_drift = False
        else:
            ks_stat, p_value = ks_2samp(x, y)
            hay_drift = p_value < alpha

        rows.append({
            "columna": c,
            "ks_stat": ks_stat,
            "p_value": p_value,
            "hay_drift": hay_drift
        })

        if hay_drift:
            cols_drift.append(c)

    df_result = (
        pd.DataFrame(rows)
        .sort_values("ks_stat", ascending=False, na_position="last")
        .reset_index(drop=True)
    )

    return df_result, cols_drift
