import pandas as pd
from sklearn.preprocessing import StandardScaler

def pivotar_y_calcular_ranking(df, exp_col, metrica_col, valor_col, metrica_objetivo):
    pivot = pd.pivot_table(
        df,
        values=valor_col,
        index=exp_col,
        columns=metrica_col,
        aggfunc="mean"
    )

    res = pivot[[metrica_objetivo]].copy()

    scaler = StandardScaler()
    res["score_normalizado"] = scaler.fit_transform(res[[metrica_objetivo]])

    res = res.sort_values("score_normalizado", ascending=False)
    res["ranking"] = range(1, len(res) + 1)

    res = res.reset_index()[[exp_col, metrica_objetivo, "score_normalizado", "ranking"]]

    return res
