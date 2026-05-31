def seleccionar_por_correlacion(df, target_col, k):
    corr = df.corr()[target_col].abs().drop(target_col)
    top_features = corr.sort_values(ascending=False).head(k).index
    return df[top_features]
