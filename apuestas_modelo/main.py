import pandas as pd

# Cargar datos
df = pd.read_csv("data.csv")

# Mostrar primeras filas
print(df.head())

# Crear columna BTTS
df["BTTS"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)

# Ver resultados
print(df[["home_goals", "away_goals", "BTTS"]].head())

btts_rate = df["BTTS"].mean()
print("Frecuencia BTTS:", btts_rate)
