import pandas as pd
import numpy as np

# =========================
# CARGA Y LIMPIEZA BÁSICA
# =========================
df = pd.read_csv("data.csv")

# Si tienes fecha, la ordenamos
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

# Crear BTTS real
df["BTTS"] = ((df["home_goals"] > 0) & (df["away_goals"] > 0)).astype(int)


# =========================
# FUNCIONES DEL MODELO
# =========================
def build_strengths(train_df):
    """
    Calcula fuerzas ofensivas y defensivas usando solo partidos del pasado.
    """
    avg_home_goals = train_df["home_goals"].mean()
    avg_away_goals = train_df["away_goals"].mean()

    home_attack = train_df.groupby("home_team")["home_goals"].mean() / avg_home_goals
    away_defense = train_df.groupby("away_team")["home_goals"].mean() / avg_home_goals
    away_attack = train_df.groupby("away_team")["away_goals"].mean() / avg_away_goals
    home_defense = train_df.groupby("home_team")["away_goals"].mean() / avg_away_goals

    return {
        "avg_home_goals": avg_home_goals,
        "avg_away_goals": avg_away_goals,
        "home_attack": home_attack,
        "away_defense": away_defense,
        "away_attack": away_attack,
        "home_defense": home_defense,
    }


def expected_goals(home, away, strengths):
    """
    Calcula goles esperados del local y visitante.
    """
    ha = strengths["home_attack"].get(home, 1.0)
    ad = strengths["away_defense"].get(away, 1.0)
    aa = strengths["away_attack"].get(away, 1.0)
    hd = strengths["home_defense"].get(home, 1.0)

    lambda_home = ha * ad * strengths["avg_home_goals"]
    lambda_away = aa * hd * strengths["avg_away_goals"]

    return lambda_home, lambda_away


def prob_btts(lambda_home, lambda_away):
    """
    Probabilidad de que ambos marquen.
    Fórmula Poisson independiente.
    """
    return 1 - np.exp(-lambda_home) - np.exp(-lambda_away) + np.exp(-(lambda_home + lambda_away))


# =========================
# PASO 8: PROBABILIDAD BTTS PARA CADA PARTIDO
# =========================
pred_probs = []

for i in range(len(df)):
    if i == 0:
        pred_probs.append(np.nan)
        continue

    train_df = df.iloc[:i].copy()
    strengths = build_strengths(train_df)

    row = df.iloc[i]
    lh, la = expected_goals(row["home_team"], row["away_team"], strengths)
    p = prob_btts(lh, la)
    pred_probs.append(p)

df["pred_btts_prob"] = pred_probs

print("\nPrimeras predicciones BTTS:")
print(df[["date", "home_team", "away_team", "BTTS", "pred_btts_prob"]].head(10))


# =========================
# PASO 9: COMPARAR CONTRA CUOTAS
# =========================
# Si tu CSV tiene columna btts_odds, la usamos.
# Si no la tienes, más abajo te explico cómo agregarla.

if "btts_odds" in df.columns:
    df["implied_prob"] = 1 / df["btts_odds"]
    df["edge"] = df["pred_btts_prob"] - df["implied_prob"]
    df["signal"] = np.where(df["edge"] > 0.05, "BET", "NO BET")

    print("\nComparación modelo vs cuota:")
    print(df[["date", "home_team", "away_team", "pred_btts_prob", "btts_odds", "implied_prob", "edge", "signal"]].head(10))
else:
    print("\nNo existe la columna 'btts_odds' en tu CSV.")
    print("Para el paso 9, agrega una columna con cuotas BTTS si quieres comparar contra la casa.")


# =========================
# PASO 10: BACKTESTING SIMPLE
# =========================
# Este backtest asume que apuestas 1 unidad solo cuando hay señal BET.
# Si no tienes cuotas, no se puede calcular ROI real.

if "btts_odds" in df.columns:
    bets = df[df["signal"] == "BET"].copy()

    stake = 1.0
    profits = []

    for _, row in bets.iterrows():
        if row["BTTS"] == 1:
            profit = (row["btts_odds"] - 1) * stake
        else:
            profit = -stake
        profits.append(profit)

    bets["profit"] = profits

    total_bets = len(bets)
    total_profit = bets["profit"].sum()
    total_staked = total_bets * stake
    roi = (total_profit / total_staked) if total_staked > 0 else 0

    hit_rate = bets["BTTS"].mean() if total_bets > 0 else 0

    print("\n========== BACKTEST ==========")
    print("Apuestas hechas:", total_bets)
    print("Ganancia total:", round(total_profit, 2))
    print("Capital apostado:", round(total_staked, 2))
    print("ROI:", round(roi * 100, 2), "%")
    print("Hit rate:", round(hit_rate * 100, 2), "%")

    print("\nPrimeras apuestas simuladas:")
    print(bets[["date", "home_team", "away_team", "pred_btts_prob", "btts_odds", "BTTS", "profit"]].head(10))
else:
    print("\nSin cuotas no hay backtest de ROI real.")
