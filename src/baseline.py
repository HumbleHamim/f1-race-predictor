import pandas as pd

df = pd.read_csv("data/races.csv")

# one row per race: who started 1st, and did they win?
pole = df[df["grid"] == 1].copy()
pole["won"] = pole["position"] == 1

print("Races with a pole sitter:", len(pole))
print("Pole sitter won:", pole["won"].sum())
print("Baseline accuracy:", round(pole["won"].mean() * 100, 1), "%")
print()

# accuracy per season
print(pole.groupby("season")["won"].mean().round(2))