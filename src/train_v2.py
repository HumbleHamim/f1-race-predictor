import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

df = pd.read_csv("data/features.csv")
df = df.sort_values(["season", "round"]).reset_index(drop=True)

# new features (only past races are used)
finished = (
    (df["status"] == "Finished")
    | df["status"].str.startswith("+")
    | (df["status"] == "Lapped")
)
df["dnf"] = (~finished).astype(int)

df["dnf_rate"] = df.groupby("driver")["dnf"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).mean()
).fillna(0.1)

df["circuit_wins"] = df.groupby(["driver", "circuit"])["won"].cumsum() - df["won"]

df["circuit_avg"] = df.groupby(["driver", "circuit"])["position"].transform(
    lambda s: s.shift(1).expanding().mean()
)
df["circuit_avg"] = df["circuit_avg"].fillna(df["driver_form"])

df.to_csv("data/features_v2.csv", index=False)

old = ["grid", "is_pole", "driver_form", "team_form",
       "driver_wins", "driver_points", "team_wins"]
new = old + ["dnf_rate", "circuit_wins", "circuit_avg"]

train = df[df["season"] <= 2020]
test = df[df["season"] >= 2021].copy()

pole = test[test["grid"] == 1]
print("Races tested:", test.groupby(["season", "round"]).ngroups)
print("Pole baseline:", round((pole["position"] == 1).mean() * 100, 1), "%")
print()

for fname, fset in [("OLD features", old), ("NEW features", new)]:
    models = {
        "Logistic": LogisticRegression(max_iter=2000),
        "Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=0
        ),
    }
    for mname, m in models.items():
        m.fit(train[fset], train["won"])
        test["p"] = m.predict_proba(test[fset])[:, 1]
        picked = test.loc[test.groupby(["season", "round"])["p"].idxmax()]
        print(fname, "|", mname,
              "| accuracy:", round(picked["won"].mean() * 100, 1), "%",
              "| log loss:", round(log_loss(test["won"], test["p"]), 4))