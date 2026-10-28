import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

df = pd.read_csv("data/features.csv")
features = ["grid", "is_pole", "driver_form", "team_form",
            "driver_wins", "driver_points", "team_wins"]

train = df[df["season"] <= 2020]
test = df[df["season"] >= 2021].copy()

n_races = test.groupby(["season", "round"]).ngroups
pole = test[test["grid"] == 1]
print("Races tested:", n_races)
print("Pole baseline:", round((pole["position"] == 1).mean() * 100, 1), "%")
print()

models = {
    "Logistic": LogisticRegression(max_iter=1000),
    "Boosting": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, random_state=0
    ),
}

for name, m in models.items():
    m.fit(train[features], train["won"])
    test["p"] = m.predict_proba(test[features])[:, 1]

    picked = test.loc[test.groupby(["season", "round"])["p"].idxmax()]
    acc = picked["won"].mean()
    ll = log_loss(test["won"], test["p"])

    not_pole = picked["grid"] != 1
    print(name)
    print("  accuracy:", round(acc * 100, 1), "%")
    print("  log loss:", round(ll, 4), "(lower is better)")
    print("  picked someone other than pole:", not_pole.sum(), "times")
    print("  ...and was right:", picked.loc[not_pole, "won"].sum(), "times")
    print()