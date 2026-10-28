import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

df = pd.read_csv("data/features_v2.csv")
features = ["grid", "is_pole", "driver_form", "team_form", "driver_wins",
            "driver_points", "team_wins", "dnf_rate", "circuit_wins", "circuit_avg"]

def make_models():
    return {
        "Logistic": LogisticRegression(max_iter=2000),
        "Boosting": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=150, random_state=0
        ),
    }

rows = []
for year in range(2018, 2027):
    train = df[df["season"] < year]
    test = df[df["season"] == year].copy()

    n_races = test.groupby("round").ngroups
    pole_hits = int(((test["grid"] == 1) & (test["position"] == 1)).sum())
    row = {"season": year, "races": n_races, "Pole": pole_hits}

    for name, m in make_models().items():
        m.fit(train[features], train["won"])
        test["p"] = m.predict_proba(test[features])[:, 1]
        picked = test.loc[test.groupby("round")["p"].idxmax()]
        row[name] = int(picked["won"].sum())

    rows.append(row)

res = pd.DataFrame(rows)
print(res.to_string(index=False))
print()

total = res["races"].sum()
for col in ["Pole", "Logistic", "Boosting"]:
    print(col, "correct:", res[col].sum(), "/", total,
          "=", round(res[col].sum() / total * 100, 1), "%")