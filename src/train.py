import pandas as pd
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data/features.csv")
features = ["grid", "is_pole", "driver_form", "team_form", "driver_wins", "driver_points", "team_wins"]

train = df[df["season"] <= 2023]
test = df[df["season"] >= 2024].copy()

model = LogisticRegression(max_iter=1000)
model.fit(train[features], train["won"])

# probability that each driver wins
test["win_prob"] = model.predict_proba(test[features])[:, 1]

# in each race, predict the driver with the highest probability
picked = test.loc[test.groupby(["season", "round"])["win_prob"].idxmax()]
model_acc = picked["won"].mean()

# baseline on the same races: pole sitter wins
pole = test[test["grid"] == 1]
pole_acc = (pole["position"] == 1).mean()

print("Races tested:", len(picked))
print("Pole baseline:", round(pole_acc * 100, 1), "%")
print("Your model:   ", round(model_acc * 100, 1), "%")