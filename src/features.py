import pandas as pd

df = pd.read_csv("data/races.csv")
df = df.sort_values(["season", "round"]).reset_index(drop=True)

# target: did this driver win?
df["won"] = (df["position"] == 1).astype(int)

# pit lane starts are stored as grid 0, treat them as last place
df.loc[df["grid"] == 0, "grid"] = 20

# driver form: average finish in the driver's previous 5 races
df["driver_form"] = df.groupby("driver")["position"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).mean()
)

# team form: average finish of the team's cars in the previous 5 races
team = (
    df.groupby(["constructor", "season", "round"], as_index=False)["position"]
    .mean()
    .sort_values(["constructor", "season", "round"])
)
team["team_form"] = team.groupby("constructor")["position"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).mean()
)
df = df.merge(
    team[["constructor", "season", "round", "team_form"]],
    on=["constructor", "season", "round"],
)

# new features
df["is_pole"] = (df["grid"] == 1).astype(int)

df["driver_wins"] = df.groupby("driver")["won"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).sum()
)
df["driver_points"] = df.groupby("driver")["points"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=1).sum()
)

tw = (
    df.groupby(["constructor", "season", "round"], as_index=False)["won"]
    .sum()
    .rename(columns={"won": "team_won"})
    .sort_values(["constructor", "season", "round"])
)
tw["team_wins"] = tw.groupby("constructor")["team_won"].transform(
    lambda s: s.shift(1).rolling(10, min_periods=1).sum()
)
df = df.merge(
    tw[["constructor", "season", "round", "team_wins"]],
    on=["constructor", "season", "round"],
)

# new drivers/teams have no history yet, use a neutral value
df["driver_form"] = df["driver_form"].fillna(10)
df["team_form"] = df["team_form"].fillna(10)
df["driver_wins"] = df["driver_wins"].fillna(0)
df["driver_points"] = df["driver_points"].fillna(0)
df["team_wins"] = df["team_wins"].fillna(0)

df = df.sort_values(["season", "round"]).reset_index(drop=True)
df.to_csv("data/features.csv", index=False)

print(df.shape)
print(df[["season", "round", "driver", "grid", "driver_wins", "driver_points", "team_wins", "won"]].tail(10))