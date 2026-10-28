import time
import requests
import pandas as pd

rows = []

for year in range(2014, 2027):
    offset = 0
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json?limit=100&offset={offset}"
        resp = requests.get(url)
        if resp.status_code == 429:
            print("Too fast, waiting 10 seconds...")
            time.sleep(10)
            continue
        data = resp.json()["MRData"]
        total = int(data["total"])
        for race in data["RaceTable"]["Races"]:
            for r in race["Results"]:
                rows.append({
                    "season": int(race["season"]),
                    "round": int(race["round"]),
                    "race_name": race["raceName"],
                    "date": race["date"],
                    "circuit": race["Circuit"]["circuitId"],
                    "driver": r["Driver"]["driverId"],
                    "constructor": r["Constructor"]["constructorId"],
                    "grid": int(r["grid"]),
                    "position": int(r["position"]),
                    "status": r["status"],
                    "points": float(r["points"]),
                })
        offset += 100
        time.sleep(0.5)
        if offset >= total:
            break
    print(year, "done")

df = pd.DataFrame(rows)
df.to_csv("data/races.csv", index=False)
print(df.shape)
print(df.head())