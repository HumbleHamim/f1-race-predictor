import requests

url = "https://api.jolpi.ca/ergast/f1/2024/1/results.json"
data = requests.get(url).json()

race = data["MRData"]["RaceTable"]["Races"][0]
print(race["raceName"], race["date"])
print()

for r in race["Results"]:
    name = r["Driver"]["familyName"]
    print("Started:", r["grid"], "| Finished:", r["position"], "|", name)