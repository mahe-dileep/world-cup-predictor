import time

import pandas as pd
import requests

BASE_URL = "https://transfermarkt-api.fly.dev"

rosters = pd.read_csv("data/raw/transfermarkt/world_cup_rosters.csv")

players = []

for _, row in rosters.iterrows():
    name = row["player_name"]
    team = row["team"]

    url = f"{BASE_URL}/players/search/{name}"

    r = requests.get(url)

    if r.status_code != 200:
        print("Failed:", name)
        continue

    data = r.json()

    results = data.get("results", [])

    if len(results) == 0:
        print("Not found:", name)
        continue

    player = results[0]

    players.append(
        {
            "team": team,
            "player_id": player.get("id"),
            "player_name": player.get("name"),
            "position": player.get("position"),
            "age": player.get("age"),
            "market_value": player.get("marketValue"),
            "club": player.get("club", {}).get("name"),
        }
    )

    print(team, name)

    time.sleep(0.5)


df = pd.DataFrame(players)

df.to_csv("data/raw/transfermarkt/player_values.csv", index=False)

print("Saved:", len(df))
