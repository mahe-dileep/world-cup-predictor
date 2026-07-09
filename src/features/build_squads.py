import pandas as pd
import requests
import time
import os

INPUT = "data/interim/squads.csv"
OUTPUT = "data/interim/player_transfermarkt.csv"

BASE = "https://transfermarkt-api.fly.dev"

df = pd.read_csv(INPUT)

players = df["player"].dropna().unique()

results = []

for i, player in enumerate(players):

    url = f"{BASE}/players/search/{player.replace(' ','%20')}"

    try:
        r = requests.get(url, timeout=10)

        if r.status_code != 200:
            print(player, "FAILED")
            continue

        data = r.json()

        matches = data.get("results", [])

        if len(matches) == 0:
            print(player, "NOT FOUND")
            continue

        best = matches[0]

        results.append({
            "player": player,
            "transfermarkt_id": best.get("id"),
            "transfermarkt_name": best.get("name"),
            "tm_market_value": best.get("marketValue")
        })

        print(i, player, "OK")

    except Exception as e:
        print(player, e)

    time.sleep(0.5)


out = pd.DataFrame(results)

os.makedirs("data/interim", exist_ok=True)

out.to_csv(
    OUTPUT,
    index=False
)

print("Saved", len(out), "players")