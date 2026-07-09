import pandas as pd
import requests
import time

BASE_URL="https://transfermarkt-api.fly.dev"

mapping=pd.read_csv(
    "data/raw/transfermarkt/team_mapping.csv"
)

for i,row in mapping.iterrows():
    if pd.isna(row["club_id"]):
        name=row["transfermarkt_name"]

        r=requests.get(
            f"{BASE_URL}/clubs/search/{name}"
        )

        data=r.json()

        results=data.get("results",[])

        if results:
            print(
                name,
                "->",
                results[0]["id"]
            )
        else:
            print(
                name,
                "NOT FOUND"
            )

        time.sleep(1)
    