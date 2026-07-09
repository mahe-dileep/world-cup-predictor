"""
Latest (current) Elo rating per team.

Derived from the canonical history at data/interim/elo_history.csv. For each
team we take its most recent match and report `elo_after` — i.e. the rating
*after* that team's last played match, which is its current strength.

Output schema: date, team, elo   (one row per team)
"""
from pathlib import Path
import pandas as pd

INPUT = Path("data/interim/elo_history.csv")
OUTPUT = Path("data/processed/latest_elo.csv")


def create_latest_elo():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    latest = (
        df.sort_values(["date", "match_id"], kind="mergesort")
        .groupby("team", as_index=False)
        .tail(1)
        .rename(columns={"elo_after": "elo"})
        .sort_values("elo", ascending=False)
    )

    latest = latest[["date", "team", "elo"]]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    latest.to_csv(OUTPUT, index=False)
    print(f"Saved {len(latest)} teams")


if __name__ == "__main__":
    create_latest_elo()
