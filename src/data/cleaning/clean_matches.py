import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/raw/matches/results.csv")
OUTPUT_PATH = Path("data/interim/cleaned_matches.csv")


def clean_matches():
    df = pd.read_csv(RAW_PATH)

    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={
        "home_score": "home_goals",
        "away_score": "away_goals",
        "tournament": "competition"
    })

    df.insert(0, "match_id", range(1, len(df) + 1))

    def get_result(row):
        if row["home_goals"] > row["away_goals"]:
            return "H"
        elif row["home_goals"] < row["away_goals"]:
            return "A"
        return "D"

    df["result"] = df.apply(get_result, axis=1)
    df["season"] = df["date"].dt.year
    df["venue"] = df["city"]

    df = df[
        [
            "match_id",
            "date",
            "competition",
            "season",
            "home_team",
            "away_team",
            "home_goals",
            "away_goals",
            "result",
            "venue",
            "neutral"
        ]
    ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} matches")


if __name__ == "__main__":
    clean_matches()