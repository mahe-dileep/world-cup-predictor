"""
Squad features — one row per World Cup 2026 national team.

Reads the canonical squad file (data/interim/squads.csv) and computes squad-level
aggregates only. No match data, no Elo, no historical information is touched, so
this cannot introduce leakage into the training set.

Age is computed from date_of_birth relative to the tournament start date
(2026-06-11); any pre-existing age column in the source is ignored on purpose.

Output: data/interim/squad_features.csv
"""
from pathlib import Path
import pandas as pd

SQUADS = Path("data/interim/squads.csv")
OUTPUT = Path("data/interim/squad_features.csv")

TOURNAMENT_START = pd.Timestamp("2026-06-11")
EXPECTED_TEAMS = 48
EXPECTED_SQUAD_SIZE = 26

POSITION_LABELS = {"GK": "goalkeepers", "DEF": "defenders",
                   "MID": "midfielders", "FWD": "forwards"}


def build_squad_features():
    df = pd.read_csv(SQUADS)

    # Age from date_of_birth at tournament start (ignore any source age column).
    dob = pd.to_datetime(df["date_of_birth"])
    df["age"] = (TOURNAMENT_START - dob).dt.days / 365.25

    # Position counts, one column per bucket, guaranteed present even if zero.
    pos_counts = (
        df.assign(pos=df["position"].map(POSITION_LABELS))
        .pivot_table(index="team", columns="pos", values="player_id",
                     aggfunc="count", fill_value=0)
        .reindex(columns=list(POSITION_LABELS.values()), fill_value=0)
    )

    agg = df.groupby("team").agg(
        players=("player_id", "count"),
        avg_age=("age", "mean"),
        median_age=("age", "median"),
        avg_height=("height_cm", "mean"),
        median_height=("height_cm", "median"),
        avg_caps=("caps", "mean"),
        median_caps=("caps", "median"),
        total_caps=("caps", "sum"),
        avg_market_value=("market_value_eur", "mean"),
        median_market_value=("market_value_eur", "median"),
        total_market_value=("market_value_eur", "sum"),
    )

    feats = agg.join(pos_counts)
    for pos in POSITION_LABELS.values():
        feats[f"pct_{pos}"] = feats[pos] / feats["players"]

    feats = feats.reset_index().sort_values("team").reset_index(drop=True)

    column_order = [
        "team", "players",
        "avg_age", "median_age",
        "avg_height", "median_height",
        "avg_caps", "median_caps", "total_caps",
        "avg_market_value", "median_market_value", "total_market_value",
        "goalkeepers", "defenders", "midfielders", "forwards",
        "pct_goalkeepers", "pct_defenders", "pct_midfielders", "pct_forwards",
    ]
    feats = feats[column_order]

    _validate(feats)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}: {len(feats)} teams x {len(feats.columns)} columns")
    return feats


def _validate(feats: pd.DataFrame):
    problems = []
    if len(feats) != EXPECTED_TEAMS:
        problems.append(f"expected {EXPECTED_TEAMS} teams, got {len(feats)}")
    if feats["team"].duplicated().any():
        problems.append(f"duplicate teams: {feats['team'][feats['team'].duplicated()].tolist()}")
    if not (feats["players"] == EXPECTED_SQUAD_SIZE).all():
        bad = feats.loc[feats["players"] != EXPECTED_SQUAD_SIZE, ["team", "players"]]
        problems.append(f"teams without exactly {EXPECTED_SQUAD_SIZE} players: {bad.to_dict('records')}")
    for col in ["avg_market_value", "total_market_value", "avg_age", "median_age",
                "avg_height", "median_height"]:
        n = int(feats[col].isna().sum())
        if n:
            problems.append(f"{n} missing values in {col}")
    if problems:
        raise ValueError("squad_features validation FAILED:\n  - " + "\n  - ".join(problems))
    print("Validation OK: 48 teams, no duplicates, 26 players each, no missing values.")


if __name__ == "__main__":
    build_squad_features()
