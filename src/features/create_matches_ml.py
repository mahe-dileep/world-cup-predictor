"""
Build the machine-learning match dataset (historical training data).

Each row is one international match with its outcome targets and ONLY leakage-free
pre-match features. Assembled from four sources, all joined on match_id:

    cleaned_matches.csv   -- identifiers, targets, venue, dates
    elo_snapshots.csv     -- pre-match home_elo / away_elo
    form_features.csv     -- rolling pre-match form (home_* / away_*)
    matchup_features.csv  -- home-minus-away difference features

Deliberately NOT joined (would leak 2026-only / future information into history):
    features.csv, squad_features.csv, latest_elo.csv

Targets kept untouched: home_goals, away_goals, result.
Output: data/processed/matches_ml.csv (one row per match_id).
"""
from pathlib import Path
import pandas as pd

MATCHES = Path("data/interim/cleaned_matches.csv")
SNAPSHOTS = Path("data/interim/elo_snapshots.csv")
FORM = Path("data/interim/form_features.csv")
MATCHUP = Path("data/interim/matchup_features.csv")
OUTPUT = Path("data/processed/matches_ml.csv")


def create_matches_ml():
    matches = pd.read_csv(MATCHES)
    snaps = pd.read_csv(SNAPSHOTS)
    form = pd.read_csv(FORM)
    matchup = pd.read_csv(MATCHUP)

    # Join on match_id only, selecting non-overlapping columns, so no match
    # column is ever duplicated -> no _x / _y artifacts.
    df = (
        matches
        .merge(snaps[["match_id", "home_elo", "away_elo"]], on="match_id", how="left")
        .merge(form, on="match_id", how="left")
        .merge(matchup, on="match_id", how="left")
    )

    _validate(df, matches)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}: {len(df)} matches x {len(df.columns)} columns")
    return df


def _validate(df, matches):
    problems = []
    if len(df) != len(matches):
        problems.append(f"rows {len(df)} != cleaned_matches {len(matches)}")
    if df["match_id"].duplicated().any():
        problems.append("duplicate match_id")
    dupe_cols = [c for c in df.columns if list(df.columns).count(c) > 1]
    if dupe_cols:
        problems.append(f"duplicate columns: {set(dupe_cols)}")
    collisions = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if collisions:
        problems.append(f"_x/_y collisions: {collisions}")
    for t in ("home_goals", "away_goals", "result"):
        if t not in df.columns:
            problems.append(f"missing target column {t}")
    # Elo must be present for every match (played or scheduled).
    for c in ("home_elo", "away_elo"):
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{n} missing {c}")
    if problems:
        raise ValueError("matches_ml validation FAILED:\n  - " + "\n  - ".join(problems))
    print("Validation OK: rows match, unique match_id, no dup/_x/_y columns, targets + Elo present.")


if __name__ == "__main__":
    create_matches_ml()
