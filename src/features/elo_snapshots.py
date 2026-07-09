"""
Pre-match Elo snapshots.

Reshapes the canonical Elo history (data/interim/elo_history.csv) into one row
per match holding both teams' ratings *immediately before kickoff*.

The join is on match_id, so it is exactly 1:1 with the match log and cannot be
confused by a team playing twice on the same date. home_elo / away_elo are the
`elo_before` values — never post-match.

Output schema:
    match_id, date, home_team, away_team, home_elo, away_elo
"""
from pathlib import Path
import pandas as pd

MATCHES = Path("data/interim/cleaned_matches.csv")
ELO = Path("data/interim/elo_history.csv")
OUTPUT = Path("data/interim/elo_snapshots.csv")


def create_snapshots():
    matches = pd.read_csv(MATCHES)
    elo = pd.read_csv(ELO)

    home = (elo[elo["is_home"] == 1]
            .rename(columns={"elo_before": "home_elo"})[["match_id", "home_elo"]])
    away = (elo[elo["is_home"] == 0]
            .rename(columns={"elo_before": "away_elo"})[["match_id", "away_elo"]])

    snapshots = (
        matches[["match_id", "date", "home_team", "away_team"]]
        .merge(home, on="match_id", how="left")
        .merge(away, on="match_id", how="left")
    )

    snapshots.to_csv(OUTPUT, index=False)
    print(f"Saved {len(snapshots)} Elo snapshots")


if __name__ == "__main__":
    create_snapshots()
