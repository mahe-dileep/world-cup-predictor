"""
Merge team-level features onto the canonical 48-team universe.

Sources combined here (one row per World Cup team):
  - Elo (current strength)      via elo_name       -> data/processed/latest_elo.csv
  - historical team statistics  via historical_name -> data/raw/teams/teams.csv
  - squad aggregates            via team (canonical) -> data/interim/squad_features.csv

Every team-identity join goes through data/interim/team_name_mapping.csv so that
naming differences between sources (South Korea/Korea Republic, USA/United States,
Bosnia, Congo DR, Iran, Türkiye/Turkey, Czechia, ...) can never cause a silent
mismatch. Squad columns are prefixed `squad_` so they never collide with a
historical column of the same name (e.g. avg_age).

This is a TEAM-level file (48 rows). Squad features are NOT joined onto historical
matches — that needs time-aware snapshots and belongs to a later phase.
"""
from pathlib import Path
import pandas as pd

MAPPING = Path("data/interim/team_name_mapping.csv")
HISTORICAL = Path("data/raw/teams/teams.csv")
ELO = Path("data/processed/latest_elo.csv")
SQUAD = Path("data/interim/squad_features.csv")
OUTPUT = Path("data/processed/features.csv")


def merge_features():
    mapping = pd.read_csv(MAPPING)
    hist = pd.read_csv(HISTORICAL)
    elo = pd.read_csv(ELO)
    squad = pd.read_csv(SQUAD)

    # Canonical universe: worldcup_name is the team identity everywhere downstream.
    df = mapping[["worldcup_name", "team_id", "elo_name", "historical_name"]].copy()

    # --- Elo (current team strength) via elo_name ---
    elo_latest = elo[["team", "elo"]].rename(columns={"team": "elo_name"})
    df = df.merge(elo_latest, on="elo_name", how="left")

    # --- Historical team statistics via historical_name ---
    hist_stats = hist.rename(columns={"team": "historical_name"})
    hist_stats = hist_stats.drop(columns=["team_country"], errors="ignore")
    df = df.merge(hist_stats, on="historical_name", how="left")

    # Present the canonical name as `team`; drop the join-helper name columns.
    df = df.rename(columns={"worldcup_name": "team"})
    df = df.drop(columns=["elo_name", "historical_name"])

    # --- Squad aggregates via canonical team; prefix to avoid column collisions ---
    squad_prefixed = squad.rename(
        columns={c: f"squad_{c}" for c in squad.columns if c != "team"}
    )
    df = df.merge(squad_prefixed, on="team", how="left")
    squad_cols = [c for c in df.columns if c.startswith("squad_")]

    # Column order: identity, elo, then everything else.
    lead = ["team", "team_id", "elo"]
    df = df[lead + [c for c in df.columns if c not in lead]]

    _validate(df, squad_cols)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}: {len(df)} teams x {len(df.columns)} columns")


def _validate(df: pd.DataFrame, squad_cols):
    problems = []
    if len(df) != 48:
        problems.append(f"expected 48 teams, got {len(df)}")

    dupes = df["team"][df["team"].duplicated()].tolist()
    if dupes:
        problems.append(f"duplicate teams: {dupes}")

    missing_elo = df.loc[df["elo"].isna(), "team"].tolist()
    if missing_elo:
        problems.append(f"missing Elo for: {missing_elo}")

    missing_squad = [c for c in squad_cols if df[c].isna().any()]
    if missing_squad:
        problems.append(f"missing squad statistics in: {missing_squad}")

    collisions = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    if collisions:
        problems.append(f"column collisions: {collisions}")

    if problems:
        raise ValueError("features.csv validation failed:\n  - " + "\n  - ".join(problems))
    print("Validation OK: 48 teams, no duplicates, no missing Elo, no missing squad stats.")


if __name__ == "__main__":
    merge_features()
