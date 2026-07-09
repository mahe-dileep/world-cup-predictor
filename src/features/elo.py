"""
Canonical Elo engine.

Reads the cleaned match log and produces the single authoritative Elo history
at data/interim/elo_history.csv. This is the ONLY place Elo ratings are
computed and the ONLY location the history is written.

For every match we record each team's rating *before* kickoff (elo_before) and
*after* the result is applied (elo_after). Ratings are processed strictly in
chronological order so elo_before is always leakage-free (it can only depend on
matches that already happened).

Unplayed fixtures (e.g. future World Cup knockout matches with no score yet)
are recorded with their pre-match rating but do NOT update ratings — otherwise
a NaN score would corrupt every subsequent rating.

Output schema (long, one row per team per match):
    match_id, date, team, is_home, elo_before, elo_after
"""
from pathlib import Path
import math
import pandas as pd

INPUT = Path("data/interim/cleaned_matches.csv")
OUTPUT = Path("data/interim/elo_history.csv")

INITIAL_ELO = 1500
K_FACTOR = 30
HOME_ADVANTAGE = 65


def expected_score(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def goal_multiplier(home_goals, away_goals):
    """Bigger rating swing for larger winning margins."""
    margin = abs(home_goals - away_goals)
    return 1 if margin <= 1 else math.log(margin + 1) * 1.5


def update_elo(rating, expected, actual, multiplier):
    return rating + K_FACTOR * multiplier * (actual - expected)


def build_elo():
    df = pd.read_csv(INPUT)
    df["date"] = pd.to_datetime(df["date"])

    # Deterministic chronological order; match_id breaks same-date ties so the
    # output is identical on every run.
    df = df.sort_values(["date", "match_id"], kind="mergesort").reset_index(drop=True)

    elo = {}
    history = []

    for row in df.itertuples(index=False):
        home, away = row.home_team, row.away_team
        home_elo = elo.get(home, INITIAL_ELO)
        away_elo = elo.get(away, INITIAL_ELO)

        played = pd.notna(row.home_goals) and pd.notna(row.away_goals)
        if played:
            adjusted_home = home_elo if row.neutral else home_elo + HOME_ADVANTAGE
            exp_home = expected_score(adjusted_home, away_elo)
            exp_away = expected_score(away_elo, adjusted_home)

            if row.result == "H":
                actual_home, actual_away = 1, 0
            elif row.result == "A":
                actual_home, actual_away = 0, 1
            else:
                actual_home, actual_away = 0.5, 0.5

            mult = goal_multiplier(row.home_goals, row.away_goals)
            home_after = update_elo(home_elo, exp_home, actual_home, mult)
            away_after = update_elo(away_elo, exp_away, actual_away, mult)
        else:
            # No result yet: keep the pre-match rating, do not update.
            home_after, away_after = home_elo, away_elo

        history.append({"match_id": row.match_id, "date": row.date, "team": home,
                        "is_home": 1, "elo_before": home_elo, "elo_after": home_after})
        history.append({"match_id": row.match_id, "date": row.date, "team": away,
                        "is_home": 0, "elo_before": away_elo, "elo_after": away_after})

        elo[home] = home_after
        elo[away] = away_after

    result = pd.DataFrame(
        history,
        columns=["match_id", "date", "team", "is_home", "elo_before", "elo_after"],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Created {len(result)} Elo records across {result['match_id'].nunique()} matches")


if __name__ == "__main__":
    build_elo()
