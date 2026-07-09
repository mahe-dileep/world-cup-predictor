"""
Matchup difference features (leakage-free).

Every column here is a home-minus-away difference of two features that are
themselves strictly pre-match (from form_features.csv) or the pre-match Elo
(from elo_snapshots.csv). No new information is introduced, so this inherits the
leakage-free property of its inputs.

We only build differences that carry signal and avoid pairs that would be exact
duplicates of one another.

Output: data/interim/matchup_features.csv  (one row per match_id)
"""
from pathlib import Path
import pandas as pd

FORM = Path("data/interim/form_features.csv")
SNAPSHOTS = Path("data/interim/elo_snapshots.csv")
OUTPUT = Path("data/interim/matchup_features.csv")

WINDOWS = [3, 5, 10]
# per-window base features to difference (home_<f>_<w> - away_<f>_<w>)
WINDOW_BASES = ["ppg", "win_pct", "goals_scored", "goals_conceded",
                "goal_diff", "avg_opp_elo", "wins", "draws", "losses"]
# non-window base features
FLAT_BASES = ["rest_days", "days_since_prev_match", "matches_prev_30d", "matches_prev_90d"]


def build_matchup_features():
    form = pd.read_csv(FORM)
    snaps = pd.read_csv(SNAPSHOTS)

    out = form[["match_id"]].copy()

    # Elo difference (pre-match).
    elo = snaps.set_index("match_id")
    out = out.merge(
        (elo["home_elo"] - elo["away_elo"]).rename("elo_diff"),
        on="match_id", how="left",
    )

    for w in WINDOWS:
        for b in WINDOW_BASES:
            out[f"{b}_diff_{w}"] = form[f"home_{b}_{w}"] - form[f"away_{b}_{w}"]

    for b in FLAT_BASES:
        out[f"{b}_diff"] = form[f"home_{b}"] - form[f"away_{b}"]

    # Venue-aware form matchups.
    # home_form_diff : each team's HOME form (home team's home vs away team's home)
    out["home_form_diff"] = form["home_home5_win_pct"] - form["away_home5_win_pct"]
    # away_form_diff : each team's AWAY form
    out["away_form_diff"] = form["home_away5_win_pct"] - form["away_away5_win_pct"]
    # venue_form_diff: the actual matchup — home team AT HOME vs away team AWAY
    out["venue_form_diff"] = form["home_home5_win_pct"] - form["away_away5_win_pct"]

    _validate(out, form)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}: {len(out)} matches x {len(out.columns)} columns "
          f"({len(out.columns) - 1} diff features)")
    return out


def _validate(out, form):
    problems = []
    if len(out) != len(form):
        problems.append(f"rows {len(out)} != form {len(form)}")
    if out["match_id"].duplicated().any():
        problems.append("duplicate match_id")
    dupe_cols = [c for c in out.columns if list(out.columns).count(c) > 1]
    if dupe_cols:
        problems.append(f"duplicate columns: {set(dupe_cols)}")
    if problems:
        raise ValueError("matchup_features validation FAILED:\n  - " + "\n  - ".join(problems))
    print("Validation OK: one row per match, no dup match_id, no dup columns.")


if __name__ == "__main__":
    build_matchup_features()
