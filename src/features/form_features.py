"""
Rolling pre-match form features (leakage-free).

For every match and BOTH teams we summarise that team's form using ONLY matches
that were played strictly before the current match (kick-off). Nothing from the
current match or any later match is ever used, so there is no target leakage.

Inputs (only these two are allowed):
    data/interim/cleaned_matches.csv   -- results, targets, venue, dates
    data/interim/elo_snapshots.csv     -- PRE-match Elo per side (for opponent Elo)

Output:
    data/interim/form_features.csv     -- one row per match_id, home_*/away_* columns

Method: build a long team-match log (2 rows per match), sort chronologically per
team, and sweep forward. At each team-match we read the running history of that
team's PAST PLAYED matches, compute the features, and only THEN append the
current result to the history. Unplayed fixtures (future 2026 knockouts) still
receive features (from prior history) but never contribute to anyone's history.
"""
from pathlib import Path
from bisect import bisect_left
import pandas as pd

MATCHES = Path("data/interim/cleaned_matches.csv")
SNAPSHOTS = Path("data/interim/elo_snapshots.csv")
OUTPUT = Path("data/interim/form_features.csv")

WINDOWS = [3, 5, 10]
SPLIT_WINDOW = 5  # last-N home / last-N away


def _build_team_log(matches, snaps):
    """Long log: two perspective rows (home, away) per match."""
    df = matches.merge(snaps[["match_id", "home_elo", "away_elo"]], on="match_id", how="left")
    df["date"] = pd.to_datetime(df["date"])
    df["ord"] = df["date"].map(pd.Timestamp.toordinal)

    played = df["home_goals"].notna() & df["away_goals"].notna()

    def side(is_home):
        gs = df["home_goals"] if is_home else df["away_goals"]
        gc = df["away_goals"] if is_home else df["home_goals"]
        own_elo = df["home_elo"] if is_home else df["away_elo"]
        opp_elo = df["away_elo"] if is_home else df["home_elo"]
        team = df["home_team"] if is_home else df["away_team"]
        # points from this team's perspective
        if is_home:
            pts = df["result"].map({"H": 3, "D": 1, "A": 0})
        else:
            pts = df["result"].map({"A": 3, "D": 1, "H": 0})
        venue = pd.Series("neutral", index=df.index)
        venue = venue.where(df["neutral"], "home" if is_home else "away")
        return pd.DataFrame({
            "match_id": df["match_id"], "ord": df["ord"], "team": team,
            "is_home": 1 if is_home else 0, "played": played,
            "gs": gs, "gc": gc, "points": pts, "opp_elo": opp_elo, "venue": venue,
        })

    log = pd.concat([side(True), side(False)], ignore_index=True)
    # Deterministic order: team, then chronological, match_id breaks ties.
    return log.sort_values(["team", "ord", "match_id"], kind="mergesort").reset_index(drop=True)


def _window_stats(hist, w, out):
    """Fill `out` with features for the last `w` matches of history list `hist`."""
    sub = hist[-w:]
    n = len(sub)
    if n == 0:
        for k in ("ppg", "wins", "draws", "losses", "goals_scored", "goals_conceded",
                  "goal_diff", "win_pct", "clean_sheets", "failed_to_score", "avg_opp_elo",
                  "avg_goals_scored", "avg_goals_conceded", "avg_goal_diff",
                  "home_played", "away_played", "neutral_played", "matches_in_window"):
            out[f"{k}_{w}"] = float("nan")
        return
    pts = [e[0] for e in sub]
    gs = [e[1] for e in sub]
    gc = [e[2] for e in sub]
    opp = [e[3] for e in sub]
    ven = [e[4] for e in sub]
    wins = sum(p == 3 for p in pts)
    gs_sum, gc_sum = sum(gs), sum(gc)
    out[f"ppg_{w}"] = sum(pts) / n
    out[f"wins_{w}"] = wins
    out[f"draws_{w}"] = sum(p == 1 for p in pts)
    out[f"losses_{w}"] = sum(p == 0 for p in pts)
    out[f"goals_scored_{w}"] = gs_sum
    out[f"goals_conceded_{w}"] = gc_sum
    out[f"goal_diff_{w}"] = gs_sum - gc_sum
    out[f"win_pct_{w}"] = wins / n
    out[f"clean_sheets_{w}"] = sum(c == 0 for c in gc)
    out[f"failed_to_score_{w}"] = sum(s == 0 for s in gs)
    out[f"avg_opp_elo_{w}"] = sum(opp) / n
    out[f"avg_goals_scored_{w}"] = gs_sum / n
    out[f"avg_goals_conceded_{w}"] = gc_sum / n
    out[f"avg_goal_diff_{w}"] = (gs_sum - gc_sum) / n
    out[f"home_played_{w}"] = sum(v == "home" for v in ven)
    out[f"away_played_{w}"] = sum(v == "away" for v in ven)
    out[f"neutral_played_{w}"] = sum(v == "neutral" for v in ven)
    out[f"matches_in_window_{w}"] = n


def _split_stats(sub, tag, out):
    """Win% / avg goals for a home-only or away-only recent slice."""
    n = len(sub)
    if n == 0:
        out[f"{tag}_win_pct"] = float("nan")
        out[f"{tag}_avg_goals_scored"] = float("nan")
        out[f"{tag}_avg_goals_conceded"] = float("nan")
        return
    out[f"{tag}_win_pct"] = sum(e[0] == 3 for e in sub) / n
    out[f"{tag}_avg_goals_scored"] = sum(e[1] for e in sub) / n
    out[f"{tag}_avg_goals_conceded"] = sum(e[2] for e in sub) / n


def build_form_features():
    matches = pd.read_csv(MATCHES)
    snaps = pd.read_csv(SNAPSHOTS)
    log = _build_team_log(matches, snaps)

    rows = []
    # running per-team state, reset when the team changes (log is team-sorted)
    cur_team = None
    hist = home_hist = away_hist = None
    hist_ords = None

    for r in log.itertuples(index=False):
        if r.team != cur_team:
            cur_team = r.team
            hist, home_hist, away_hist, hist_ords = [], [], [], []

        out = {"match_id": r.match_id, "is_home": r.is_home}
        for w in WINDOWS:
            _window_stats(hist, w, out)

        # recency / rest
        if hist_ords:
            gap = r.ord - hist_ords[-1]
            out["days_since_prev_match"] = gap
            # Rest days between matches; two matches on the same day => 0, never negative.
            out["rest_days"] = max(gap - 1, 0)
            i30 = bisect_left(hist_ords, r.ord - 30)
            i90 = bisect_left(hist_ords, r.ord - 90)
            out["matches_prev_30d"] = len(hist_ords) - i30
            out["matches_prev_90d"] = len(hist_ords) - i90
        else:
            out["days_since_prev_match"] = float("nan")
            out["rest_days"] = float("nan")
            out["matches_prev_30d"] = 0
            out["matches_prev_90d"] = 0

        _split_stats(home_hist[-SPLIT_WINDOW:], "home5", out)
        _split_stats(away_hist[-SPLIT_WINDOW:], "away5", out)

        rows.append(out)

        # append AFTER computing features; only played matches enter history
        if r.played:
            entry = (r.points, r.gs, r.gc, r.opp_elo, r.venue)
            hist.append(entry)
            hist_ords.append(r.ord)
            if r.venue == "home":
                home_hist.append(entry)
            elif r.venue == "away":
                away_hist.append(entry)

    feat = pd.DataFrame(rows)
    feature_cols = [c for c in feat.columns if c not in ("match_id", "is_home")]

    home = (feat[feat["is_home"] == 1].drop(columns="is_home")
            .rename(columns={c: f"home_{c}" for c in feature_cols}))
    away = (feat[feat["is_home"] == 0].drop(columns="is_home")
            .rename(columns={c: f"away_{c}" for c in feature_cols}))

    wide = matches[["match_id"]].merge(home, on="match_id", how="left").merge(away, on="match_id", how="left")

    _validate(wide, matches)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(OUTPUT, index=False)
    print(f"Saved {OUTPUT}: {len(wide)} matches x {len(wide.columns)} columns "
          f"({len(feature_cols)} features per team)")
    return wide


def _validate(wide, matches):
    problems = []
    if len(wide) != len(matches):
        problems.append(f"rows {len(wide)} != cleaned_matches {len(matches)}")
    if wide["match_id"].duplicated().any():
        problems.append("duplicate match_id")
    dupe_cols = [c for c in wide.columns if list(wide.columns).count(c) > 1]
    if dupe_cols:
        problems.append(f"duplicate columns: {set(dupe_cols)}")
    if problems:
        raise ValueError("form_features validation FAILED:\n  - " + "\n  - ".join(problems))
    print("Validation OK: one row per match, no dup match_id, no dup columns.")


if __name__ == "__main__":
    build_form_features()
