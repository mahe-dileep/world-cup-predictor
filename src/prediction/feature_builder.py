"""Reconstruct the exact training feature vector for an arbitrary matchup.

The model consumes 167 columns: competition, neutral, per-team pre-match Elo,
64 rolling-form features per side, and 35 home-minus-away matchup differences.
`features.csv` / `latest_elo.csv` only supply Elo, so the rolling-form part is
taken from each team's most recent snapshot in the frozen `matches_ml.csv`
(read-only) — the freshest form vector the pipeline produced for that team.

Namespaces are bridged only through team_name_mapping (elo_name <-> worldcup_name);
there are no hardcoded aliases and no direct string joins, so USA<->United States,
Türkiye<->Turkey, Congo DR<->DR Congo, etc. all resolve automatically.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from src.prediction import loaders, validation as V
from src.features.matchup_features import WINDOWS, WINDOW_BASES, FLAT_BASES

DEFAULT_COMPETITION = "FIFA World Cup"
DEFAULT_NEUTRAL = True

# Small precomputed per-team form snapshot (see app/precompute.py). When present it
# is used instead of reading the 50 MB matches_ml.csv at runtime — faster startup
# and a far lighter deploy. Falls back to matches_ml when absent (dev).
SNAPSHOT_FILE = loaders.ROOT / "data" / "processed" / "team_form_snapshot.json"

# home_/away_-prefixed model columns that are NOT per-team rolling form.
_NON_FORM = {"home_elo", "away_elo", "home_form_diff", "away_form_diff"}


class FeatureBuilder:
    def __init__(self):
        self.resolver = loaders.get_name_resolver()
        self.feature_columns = loaders.get_feature_columns()

        elo = loaders.get_latest_elo()
        self.elo_map = dict(zip(elo["team"], elo["elo"]))

        # Per-team rolling-form bases come from the MODEL CONTRACT (feature_columns),
        # which excludes ids/targets, so home_team/home_goals can never leak in.
        self.form_base = [c[len("home_"):] for c in self.feature_columns
                          if c.startswith("home_") and c not in _NON_FORM]
        away_base = [c[len("away_"):] for c in self.feature_columns
                     if c.startswith("away_") and c not in _NON_FORM]
        if sorted(self.form_base) != sorted(away_base):
            raise V.PredictionError("home/away form columns are not symmetric in feature_columns")

        if SNAPSHOT_FILE.exists():
            self.snapshots = json.loads(SNAPSHOT_FILE.read_text())
        else:
            self.snapshots = self._build_team_snapshots(loaders.get_matches_ml())

    # ---- per-team latest form snapshot (own side of most recent appearance) ----
    def _build_team_snapshots(self, matches):
        home = matches[["match_id", "date", "home_team"] +
                       [f"home_{b}" for b in self.form_base]].copy()
        home.columns = ["match_id", "date", "team"] + self.form_base
        away = matches[["match_id", "date", "away_team"] +
                       [f"away_{b}" for b in self.form_base]].copy()
        away.columns = ["match_id", "date", "team"] + self.form_base
        long = pd.concat([home, away], ignore_index=True)
        long = long.sort_values(["team", "date", "match_id"], kind="mergesort")
        latest = long.groupby("team", as_index=False).tail(1).set_index("team")
        return latest[self.form_base].to_dict("index")

    # ---- public: per-team feature contribution ----
    def build_features(self, team: str) -> dict:
        """Return {elo + 64 form features} for one team; raise if unavailable."""
        elo_name = self.resolver.to_elo_name(team)
        if elo_name not in self.elo_map:
            raise V.UnknownTeamError(f"no Elo for {team!r} (elo_name={elo_name!r})")
        if elo_name not in self.snapshots:
            raise V.UnknownTeamError(f"no match history for {team!r} (elo_name={elo_name!r})")
        feats = {"elo": float(self.elo_map[elo_name])}
        feats.update({b: self.snapshots[elo_name][b] for b in self.form_base})
        missing = [k for k, v in feats.items() if pd.isna(v)]
        if missing:
            raise V.MissingFeatureValueError(f"{team!r} has missing features: {missing}")
        return feats

    # ---- public: full match feature row ----
    def build_match_features(self, home_team: str, away_team: str,
                             competition: str = DEFAULT_COMPETITION,
                             neutral: bool = DEFAULT_NEUTRAL) -> pd.DataFrame:
        hf = self.build_features(home_team)
        af = self.build_features(away_team)
        row = {"competition": competition, "neutral": bool(neutral),
               "home_elo": hf["elo"], "away_elo": af["elo"]}
        for b in self.form_base:
            row[f"home_{b}"] = hf[b]
            row[f"away_{b}"] = af[b]

        # matchup differences — identical construction to matchup_features.py
        row["elo_diff"] = hf["elo"] - af["elo"]
        for w in WINDOWS:
            for b in WINDOW_BASES:
                row[f"{b}_diff_{w}"] = hf[f"{b}_{w}"] - af[f"{b}_{w}"]
        for b in FLAT_BASES:
            row[f"{b}_diff"] = hf[b] - af[b]
        row["home_form_diff"] = hf["home5_win_pct"] - af["home5_win_pct"]
        row["away_form_diff"] = hf["away5_win_pct"] - af["away5_win_pct"]
        row["venue_form_diff"] = hf["home5_win_pct"] - af["away5_win_pct"]

        df = pd.DataFrame([row]).reindex(columns=self.feature_columns)
        V.validate_feature_frame(df, self.feature_columns)
        return df

    def build_many(self, fixtures: pd.DataFrame) -> pd.DataFrame:
        """fixtures needs home_team, away_team (+ optional competition, neutral)."""
        rows = []
        for r in fixtures.itertuples(index=False):
            comp = getattr(r, "competition", DEFAULT_COMPETITION)
            neu = getattr(r, "neutral", DEFAULT_NEUTRAL)
            rows.append(self.build_match_features(r.home_team, r.away_team, comp, neu))
        out = pd.concat(rows, ignore_index=True)
        V.validate_feature_frame(out, self.feature_columns)
        return out
