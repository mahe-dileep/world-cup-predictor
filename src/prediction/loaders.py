"""Centralised, cached loading of every model artifact and reference table.

Each artifact is read from disk at most once (functools.lru_cache). Nothing here
mutates any file. The saved model is a CalibratedModel from
src.models.ensemble, so the project root must be importable — we ensure that.
"""
from __future__ import annotations
import sys
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import validation as V

MODEL_PKL = ROOT / "models" / "catboost.pkl"
FEATURE_COLUMNS_JSON = ROOT / "models" / "feature_columns.json"
LABEL_ENCODER_PKL = ROOT / "models" / "label_encoder.pkl"
METADATA_JSON = ROOT / "models" / "training_metadata.json"
FEATURES_CSV = ROOT / "data" / "processed" / "features.csv"
LATEST_ELO_CSV = ROOT / "data" / "processed" / "latest_elo.csv"
TEAM_MAPPING_CSV = ROOT / "data" / "interim" / "team_name_mapping.csv"
# Needed to reconstruct per-team rolling-form snapshots (see feature_builder).
MATCHES_ML_CSV = ROOT / "data" / "processed" / "matches_ml.csv"


@lru_cache(maxsize=1)
def get_model():
    V.require_files([MODEL_PKL])
    return joblib.load(MODEL_PKL)


@lru_cache(maxsize=1)
def get_feature_columns():
    V.require_files([FEATURE_COLUMNS_JSON])
    return json.loads(FEATURE_COLUMNS_JSON.read_text())["feature_columns"]


@lru_cache(maxsize=1)
def get_label_encoder():
    V.require_files([LABEL_ENCODER_PKL])
    return joblib.load(LABEL_ENCODER_PKL)


@lru_cache(maxsize=1)
def get_metadata():
    V.require_files([METADATA_JSON])
    return json.loads(METADATA_JSON.read_text())


@lru_cache(maxsize=1)
def get_features_csv():
    V.require_files([FEATURES_CSV])
    return pd.read_csv(FEATURES_CSV)


@lru_cache(maxsize=1)
def get_latest_elo():
    V.require_files([LATEST_ELO_CSV])
    return pd.read_csv(LATEST_ELO_CSV)


@lru_cache(maxsize=1)
def get_team_mapping():
    V.require_files([TEAM_MAPPING_CSV])
    mp = pd.read_csv(TEAM_MAPPING_CSV)
    V.validate_mapping_integrity(mp)
    return mp


@lru_cache(maxsize=1)
def get_matches_ml():
    V.require_files([MATCHES_ML_CSV])
    return pd.read_csv(MATCHES_ML_CSV, parse_dates=["date"])


class NameResolver:
    """Bidirectional team-name resolution driven entirely by the mapping table.

    Any alias (worldcup / elo / historical / transfermarkt name) resolves to the
    canonical worldcup_name; the canonical name resolves to the elo_name used in
    latest_elo.csv and matches_ml.csv. No hardcoded aliases, no string joins.
    """

    ALIAS_COLUMNS = ["worldcup_name", "elo_name", "historical_name", "transfermarkt_name"]

    def __init__(self, mapping: pd.DataFrame):
        self.mapping = mapping
        self._alias_to_canon = {}
        for _, row in mapping.iterrows():
            canon = row["worldcup_name"]
            for col in self.ALIAS_COLUMNS:
                val = row.get(col)
                if isinstance(val, str) and val.strip():
                    key = self._norm(val)
                    prev = self._alias_to_canon.get(key)
                    if prev is not None and prev != canon:
                        raise V.DuplicateTeamError(
                            f"alias {val!r} maps to both {prev!r} and {canon!r}")
                    self._alias_to_canon[key] = canon
        self._canon_to_elo = dict(zip(mapping["worldcup_name"], mapping["elo_name"]))

    @staticmethod
    def _norm(name: str) -> str:
        return " ".join(str(name).strip().casefold().split())

    def to_canonical(self, name: str) -> str:
        canon = self._alias_to_canon.get(self._norm(name))
        if canon is None:
            raise V.UnknownTeamError(f"unknown team: {name!r}")
        return canon

    def to_elo_name(self, name: str) -> str:
        return self._canon_to_elo[self.to_canonical(name)]

    @property
    def teams(self):
        return list(self.mapping["worldcup_name"])


@lru_cache(maxsize=1)
def get_name_resolver() -> NameResolver:
    return NameResolver(get_team_mapping())


def clear_caches():
    """Testing/hygiene helper — drop every cached artifact."""
    for fn in (get_model, get_feature_columns, get_label_encoder, get_metadata,
               get_features_csv, get_latest_elo, get_team_mapping,
               get_matches_ml, get_name_resolver):
        fn.cache_clear()
