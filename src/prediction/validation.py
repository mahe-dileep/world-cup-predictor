"""Loud, meaningful validation for the inference layer.

Every failure mode requested in Phase 6 has a dedicated exception so callers
(API, UI) can react precisely instead of parsing strings.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


class PredictionError(Exception):
    """Base class for all prediction-layer errors."""


class ModelArtifactError(PredictionError):
    """A required model artifact is missing or unreadable."""


class MissingColumnsError(PredictionError):
    """The feature frame is missing columns the model requires."""


class ExtraColumnsError(PredictionError):
    """The feature frame contains columns the model does not expect."""


class ColumnOrderError(PredictionError):
    """Feature columns are present but not in the trained order."""


class DuplicateFeatureError(PredictionError):
    """Duplicate feature names in the feature frame."""


class MissingFeatureValueError(PredictionError):
    """A feature value is missing where the model requires one."""


class UnknownTeamError(PredictionError):
    """A team name cannot be resolved through the canonical mapping."""


class DuplicateTeamError(PredictionError):
    """A team resolves ambiguously (mapping integrity problem)."""


class ProbabilityError(PredictionError):
    """A probability vector is invalid (NaN, negative, or not summing to 1)."""


def require_files(paths) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise ModelArtifactError(f"missing required artifact(s): {missing}")


def validate_feature_frame(df: pd.DataFrame, feature_columns,
                           allow_nan: bool = True) -> None:
    """Enforce the exact model contract: same columns, same order, no dupes."""
    cols = list(df.columns)
    if len(cols) != len(set(cols)):
        dup = sorted({c for c in cols if cols.count(c) > 1})
        raise DuplicateFeatureError(f"duplicate feature columns: {dup}")

    have, want = set(cols), set(feature_columns)
    missing = want - have
    if missing:
        raise MissingColumnsError(f"feature frame missing {len(missing)} columns: "
                                  f"{sorted(missing)[:10]}...")
    extra = have - want
    if extra:
        raise ExtraColumnsError(f"feature frame has unexpected columns: {sorted(extra)[:10]}")
    if cols != list(feature_columns):
        raise ColumnOrderError("feature columns are not in the trained order")

    if not allow_nan:
        # Only the non-categorical numeric columns must be complete.
        na = df.isna().any()
        bad = [c for c in feature_columns if na.get(c, False)]
        if bad:
            raise MissingFeatureValueError(f"missing feature values in: {bad[:10]}")


def validate_probabilities(P, tol: float = 1e-6) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    if not np.all(np.isfinite(P)):
        raise ProbabilityError("probabilities contain NaN/inf")
    if np.any(P < -tol):
        raise ProbabilityError("probabilities contain negative values")
    sums = P.sum(axis=-1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        raise ProbabilityError(f"probabilities do not sum to 1 (min={sums.min():.6f}, "
                               f"max={sums.max():.6f})")
    return P


def validate_mapping_integrity(mapping: pd.DataFrame) -> None:
    """Fail loudly on duplicate teams in the canonical mapping."""
    for col in ["worldcup_name", "elo_name"]:
        dups = mapping[col][mapping[col].duplicated()].tolist()
        if dups:
            raise DuplicateTeamError(f"duplicate {col} in team mapping: {dups}")
