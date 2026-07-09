"""Shared, dependency-free helpers for the prediction engine."""
from __future__ import annotations
import json
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Fix all RNGs used by the prediction/simulation layer."""
    random.seed(seed)
    np.random.seed(seed)


def normalize_probabilities(vec) -> np.ndarray:
    """Return a non-negative vector that sums to 1 (uniform if it sums to 0)."""
    v = np.asarray(vec, dtype=float)
    v = np.clip(v, 0.0, None)
    s = v.sum()
    if s <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / s


def validate_probability_vector(vec, tol: float = 1e-6) -> np.ndarray:
    """Raise ValueError if `vec` is not a clean probability distribution."""
    v = np.asarray(vec, dtype=float)
    if v.ndim != 1:
        raise ValueError(f"probability vector must be 1-D, got shape {v.shape}")
    if not np.all(np.isfinite(v)):
        raise ValueError("probability vector contains NaN/inf")
    if np.any(v < -tol):
        raise ValueError(f"probability vector has negative entries: {v}")
    if abs(v.sum() - 1.0) > tol:
        raise ValueError(f"probabilities sum to {v.sum():.6f}, not 1")
    return v


def safe_json(obj):
    """Recursively convert numpy/pandas scalars so json.dumps never fails."""
    if isinstance(obj, dict):
        return {str(k): safe_json(x) for k, x in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(x) for x in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return safe_json(obj.tolist())
    return obj


def canonical_team_name(name: str) -> str:
    """Resolve any known alias to its canonical World Cup name.

    Thin wrapper around the cached NameResolver so callers don't need the
    mapping table. Raises UnknownTeamError for names not in the mapping.
    """
    from src.prediction.loaders import get_name_resolver
    return get_name_resolver().to_canonical(name)


def dumps(obj, **kw) -> str:
    return json.dumps(safe_json(obj), **kw)
