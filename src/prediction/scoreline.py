"""Exact-scoreline model.

The trained model only predicts the result (H/D/A). To get a scoreline we fit an
independent bivariate-Poisson goal model whose expected goals start from each
team's attack/defence form and are then adjusted so the Poisson-implied result
probabilities match the model's trusted (H, D, A) probabilities. The scoreline
distribution is therefore always consistent with the result model.
"""
from __future__ import annotations
from math import factorial
import numpy as np
from scipy.optimize import minimize

MAX_GOALS = 10


def poisson_pmf(k: int, lam: float) -> float:
    return np.exp(-lam) * lam ** k / factorial(k)


def score_matrix(lh: float, la: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    """M[i, j] = P(home scores i, away scores j)."""
    h = np.array([poisson_pmf(i, lh) for i in range(max_goals + 1)])
    a = np.array([poisson_pmf(j, la) for j in range(max_goals + 1)])
    M = np.outer(h, a)
    return M / M.sum()


def result_probs(M: np.ndarray):
    """(P_home_win, P_draw, P_away_win) from a scoreline matrix."""
    home = float(np.tril(M, -1).sum())   # i > j
    away = float(np.triu(M, 1).sum())    # i < j
    draw = float(np.trace(M))
    return home, draw, away


def fit_lambdas(pH, pD, pA, init=(1.35, 1.15), max_goals=MAX_GOALS):
    """Find (lambda_home, lambda_away) whose Poisson result probs match the model."""
    lo, hi = 0.05, 6.0

    def obj(x):
        lh, la = np.clip(x, lo, hi)
        H, D, A = result_probs(score_matrix(lh, la, max_goals))
        return (H - pH) ** 2 + (D - pD) ** 2 + (A - pA) ** 2

    r = minimize(obj, np.clip(init, lo, hi), method="Nelder-Mead",
                 options={"xatol": 1e-4, "fatol": 1e-9, "maxiter": 2000})
    return float(np.clip(r.x[0], lo, hi)), float(np.clip(r.x[1], lo, hi))


def top_scores(M: np.ndarray, n: int = 5):
    flat = np.argsort(-M.ravel())[:n]
    out = []
    for f in flat:
        i, j = np.unravel_index(f, M.shape)
        out.append(((int(i), int(j)), float(M[i, j])))
    return out


def markets(M: np.ndarray):
    idx = np.indices(M.shape)
    total = idx[0] + idx[1]
    over25 = float(M[total >= 3].sum())
    btts = float(M[1:, 1:].sum())
    # margin-of-victory buckets
    diff = idx[0] - idx[1]
    return {
        "over_2.5": over25, "under_2.5": 1 - over25,
        "btts_yes": btts, "btts_no": 1 - btts,
        "home_by_2plus": float(M[diff >= 2].sum()),
        "away_by_2plus": float(M[diff <= -2].sum()),
    }


def main_handicap_line(lh: float, la: float) -> float:
    """A sensible Asian-handicap line for the home team: expected goal margin
    rounded to the nearest half goal (never 0, so there's always a favourite)."""
    line = round((lh - la) * 2) / 2          # nearest 0.5
    line = -line                              # handicap is applied TO the home team
    if line == 0:
        line = -0.5 if lh >= la else 0.5
    return line


def asian_handicap(M: np.ndarray, line: float):
    """Probabilities for a home-team Asian handicap `line` (e.g. -1.5 means home
    must win by 2+). Whole lines can push (stake returned)."""
    idx = np.indices(M.shape)
    adj = (idx[0] + line) - idx[1]            # home margin after handicap
    return {
        "line": line,
        "home_cover": float(M[adj > 1e-9].sum()),
        "push": float(M[np.abs(adj) <= 1e-9].sum()),
        "away_cover": float(M[adj < -1e-9].sum()),
    }
