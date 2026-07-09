"""Monte Carlo simulation on top of the model's match probabilities.

All randomness flows from a seeded numpy Generator, so every simulation is
reproducible. No arbitrary weighting is applied — outcomes are sampled directly
from the predicted (H, D, A) distribution.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from src.prediction.predictor import Predictor
from src.prediction.utils import validate_probability_vector

OUTCOMES = ["H", "D", "A"]
WC_COMP, WC_NEUTRAL = "FIFA World Cup", True


class MonteCarloSimulator:
    def __init__(self, predictor: Predictor | None = None, seed: int = 42):
        self.predictor = predictor or Predictor()
        self.seed = seed
        self._pair_cache: dict = {}

    # ---- probability helpers ----
    def _probs(self, home, away, competition=WC_COMP, neutral=WC_NEUTRAL) -> np.ndarray:
        d = self.predictor.proba_dict(home, away, competition, neutral)
        return np.array([d["H"], d["D"], d["A"]], dtype=float)

    def symmetric_probs(self, a, b) -> np.ndarray:
        """Neutral-venue probabilities for a vs b, averaging both home slots so
        the arbitrary 'home' designation of a neutral match introduces no bias.
        Returns [P(a wins), P(draw), P(b wins)]."""
        key = (a, b)
        if key not in self._pair_cache:
            pa = self._probs(a, b)             # a as home
            pb = self._probs(b, a)             # b as home -> flip to a's view
            v = np.array([(pa[0] + pb[2]) / 2, (pa[1] + pb[1]) / 2, (pa[2] + pb[0]) / 2])
            self._pair_cache[key] = v / v.sum()
        return self._pair_cache[key]

    def advance_prob(self, a, b) -> float:
        """P(a beats b) in a knockout, splitting draws 50/50 (shootout)."""
        p = self.symmetric_probs(a, b)
        pa, pb = p[0] + 0.5 * p[1], p[2] + 0.5 * p[1]
        return pa / (pa + pb)

    # ---- public simulations ----
    def simulate_match(self, home, away, n_sims=10000,
                       competition=WC_COMP, neutral=WC_NEUTRAL, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        p = validate_probability_vector(self._probs(home, away, competition, neutral))
        counts = np.bincount(rng.choice(3, size=n_sims, p=p), minlength=3)
        emp = counts / n_sims
        return {
            "home_team": home, "away_team": away, "n_sims": n_sims,
            "predicted": dict(zip(OUTCOMES, p.round(6))),
            "empirical": dict(zip(OUTCOMES, emp.round(6))),
            "counts": dict(zip(OUTCOMES, counts.tolist())),
        }

    def simulate_fixture_list(self, fixtures: pd.DataFrame, n_sims=10000, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        rows = []
        for r in fixtures.itertuples(index=False):
            comp = getattr(r, "competition", WC_COMP)
            neu = getattr(r, "neutral", WC_NEUTRAL)
            p = self._probs(r.home_team, r.away_team, comp, neu)
            emp = np.bincount(rng.choice(3, size=n_sims, p=p), minlength=3) / n_sims
            rows.append({"home_team": r.home_team, "away_team": r.away_team,
                         "home_win": float(emp[0]), "draw": float(emp[1]),
                         "away_win": float(emp[2])})
        return pd.DataFrame(rows)

    def simulate_group(self, teams, n_sims=10000, seed=None):
        """Round-robin group. Returns each team's finishing-position probabilities
        and probability of advancing (top 2)."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        teams = list(teams)
        n = len(teams)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        probs = {(i, j): self.symmetric_probs(teams[i], teams[j]) for i, j in pairs}
        pos = np.zeros((n, n))
        for _ in range(n_sims):
            pts = np.zeros(n)
            for (i, j), p in probs.items():
                o = rng.choice(3, p=p)
                if o == 0:
                    pts[i] += 3
                elif o == 1:
                    pts[i] += 1; pts[j] += 1
                else:
                    pts[j] += 3
            order = sorted(range(n), key=lambda t: (pts[t], rng.random()), reverse=True)
            for rank, t in enumerate(order):
                pos[t, rank] += 1
        pos /= n_sims
        return {teams[t]: {**{f"P_pos{r+1}": float(pos[t, r]) for r in range(n)},
                           "P_advance": float(pos[t, 0] + pos[t, 1])}
                for t in range(n)}

    def simulate_knockout(self, bracket_teams, n_sims=10000, seed=None):
        """Single-elimination over a power-of-two ordered bracket. Returns each
        team's championship probability."""
        rng = np.random.default_rng(self.seed if seed is None else seed)
        assert len(bracket_teams) & (len(bracket_teams) - 1) == 0, "bracket size must be power of 2"
        champs = {t: 0 for t in bracket_teams}
        for _ in range(n_sims):
            alive = list(bracket_teams)
            while len(alive) > 1:
                nxt = []
                for i in range(0, len(alive), 2):
                    a, b = alive[i], alive[i + 1]
                    nxt.append(a if rng.random() < self.advance_prob(a, b) else b)
                alive = nxt
            champs[alive[0]] += 1
        return {t: champs[t] / n_sims for t in bracket_teams}
