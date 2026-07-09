"""World Cup 2026 tournament simulator.

Uses the model's neutral-venue match probabilities (via MonteCarloSimulator) to
Monte-Carlo the full competition: 12 groups of 4 (round robin) -> top two of each
group + eight best third-placed teams -> 32-team single-elimination bracket ->
champion. Aggregates the round each team reaches over many simulations.

NOTE (documented approximation): the official 2026 Round-of-32 slotting table is
intricate; this simulator seeds the 32 qualifiers by group-stage points and folds
a standard bracket (seed s vs seed 33-s, top seeds separated). This keeps stronger
teams apart until later rounds but is NOT the exact official pairing. Champion
probabilities are therefore indicative. The group assignments themselves come
straight from data/raw/world_cup_2026/teams.csv (group_letter).
"""
from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd

from src.prediction import loaders
from src.prediction.predictor import Predictor
from src.prediction.simulator import MonteCarloSimulator

ROOT = loaders.ROOT
WC_TEAMS_CSV = ROOT / "data" / "raw" / "world_cup_2026" / "teams.csv"


class WorldCupSimulator:
    def __init__(self, predictor: Predictor | None = None, seed: int = 42):
        self.predictor = predictor or Predictor()
        self.sim = MonteCarloSimulator(self.predictor, seed)
        self.seed = seed
        self.groups = self._load_groups()
        self.teams = [t for g in self.groups.values() for t in g]

    def _load_groups(self):
        df = pd.read_csv(WC_TEAMS_CSV)
        groups = defaultdict(list)
        for _, r in df.sort_values(["group_letter", "team_name"]).iterrows():
            groups[r["group_letter"]].append(r["team_name"])
        return dict(groups)

    def _precompute(self):
        """Warm the symmetric-probability cache for every needed pairing."""
        for i, a in enumerate(self.teams):
            for b in self.teams[i + 1:]:
                self.sim.symmetric_probs(a, b)
                self.sim.advance_prob(a, b)

    def _sim_group(self, teams, rng):
        n = len(teams)
        pts = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                p = self.sim.symmetric_probs(teams[i], teams[j])
                o = rng.choice(3, p=p)
                if o == 0:
                    pts[i] += 3
                elif o == 1:
                    pts[i] += 1; pts[j] += 1
                else:
                    pts[j] += 3
        order = sorted(range(n), key=lambda t: (pts[t], rng.random()), reverse=True)
        ranked = [(teams[t], pts[t]) for t in order]
        return ranked  # [(team, points)] best-first

    def _knockout(self, seeded, rng, track):
        """seeded: list of teams in bracket order (power of 2). Records the round
        reached in `track`. Returns (champion, finalist_pair)."""
        round_names = {32: "round_of_32", 16: "round_of_16", 8: "quarterfinal",
                       4: "semifinal", 2: "final"}
        alive = list(seeded)
        finalists = None
        while len(alive) > 1:
            if len(alive) == 2:
                finalists = tuple(sorted(alive))
            for t in alive:
                track[t].add(round_names.get(len(alive), f"round_{len(alive)}"))
            nxt = []
            for i in range(0, len(alive), 2):
                a, b = alive[i], alive[i + 1]
                nxt.append(a if rng.random() < self.sim.advance_prob(a, b) else b)
            alive = nxt
        track[alive[0]].add("champion")
        return alive[0], finalists

    def run(self, n_sims=10000, seed=None):
        self._precompute()
        rng = np.random.default_rng(self.seed if seed is None else seed)
        reached = {t: defaultdict(int) for t in self.teams}
        final_pairs = defaultdict(int)

        for _ in range(n_sims):
            winners, runners, thirds = [], [], []
            for letter, teams in self.groups.items():
                ranked = self._sim_group(teams, rng)
                winners.append(ranked[0])
                runners.append(ranked[1])
                thirds.append(ranked[2])
            # eight best third-placed teams by points (random tiebreak)
            thirds_sorted = sorted(thirds, key=lambda tp: (tp[1], rng.random()), reverse=True)
            qualifiers = winners + runners + thirds_sorted[:8]   # 12+12+8 = 32
            # seed by points desc, standard bracket fold (s vs 33-s)
            qualifiers.sort(key=lambda tp: (tp[1], rng.random()), reverse=True)
            seeds = [t for t, _ in qualifiers]
            bracket = []
            lo, hi = 0, len(seeds) - 1
            while lo < hi:
                bracket += [seeds[lo], seeds[hi]]
                lo += 1; hi -= 1

            track = defaultdict(set)
            _, finalists = self._knockout(bracket, rng, track)
            if finalists:
                final_pairs[finalists] += 1
            for t, rounds in track.items():
                for r in rounds:
                    reached[t][r] += 1

        def pct(stage):
            return {t: reached[t].get(stage, 0) / n_sims for t in self.teams}

        champion = dict(sorted(pct("champion").items(), key=lambda kv: -kv[1]))

        # Most likely final matchup + a projected scoreline for it.
        (fa, fb), freq = max(final_pairs.items(), key=lambda kv: kv[1])
        score = self.predictor.predict_score(fa, fb)      # neutral WC venue
        projected_final = {
            "matchup": [fa, fb],
            "probability": freq / n_sims,
            "projected_score": f"{fa} {score['most_likely_score']} {fb}",
            "expected_goals": score["expected_goals"],
            "result_probs": score["result_probs"],
        }
        return {
            "n_sims": n_sims,
            "champion": champion,
            "finalist": pct("final"),
            "semifinalist": pct("semifinal"),
            "quarterfinalist": pct("quarterfinal"),
            "round_of_16": pct("round_of_16"),
            "projected_final": projected_final,
        }

    @staticmethod
    def top(champion_probs: dict, k=10):
        return list(champion_probs.items())[:k]
