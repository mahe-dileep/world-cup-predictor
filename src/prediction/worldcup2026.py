"""Faithful FIFA World Cup 2026 simulator using the REAL fixed bracket.

Two modes:
  * simulate_from_groups(n)   -- simulate the whole tournament from the group
                                 stage through the official fixed Round-of-32
                                 slotting (12 winners + 12 runners-up + 8 best
                                 thirds) to a champion.
  * project_current_bracket(n)-- take the tournament's real current state from
                                 the data (played through the Round of 16) and
                                 project the remaining bracket.

The fixed slot template and third-placed allowed-group sets are the official
2026 structure; group assignments come from the tournament data.
"""
from __future__ import annotations
from collections import defaultdict
import pandas as pd
import numpy as np

from src.prediction import loaders
from src.prediction.predictor import Predictor
from src.prediction.simulator import MonteCarloSimulator

ROOT = loaders.ROOT
TEAMS_CSV = ROOT / "data" / "raw" / "world_cup_2026" / "teams.csv"
MATCHES_CSV = ROOT / "data" / "raw" / "world_cup_2026" / "matches_detailed.csv"

# Fixed Round-of-32 template (data slot order 0..15): (homeLabel, awayLabel).
# Third slots carry the FIFA allowed-group set for the qualifying third-placed team.
_T = "3rd"
R32_TEMPLATE = [
    ("2A", "2B"), ("1C", "2F"), ("1E", (_T, set("ABCDF"))), ("1F", "2C"),
    ("2E", "2I"), ("1I", (_T, set("CDFGH"))), ("1A", (_T, set("CEFHI"))),
    ("1L", (_T, set("EHIJK"))), ("1G", (_T, set("AEHIJ"))), ("1D", (_T, set("BEFIJ"))),
    ("1H", "2J"), ("2K", "2L"), ("1B", (_T, set("EFGIJ"))), ("2D", "2G"),
    ("1J", "2H"), ("1K", (_T, set("DEIJL"))),
]
THIRD_SLOTS = [i for i, (_, a) in enumerate(R32_TEMPLATE) if isinstance(a, tuple)]
# Fixed R32 -> R16 tree: which two R32 slots (0-indexed) feed each R16 match.
FEEDERS = [(2, 5), (0, 3), (1, 4), (6, 7), (11, 10), (9, 8), (14, 13), (12, 15)]


class WorldCup2026:
    def __init__(self, predictor: Predictor | None = None, seed: int = 42):
        self.predictor = predictor or Predictor()
        self.sim = MonteCarloSimulator(self.predictor, seed)
        self.seed = seed
        self.res = loaders.get_name_resolver()

        tdf = pd.read_csv(TEAMS_CSV)
        self.groups = {}
        for _, r in tdf.iterrows():
            self.groups.setdefault(r["group_letter"], []).append(r["team_name"])
        elo = loaders.get_latest_elo()
        elo_map = dict(zip(elo["team"], elo["elo"]))
        self.team_elo = {t: elo_map[self.res.to_elo_name(t)]
                         for L in self.groups for t in self.groups[L]}
        self.teams = [t for L in self.groups for t in self.groups[L]]

    # ---------------------------------------------------------------- helpers
    def _sim_group(self, gteams, rng):
        n = len(gteams)
        pts = np.zeros(n)
        for i in range(n):
            for j in range(i + 1, n):
                o = rng.choice(3, p=self.sim.symmetric_probs(gteams[i], gteams[j]))
                if o == 0:
                    pts[i] += 3
                elif o == 1:
                    pts[i] += 1; pts[j] += 1
                else:
                    pts[j] += 3
        order = sorted(range(n), key=lambda t: (pts[t], self.team_elo[gteams[t]]), reverse=True)
        return [gteams[t] for t in order], [pts[t] for t in order]

    def _assign_thirds(self, qual_groups):
        allowed = {s: R32_TEMPLATE[s][1][1] for s in THIRD_SLOTS}
        order = sorted(qual_groups)
        assign, used = {}, set()

        def bt(k):
            if k == len(order):
                return True
            g = order[k]
            for s in THIRD_SLOTS:
                if s not in used and g in allowed[s]:
                    used.add(s); assign[g] = s
                    if bt(k + 1):
                        return True
                    used.discard(s); del assign[g]
            return False

        if not bt(0):  # safety fallback (should never happen for valid combos)
            for g, s in zip(order, THIRD_SLOTS):
                assign[g] = s
        return {s: g for g, s in assign.items()}

    def _play(self, a, b, rng):
        return a if rng.random() < self.sim.advance_prob(a, b) else b

    def _run_knockout(self, slot_teams, rng, reached):
        r32w = [self._play(h, a, rng) for h, a in slot_teams]
        for t in r32w:
            reached[t]["R16"] += 1
        r16w = [self._play(r32w[a], r32w[b], rng) for a, b in FEEDERS]
        for t in r16w:
            reached[t]["QF"] += 1
        qf = [self._play(r16w[2 * k], r16w[2 * k + 1], rng) for k in range(4)]
        for t in qf:
            reached[t]["SF"] += 1
        sf = [self._play(qf[0], qf[1], rng), self._play(qf[2], qf[3], rng)]
        for t in sf:
            reached[t]["Final"] += 1
        champ = self._play(sf[0], sf[1], rng)
        reached[champ]["Champion"] += 1
        return champ, tuple(sorted(sf))

    # ---------------------------------------------------------------- public
    def simulate_from_groups(self, n_sims=10000, seed=None):
        rng = np.random.default_rng(self.seed if seed is None else seed)
        reached = {t: defaultdict(int) for t in self.teams}
        final_pairs = defaultdict(int)

        for _ in range(n_sims):
            pos, thirds = {}, []
            for L, gt in self.groups.items():
                ranked, pts = self._sim_group(gt, rng)
                pos[f"1{L}"] = ranked[0]; pos[f"2{L}"] = ranked[1]
                thirds.append((L, ranked[2], pts[2]))
                reached[ranked[0]]["GroupWin"] += 1
            thirds.sort(key=lambda x: (x[2], self.team_elo[x[1]]), reverse=True)
            best = thirds[:8]
            third_team = {g: t for g, t, _ in best}
            slot_third = self._assign_thirds([g for g, _, _ in best])
            slot_teams = []
            for i, (hl, al) in enumerate(R32_TEMPLATE):
                home = pos[hl]
                away = third_team[slot_third[i]] if isinstance(al, tuple) else pos[al]
                slot_teams.append((home, away))
            for h, a in slot_teams:
                reached[h]["Qualify"] += 1; reached[a]["Qualify"] += 1
            _, finalists = self._run_knockout(slot_teams, rng, reached)
            final_pairs[finalists] += 1

        return self._summarise(reached, final_pairs, n_sims)

    def project_current_bracket(self, n_sims=20000, seed=None):
        """Project from the real state in the data (played through Round of 16)."""
        d = pd.read_csv(MATCHES_CSV)

        def winner(r):
            if pd.isna(r.home_score):
                return None
            if r.home_score != r.away_score:
                return r.home_team_name if r.home_score > r.away_score else r.away_team_name
            return r.home_team_name if r.home_penalty_score > r.away_penalty_score else r.away_team_name

        r16 = d[d.stage_name == "Round of 16"]
        qf_teams = [winner(r) for _, r in r16.iterrows()]   # 8, in bracket order
        QF = [(qf_teams[i], qf_teams[i + 1]) for i in range(0, 8, 2)]
        rng = np.random.default_rng(self.seed if seed is None else seed)
        sf = dict.fromkeys(qf_teams, 0); fin = dict.fromkeys(qf_teams, 0)
        champ = dict.fromkeys(qf_teams, 0)
        for _ in range(n_sims):
            q = [self._play(*QF[i], rng) for i in range(4)]
            for t in q:
                sf[t] += 1
            s1 = self._play(q[0], q[1], rng); s2 = self._play(q[2], q[3], rng)
            fin[s1] += 1; fin[s2] += 1
            champ[self._play(s1, s2, rng)] += 1
        return {
            "quarterfinalists": qf_teams,
            "quarterfinals": [{"home": h, "away": a,
                               "probs": self.sim.symmetric_probs(h, a).round(4).tolist()}
                              for h, a in QF],
            "reach_sf": {t: sf[t] / n_sims for t in qf_teams},
            "reach_final": {t: fin[t] / n_sims for t in qf_teams},
            "champion": dict(sorted(((t, champ[t] / n_sims) for t in qf_teams),
                                    key=lambda kv: -kv[1])),
            "n_sims": n_sims,
        }

    def _summarise(self, reached, final_pairs, n):
        def pct(stage):
            return {t: reached[t].get(stage, 0) / n for t in self.teams}

        champion = dict(sorted(pct("Champion").items(), key=lambda kv: -kv[1]))
        (fa, fb), freq = max(final_pairs.items(), key=lambda kv: kv[1])
        score = self.predictor.predict_score(fa, fb)
        return {
            "n_sims": n,
            "champion": champion,
            "group_win": pct("GroupWin"),
            "qualify": pct("Qualify"),
            "reach_r16": pct("R16"),
            "reach_qf": pct("QF"),
            "reach_sf": pct("SF"),
            "reach_final": pct("Final"),
            "projected_final": {
                "matchup": [fa, fb], "probability": freq / n,
                "projected_score": f"{fa} {score['most_likely_score']} {fb}",
                "expected_goals": score["expected_goals"],
            },
        }
