"""Reproducible evaluation: the baseline/softmax/CatBoost table + two experiments.

    python -m src.models.experiments

Writes a summary to reports/experiments.md and prints it. Uses the same strict
temporal split as training (train < 2015, test 2019-2025).
"""
from __future__ import annotations
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

from src.models.softmax import SoftmaxRegression
from src.features.elo import (expected_score, update_elo, goal_multiplier,
                              INITIAL_ELO, HOME_ADVANTAGE)

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
TRAIN_END, TEST_START, TEST_END = "2015-01-01", "2019-01-01", "2026-01-01"

# Compact, well-conditioned inputs for the linear softmax — a handful of pre-match
# difference features rather than all 167 (a linear model overfits the wide set).
SOFTMAX_FEATURES = ["elo_diff", "ppg_diff_10", "ppg_diff_5", "win_pct_diff_10",
                    "goals_scored_diff_10", "goals_conceded_diff_10",
                    "avg_opp_elo_diff_10", "goal_diff_diff_10", "rest_days_diff", "neutral"]


def _ll(y, P):
    return log_loss(y, P, labels=[0, 1, 2])


def load():
    ml = pd.read_csv(ROOT / "data/processed/matches_ml.csv", parse_dates=["date"])
    ml = ml[ml.home_goals.notna()].copy()
    fc = json.loads((ROOT / "models/feature_columns.json").read_text())["feature_columns"]
    le = joblib.load(ROOT / "models/label_encoder.pkl")
    ml["y"] = le.transform(ml.result)
    return ml, fc, list(le.classes_)


def table(ml, fc, classes):
    tr, te = ml.date < TRAIN_END, (ml.date >= TEST_START) & (ml.date < TEST_END)
    ytr, yte = ml.y[tr].values, ml.y[te].values
    Hi, Ai = classes.index("H"), classes.index("A")

    # compact difference-feature matrix for the linear softmax
    med = ml.loc[tr, SOFTMAX_FEATURES].median()
    X = ml[SOFTMAX_FEATURES].fillna(med)

    rows = []
    # majority-class / prior
    prior = np.bincount(ytr, minlength=3) / tr.sum()
    rows.append(("Majority class (always Home)",
                 accuracy_score(yte, np.full(te.sum(), prior.argmax())),
                 _ll(yte, np.tile(prior, (te.sum(), 1)))))
    # Elo-only
    elo_pick = np.where((ml.home_elo[te] > ml.away_elo[te]).values, Hi, Ai)
    lr = LogisticRegression(max_iter=3000, multi_class="multinomial").fit(ml.loc[tr, ["elo_diff"]], ytr)
    Pe = lr.predict_proba(ml.loc[te, ["elo_diff"]])
    rows.append(("Higher-Elo pick / Elo-only logit",
                 accuracy_score(yte, elo_pick), _ll(yte, Pe)))
    # OUR softmax engine
    sm = SoftmaxRegression().fit(X[tr].values, ytr, classes=[0, 1, 2])
    Ps = sm.predict_proba(X[te].values)
    rows.append(("Softmax engine (from scratch)",
                 accuracy_score(yte, Ps.argmax(1)), _ll(yte, Ps)))
    # CatBoost (production)
    cb = joblib.load(ROOT / "models/catboost.pkl")
    Pc = np.asarray(cb.predict_proba(ml.loc[te, fc]))
    rows.append(("CatBoost (production model)",
                 accuracy_score(yte, Pc.argmax(1)), _ll(yte, Pc)))
    return rows, int(te.sum())


def experiment1(ml, classes):
    """Does adding squad market value help a softmax that already has Elo + form?"""
    mp = pd.read_csv(ROOT / "data/interim/team_name_mapping.csv")
    feats = pd.read_csv(ROOT / "data/processed/features.csv")
    e2c = dict(zip(mp.elo_name, mp.worldcup_name))
    c2mv = dict(zip(feats.team, feats.squad_total_market_value))
    mv = lambda t: c2mv.get(e2c.get(t), np.nan)
    wc = set(mp.elo_name)

    sub = ml[ml.home_team.isin(wc) & ml.away_team.isin(wc)].copy()
    sub["hmv"], sub["amv"] = sub.home_team.map(mv), sub.away_team.map(mv)
    sub = sub.dropna(subset=["hmv", "amv"])
    sub["sqval_diff"] = np.log1p(sub.hmv) - np.log1p(sub.amv)   # log market-value gap

    base = ["elo_diff", "ppg_diff_10", "win_pct_diff_10",
            "goals_scored_diff_10", "goals_conceded_diff_10", "avg_opp_elo_diff_10"]
    tr, te = sub.date < TRAIN_END, (sub.date >= TEST_START) & (sub.date < TEST_END)
    ytr, yte = sub.y[tr].values, sub.y[te].values

    def run(cols):
        med = sub.loc[tr, cols].median()
        Xtr, Xte = sub.loc[tr, cols].fillna(med).values, sub.loc[te, cols].fillna(med).values
        P = SoftmaxRegression().fit(Xtr, ytr, classes=[0, 1, 2]).predict_proba(Xte)
        return accuracy_score(yte, P.argmax(1)), _ll(yte, P)

    a0, l0 = run(base)
    a1, l1 = run(base + ["sqval_diff"])
    return {"n_train": int(tr.sum()), "n_test": int(te.sum()),
            "base": (a0, l0), "plus_sqval": (a1, l1),
            "d_acc": a1 - a0, "d_ll": l1 - l0}


def experiment2(ml, classes):
    """Weighted-K (goal-margin) vs flat-K Elo, judged on tournament (World Cup) matches."""
    cm = pd.read_csv(ROOT / "data/interim/cleaned_matches.csv", parse_dates=["date"])
    cm = cm.sort_values(["date", "match_id"], kind="mergesort")

    def run_elo(weighted):
        elo, before = {}, {}
        for r in cm.itertuples(index=False):
            he, ae = elo.get(r.home_team, INITIAL_ELO), elo.get(r.away_team, INITIAL_ELO)
            before[r.match_id] = he - ae
            if pd.isna(r.home_goals):
                continue
            adj = he if r.neutral else he + HOME_ADVANTAGE
            ah, aa = {"H": (1, 0), "A": (0, 1), "D": (.5, .5)}[r.result]
            m = goal_multiplier(r.home_goals, r.away_goals) if weighted else 1.0
            elo[r.home_team] = update_elo(he, expected_score(adj, ae), ah, m)
            elo[r.away_team] = update_elo(ae, expected_score(ae, adj), aa, m)
        return before

    dw, df = run_elo(True), run_elo(False)
    le = joblib.load(ROOT / "models/label_encoder.pkl")
    played = cm[cm.home_goals.notna()].copy()
    played["y"] = le.transform(played.result)
    played["ew"] = played.match_id.map(dw)
    played["ef"] = played.match_id.map(df)
    wc = played[played.competition == "FIFA World Cup"]
    nonwc = played[played.competition != "FIFA World Cup"]

    def ev(col):
        lr = LogisticRegression(max_iter=3000, multi_class="multinomial").fit(nonwc[[col]], nonwc.y)
        P = lr.predict_proba(wc[[col]])
        return accuracy_score(wc.y, P.argmax(1)), _ll(wc.y, P)

    aw, lw = ev("ew")
    af, lf = ev("ef")
    return {"n_wc": len(wc), "weighted": (aw, lw), "flat": (af, lf),
            "d_acc": aw - af, "d_ll": lw - lf}


def main():
    ml, fc, classes = load()
    rows, n_te = table(ml, fc, classes)
    e1 = experiment1(ml, classes)
    e2 = experiment2(ml, classes)

    L = []
    L.append("# Results\n")
    L.append(f"Held-out test set: **2019–2025, {n_te:,} matches** (strict temporal split).\n")
    L.append("## Baselines vs models\n")
    L.append("| Model | Accuracy | Log loss |")
    L.append("| --- | --- | --- |")
    for name, a, l in rows:
        L.append(f"| {name} | {a:.3f} | {l:.3f} |")
    L.append("")
    L.append("## Experiment 1 — does squad market value help?\n")
    L.append(f"Softmax on Elo + form vs the same + `squad_value_diff`, on matches between "
             f"the 48 finalists ({e1['n_train']:,} train / {e1['n_test']:,} test). "
             f"Note: market values are a *current* snapshot used as a static team attribute.\n")
    L.append(f"| Features | Accuracy | Log loss |")
    L.append(f"| --- | --- | --- |")
    L.append(f"| Elo + form | {e1['base'][0]:.3f} | {e1['base'][1]:.3f} |")
    L.append(f"| + squad_value_diff | {e1['plus_sqval'][0]:.3f} | {e1['plus_sqval'][1]:.3f} |")
    L.append(f"\n**Delta: {e1['d_acc']:+.3f} accuracy, {e1['d_ll']:+.3f} log loss** "
             f"(negative log loss = better).\n")
    L.append("## Experiment 2 — weighted-K vs flat-K Elo (tournament matches)\n")
    L.append(f"Logistic on Elo difference, trained on non-World-Cup matches, evaluated on "
             f"all {e2['n_wc']:,} World Cup finals matches.\n")
    L.append(f"| Elo variant | Accuracy | Log loss |")
    L.append(f"| --- | --- | --- |")
    L.append(f"| Weighted-K (goal margin) | {e2['weighted'][0]:.3f} | {e2['weighted'][1]:.3f} |")
    L.append(f"| Flat-K | {e2['flat'][0]:.3f} | {e2['flat'][1]:.3f} |")
    L.append(f"\n**Delta (weighted − flat): {e2['d_acc']:+.3f} accuracy, {e2['d_ll']:+.3f} log loss.**\n")

    out = "\n".join(L)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "experiments.md").write_text(out)
    print(out)


if __name__ == "__main__":
    main()
