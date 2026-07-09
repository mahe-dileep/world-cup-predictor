# ⚽ Touchline — World Cup 2026 match predictor

A leakage-free machine-learning pipeline that predicts international football
results, trained on ~150 years of matches and wrapped in a Streamlit dashboard
for the 2026 World Cup.

It produces **calibrated probabilities** for home win / draw / away win, projects
**exact scorelines** and **betting odds**, and simulates the **2026 World Cup**
from the group stage through the real fixed bracket.

---

## What it does

* Win / draw / loss probabilities for any matchup between the 48 finalists
* Most-likely scoreline and expected goals (a Poisson goal model fitted to the result probabilities)
* Betting markets — 1X2, over/under 2.5, both teams to score, correct score, Asian handicap — as fair and bookmaker-margin odds
* Monte Carlo: replay a single match up to 100,000 times and watch it converge on the model
* Full tournament simulation on the official 2026 bracket → champion probabilities and a projected final

---

## Data

| Source | Used for |
| --- | --- |
| International results, 1872–2026 (~49,500 matches) | Elo, rolling form, training labels |
| World Cup 2026 squads & fixtures | team universe, squad features, the real bracket |
| Team season statistics | team-level context (used in the app's Teams view) |
| Transfermarkt squad values | squad market value |

Team names across every source are reconciled through one canonical mapping
(`data/interim/team_name_mapping.csv`), so "USA / United States",
"Türkiye / Turkey", "Congo DR / DR Congo", "Cabo Verde / Cape Verde" and friends
never silently mismatch on a join.

---

## Pipeline

```text
data/raw/matches/results.csv
        │  clean
        ▼
cleaned_matches ──► Elo (pre-match) ──► elo_snapshots
        │                                    │
        ▼                                    ▼
rolling form (last 3 / 5 / 10)  +  matchup differences
        │
        ▼
matches_ml.csv  — 167 strictly pre-match features, one row per match
        │
        ▼
CatBoost · LightGBM · XGBoost   (temporal split, probability-calibrated)
        │
        ▼
inference layer  ──►  Touchline app
```

---

## No data leakage

Every feature for a match is built from information available **before kickoff only**:

* **Pre-match Elo** — each rating reflects results up to, but not including, that match.
* **Rolling form** over a team's previous 3 / 5 / 10 games — points per game, goals for and against, clean sheets, opponent strength, rest days, and fixture congestion.
* **Matchup differences** between the two sides.

Training uses a strict **temporal split** — train before 2015, validation 2015–2018,
test 2019–2025 — never a random shuffle. This was verified by independently
recomputing features on hundreds of sampled matches across every era and confirming
each one used only prior results.

---

## Models

Three gradient-boosted models predict the three-way result, each with early
stopping and a small hyper-parameter search. Evaluated on the 2019–2025 hold-out
(6,868 matches):

| Model | Accuracy | Log loss | Macro F1 |
| --- | --- | --- | --- |
| **CatBoost** (recommended) | 0.611 | 0.859 | 0.46 |
| LightGBM | 0.610 | 0.860 | 0.46 |
| XGBoost | 0.607 | 0.865 | 0.45 |

The models are already well-calibrated (expected calibration error ≈ 0.017), so
post-hoc calibration was tested and correctly skipped. `competition` is handled as
a native categorical feature. `elo_diff` is by far the strongest predictor,
followed by recent-form differences.

Outputs are **probabilities, not just a predicted winner** — which is what makes
the odds and the tournament simulation meaningful.

---

## Results & experiments

Reproduce everything with one command (writes `reports/experiments.md`):

```bash
python -m src.models.experiments
```

> Needs the full match dataset, which is gitignored because of its size. On a fresh
> clone, rebuild it first with the [Rebuild data & models](#rebuild-data--models)
> steps (the app itself runs without this).

**Did it learn anything?** Held-out test, 2019–2025 (6,868 matches):

| Model | Accuracy | Log loss |
| --- | --- | --- |
| Majority class (always Home) | 0.479 | 1.050 |
| Higher-Elo pick / Elo-only | 0.600 | 0.868 |
| **Softmax engine** (from scratch, 10 features) | 0.611 | 0.862 |
| CatBoost (production) | 0.612 | 0.859 |

A hand-written softmax (`src/models/softmax.py`, ~40 lines of numpy) **matches the
tuned gradient-boosted model** — so the signal is mostly linear in Elo + recent form,
and boosting adds only a hair. Both clearly beat the trivial baseline; both beat pure
Elo only slightly.

**Experiment 1 — does squad market value help?** Adding `squad_value_diff` to a
softmax that already has Elo + form (matches between the 48 finalists): **−0.003 log
loss** — essentially no help. Elo already encodes strength. (Market values are a
current snapshot, so this is a *lenient* test — and it still doesn't move.)

**Experiment 2 — weighted-K vs flat-K Elo**, judged on 1,049 World Cup finals
matches: the goal-margin **weighted-K wins by 0.008 log loss** (0.971 vs 0.979) — a
small but consistent edge, which is why the pipeline uses it.

---

## The app — Touchline

`app/streamlit_app.py`, five views:

* **Match** — win/draw/loss probabilities, most-likely scoreline, expected goals.
* **Odds board** — 1X2, over/under 2.5, both teams to score, correct score, Asian handicap (fair + bookmaker odds).
* **Monte Carlo** — replay a match and watch the simulation converge on the model.
* **World Cup 2026** — champion odds simulated from the group stage through the real fixed bracket, the projected final, and the live bracket state, plus a "run your own" simulator.
* **Teams** — all 48 finalists by Elo, title odds, squad value, and age.

### Run it locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

The model loads once and caches. Tournament numbers are precomputed into
`app/data/tournament.json`, and each team's current form is baked into
`data/processed/team_form_snapshot.json`, so the app starts fast and never reads
the large `matches_ml.csv` at runtime.

### Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Create an app pointing at `app/streamlit_app.py`.

Everything the app needs is committed (model artifacts, small snapshots, and the
World Cup fixture data). The heavy pipeline intermediates are gitignored because
they are regenerable and unused at runtime.

### Rebuild data & models

Only needed if you change the pipeline or retrain. Run from the repo root, in order:

```bash
python src/data/cleaning/clean_matches.py
python src/data/build_team_mapping.py
python src/features/elo.py
python src/features/elo_snapshots.py
python src/features/latest_elo.py
python src/features/squad_features.py
python src/features/form_features.py
python src/features/matchup_features.py
python src/features/merge_features.py     # -> data/processed/features.csv
python src/features/create_matches_ml.py  # -> data/processed/matches_ml.csv
python -m src.models.train                # -> models/*
python -m app.precompute                  # -> tournament.json + form snapshot
```

---

## Project structure

```text
football-predictor/
├── app/            # Streamlit dashboard + precompute
├── src/
│   ├── data/       # cleaning, canonical team mapping, collectors
│   ├── features/   # Elo, rolling form, matchup diffs, feature assembly
│   ├── models/     # training + ensemble
│   └── prediction/ # inference: predictor, odds, Monte Carlo, WC2026 sim
├── data/
│   ├── raw/        # source data
│   ├── interim/    # pipeline intermediates
│   └── processed/  # model-ready datasets + snapshots
├── models/         # trained models + metadata
└── requirements.txt
```

---

## Disclaimer

Football contains a significant amount of randomness. This project estimates
probabilities based on historical data and should not be interpreted as
guaranteeing future outcomes, nor as betting advice.
