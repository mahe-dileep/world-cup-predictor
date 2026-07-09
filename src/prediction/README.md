# Prediction engine (inference only)

Sits between the trained CatBoost model (`models/`) and any future API/UI.
No training, no feature-engineering changes — pure inference.

Modules:
- `loaders.py` — cached loading of every artifact / reference table (load once).
- `validation.py` — typed exceptions + model-contract checks (fail loudly).
- `utils.py` — seeding, probability helpers, `canonical_team_name`, `safe_json`.
- `feature_builder.py` — reconstructs the exact 167-column training vector for any
  matchup; bridges namespaces only through `team_name_mapping` (elo_name ↔
  worldcup_name); rolling-form comes from each team's latest `matches_ml` snapshot.
- `predictor.py` — `Predictor.predict_match`, `predict_proba`, `predict_many`,
  `predict_score` (exact scoreline), `predict_odds` (betting odds).
- `scoreline.py` — Poisson goal model fitted to the result probabilities.
- `betting.py` — fair + bookmaker (margin) odds, decimal & American.
- `simulator.py` — seeded Monte Carlo (`simulate_match/_fixture_list/_group/_knockout`).
- `tournament.py` — `WorldCupSimulator.run()` → champion probabilities.

Example:
```python
from src.prediction import Predictor
p = Predictor()
p.predict_match("Brazil", "Germany")
# {'home_team': 'Brazil', 'away_team': 'Germany',
#  'home_win': 0.53, 'draw': 0.20, 'away_win': 0.27, 'predicted_result': 'H'}
```

Aliases resolve automatically (USA↔United States, Türkiye↔Turkey, Congo DR↔DR Congo, …).
