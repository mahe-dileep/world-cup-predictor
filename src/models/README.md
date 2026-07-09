# Models

Production multiclass result predictor (H / D / A). Trained on the leakage-free
historical dataset `data/processed/matches_ml.csv`.

Run training from the repository root:

```bash
python -m src.models.train
```

Modules:
- `train.py`  — end-to-end training: temporal split, CatBoost/LightGBM/XGBoost
  with early stopping + small hyper-parameter search, probability calibration
  (applied only if it helps), weight-optimised soft-voting ensemble, evaluation,
  SHAP, and artifact saving. Deterministic (fixed seeds).
- `ensemble.py` — picklable wrappers reused at prediction time:
  `prepare_features`, `MulticlassCalibrator`, `CalibratedModel`,
  `SoftVotingEnsemble`.

Artifacts are written to the top-level `models/` directory (catboost.pkl,
lightgbm.pkl, xgboost.pkl, ensemble.pkl, label_encoder.pkl,
feature_columns.json, training_metadata.json, feature_importance.csv).

Prediction-time note: `matches_ml` team names are in the Elo/results namespace;
join `features.csv` / `squad_features.csv` only through
`data/interim/team_name_mapping.csv`.
