"""
Reusable, picklable model wrappers for the football result predictor.

Kept in their own importable module so that artifacts saved with joblib
(catboost.pkl, lightgbm.pkl, xgboost.pkl, ensemble.pkl) can be reloaded from
anywhere the project is on sys.path (training, API, simulation).

Contents:
    prepare_features     - deterministic X preparation shared by every wrapper
    MulticlassCalibrator - probability calibration operating on prob arrays only
    CalibratedModel      - base gradient-booster + optional calibrator
    SoftVotingEnsemble   - weighted average of several CalibratedModels
"""
from __future__ import annotations
import numpy as np
import pandas as pd


UNSEEN = "__OTHER__"  # sentinel level for categories unseen at training time


def prepare_features(X, feature_columns, categorical_features, category_levels):
    """Select/order columns, cast bools to int, pin categorical levels.

    Categorical levels are fixed at TRAIN time (and include the UNSEEN sentinel).
    Any value not seen in training is mapped to UNSEEN rather than to NaN, so no
    NaN ever appears in a categorical column (CatBoost forbids NaN categoricals)
    and train-time / predict-time encodings are guaranteed identical.
    """
    X = X.loc[:, list(feature_columns)].copy()
    for col in X.columns:
        if X[col].dtype == bool:
            X[col] = X[col].astype(int)
    for col in categorical_features:
        levels = list(category_levels[col])
        vals = X[col].astype(str)
        if UNSEEN in levels:
            vals = vals.where(vals.isin(levels), UNSEEN)
        X[col] = pd.Categorical(vals, categories=levels)
    return X


class MulticlassCalibrator:
    """One-vs-rest probability calibration with renormalisation.

    Operates purely on predicted-probability matrices (no feature matrix), so it
    is agnostic to categorical dtypes and works identically for every booster.
    """

    def __init__(self, method="isotonic"):
        assert method in ("isotonic", "sigmoid")
        self.method = method
        self.calibrators = []
        self.n_classes_ = None

    def fit(self, P, y):
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression

        P = np.asarray(P)
        self.n_classes_ = P.shape[1]
        self.calibrators = []
        for k in range(self.n_classes_):
            yk = (np.asarray(y) == k).astype(int)
            if self.method == "isotonic":
                c = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                c.fit(P[:, k], yk)
            else:  # Platt scaling
                c = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                c.fit(P[:, k].reshape(-1, 1), yk)
            self.calibrators.append(c)
        return self

    def transform(self, P):
        P = np.asarray(P)
        out = np.zeros_like(P, dtype=float)
        for k, c in enumerate(self.calibrators):
            if self.method == "isotonic":
                out[:, k] = c.predict(P[:, k])
            else:
                out[:, k] = c.predict_proba(P[:, k].reshape(-1, 1))[:, 1]
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


class CalibratedModel:
    """A fitted booster plus an optional probability calibrator."""

    def __init__(self, base, calibrator, feature_columns, categorical_features,
                 category_levels, classes):
        self.base = base
        self.calibrator = calibrator
        self.feature_columns = list(feature_columns)
        self.categorical_features = list(categorical_features)
        self.category_levels = category_levels
        self.classes_ = np.asarray(classes)

    def _prep(self, X):
        return prepare_features(X, self.feature_columns,
                                self.categorical_features, self.category_levels)

    def predict_proba(self, X):
        P = np.asarray(self.base.predict_proba(self._prep(X)))
        if P.ndim == 1:  # defensive: some APIs squeeze binary
            P = np.column_stack([1 - P, P])
        return self.calibrator.transform(P) if self.calibrator is not None else P

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SoftVotingEnsemble:
    """Weighted average of the probability outputs of several CalibratedModels."""

    def __init__(self, models, weights, classes):
        self.models = models              # dict name -> CalibratedModel
        self.weights = dict(weights)      # dict name -> float
        self.classes_ = np.asarray(classes)

    def predict_proba(self, X):
        total = float(sum(self.weights.values()))
        proba = None
        for name, model in self.models.items():
            p = model.predict_proba(X) * self.weights[name]
            proba = p if proba is None else proba + p
        return proba / total

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
