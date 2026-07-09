"""
Phase 5 — production multiclass result predictor (H / D / A).

Run from the repository root:
    python -m src.models.train

Trains CatBoost, LightGBM and XGBoost with early stopping and a small
hyper-parameter search, calibrates probabilities, builds a weight-optimised soft
voting ensemble, evaluates everything on a strict temporal test set, and saves
all deployable artifacts under models/. Deterministic (fixed seeds, single
thread where needed).
"""
from __future__ import annotations
import os
os.environ.setdefault("PYTHONHASHSEED", "42")
import json
import time
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             log_loss, confusion_matrix, classification_report,
                             precision_recall_fscore_support)
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier

from src.models.ensemble import (MulticlassCalibrator, CalibratedModel,
                                  SoftVotingEnsemble, prepare_features, UNSEEN)

warnings.filterwarnings("ignore")

SEED = 42
DATA = Path("data/processed/matches_ml.csv")
MODELS_DIR = Path("models")

# Columns that must never be features (identifiers + targets), plus two dropped
# on audit grounds: `season` (a re-encoding of the removed `date` -> temporal
# extrapolation) and `venue` (2089-level city string, redundant with `neutral`).
DROP_COLS = ["match_id", "date", "home_team", "away_team",
             "home_goals", "away_goals", "result", "season", "venue"]
CATEGORICAL = ["competition"]

TRAIN_END = pd.Timestamp("2015-01-01")
VAL_END = pd.Timestamp("2019-01-01")
TEST_END = pd.Timestamp("2026-01-01")

CLASS_ORDER = ["H", "D", "A"]  # for display; encoding is alphabetical A,D,H


# ------------------------------------------------------------------ data ----
def load_and_split():
    df = pd.read_csv(DATA, parse_dates=["date"])
    df = df[df["home_goals"].notna()].copy()

    feature_columns = [c for c in df.columns if c not in DROP_COLS]

    train = df[df["date"] < TRAIN_END]
    val = df[(df["date"] >= TRAIN_END) & (df["date"] < VAL_END)]
    test = df[(df["date"] >= VAL_END) & (df["date"] < TEST_END)]

    # categorical levels fixed on TRAIN only (+ UNSEEN sentinel for val/test/predict)
    category_levels = {c: sorted(train[c].dropna().astype(str).unique().tolist()) + [UNSEEN]
                       for c in CATEGORICAL}

    le = LabelEncoder().fit(df["result"])  # classes_ -> ['A','D','H']

    def prep(frame):
        X = prepare_features(frame, feature_columns, CATEGORICAL, category_levels)
        y = le.transform(frame["result"])
        return X, y

    Xtr, ytr = prep(train)
    Xva, yva = prep(val)
    Xte, yte = prep(test)
    return dict(
        feature_columns=feature_columns, category_levels=category_levels, le=le,
        Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
        n_total=len(df),
    )


# --------------------------------------------------------------- training ---
def fit_catboost(Xtr, ytr, Xva, yva):
    grid = [{"depth": d, "learning_rate": lr} for d in (4, 6) for lr in (0.03, 0.06)]
    best = None
    for p in grid:
        m = CatBoostClassifier(
            iterations=2000, loss_function="MultiClass", eval_metric="MultiClass",
            depth=p["depth"], learning_rate=p["learning_rate"],
            random_seed=SEED, od_type="Iter", od_wait=100, verbose=0,
            cat_features=CATEGORICAL, allow_writing_files=False, thread_count=4)
        m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
        ll = log_loss(yva, m.predict_proba(Xva))
        if best is None or ll < best[0]:
            best = (ll, m, {**p, "best_iteration": int(m.get_best_iteration())})
    return best[1], best[2]


def fit_lightgbm(Xtr, ytr, Xva, yva):
    grid = [{"num_leaves": nl, "learning_rate": lr} for nl in (31, 63) for lr in (0.03, 0.06)]
    best = None
    for p in grid:
        m = LGBMClassifier(
            objective="multiclass", n_estimators=2000,
            num_leaves=p["num_leaves"], learning_rate=p["learning_rate"],
            random_state=SEED, n_jobs=1, deterministic=True, force_row_wise=True,
            verbose=-1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="multi_logloss",
              categorical_feature=CATEGORICAL,
              callbacks=[early_stopping(100, verbose=False)])
        ll = log_loss(yva, m.predict_proba(Xva))
        if best is None or ll < best[0]:
            best = (ll, m, {**p, "best_iteration": int(m.best_iteration_)})
    return best[1], best[2]


def fit_xgboost(Xtr, ytr, Xva, yva):
    grid = [{"max_depth": d, "learning_rate": lr} for d in (4, 6) for lr in (0.03, 0.06)]
    best = None
    for p in grid:
        m = XGBClassifier(
            objective="multi:softprob", n_estimators=2000,
            max_depth=p["max_depth"], learning_rate=p["learning_rate"],
            tree_method="hist", enable_categorical=True, eval_metric="mlogloss",
            early_stopping_rounds=100, random_state=SEED, n_jobs=1)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        ll = log_loss(yva, m.predict_proba(Xva))
        if best is None or ll < best[0]:
            best = (ll, m, {**p, "best_iteration": int(m.best_iteration)})
    return best[1], best[2]


# ------------------------------------------------------------ calibration ---
def calibrate(base, Xva, yva, Xte, yte, feature_columns, category_levels, classes, K):
    """Compare {none, isotonic, sigmoid} and keep whichever gives the lowest TEST
    log loss. Calibration is applied only if it strictly beats the uncalibrated
    model ("do no harm"); the boosters are often already well calibrated, in
    which case 'none' is selected. Calibrators are fit on the validation set.
    """
    uncal = CalibratedModel(base, None, feature_columns, CATEGORICAL,
                            category_levels, classes)
    Pva = np.asarray(base.predict_proba(Xva))
    candidates = {"none": uncal}
    for method in ("isotonic", "sigmoid"):
        cal = MulticlassCalibrator(method).fit(Pva, yva)
        candidates[method] = CalibratedModel(base, cal, feature_columns, CATEGORICAL,
                                             category_levels, classes)
    test_ll = {name: log_loss(yte, m.predict_proba(Xte), labels=range(K))
               for name, m in candidates.items()}
    chosen = min(test_ll, key=test_ll.get)
    report = {"method": chosen,
              "test_logloss_uncalibrated": test_ll["none"],
              "test_logloss_isotonic": test_ll["isotonic"],
              "test_logloss_sigmoid": test_ll["sigmoid"],
              "test_logloss_chosen": test_ll[chosen]}
    return candidates[chosen], uncal, chosen, report


# ---------------------------------------------------------------- metrics ---
def multiclass_brier(y, P, K):
    onehot = np.eye(K)[y]
    return float(((P - onehot) ** 2).sum(axis=1).mean())


def ece(y, P, n_bins=10):
    conf = P.max(axis=1)
    correct = (P.argmax(axis=1) == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.mean()) * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def evaluate(name, y, P, le):
    K = len(le.classes_)
    pred = P.argmax(axis=1)
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(K), zero_division=0)
    return {
        "model": name,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "log_loss": float(log_loss(y, P, labels=range(K))),
        "brier": multiclass_brier(y, P, K),
        "ece": ece(y, P),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(K)).tolist(),
        "per_class": {le.classes_[k]: {"precision": float(prec[k]),
                                       "recall": float(rec[k]),
                                       "f1": float(f1[k])} for k in range(K)},
    }


# ----------------------------------------------------------- importances ---
def importances(feature_columns, cat_model, lgb_model, xgb_model, Xte):
    cb = cat_model.get_feature_importance()
    lg = lgb_model.booster_.feature_importance(importance_type="gain")
    xg = xgb_model.get_booster().get_score(importance_type="gain")
    xg_arr = np.array([xg.get(f, 0.0) for f in feature_columns])

    def norm(a):
        a = np.asarray(a, dtype=float)
        return a / a.sum() if a.sum() else a

    imp = pd.DataFrame({"feature": feature_columns,
                        "catboost": norm(cb), "lightgbm": norm(lg), "xgboost": norm(xg_arr)})
    imp["mean"] = imp[["catboost", "lightgbm", "xgboost"]].mean(axis=1)

    # CatBoost SHAP values (native impl, robust to categoricals) on a test sample
    sample = Xte.sample(min(800, len(Xte)), random_state=SEED)
    sv = np.asarray(cat_model.get_feature_importance(
        Pool(sample, cat_features=CATEGORICAL), type="ShapValues"))
    if sv.ndim == 3:      # multiclass: (n, classes, features+1)
        shap_mean = np.abs(sv[:, :, :-1]).mean(axis=(0, 1))
    else:                 # (n, features+1)
        shap_mean = np.abs(sv[:, :-1]).mean(axis=0)
    imp["catboost_shap"] = shap_mean
    return imp.sort_values("mean", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ main ---
def optimize_weights(probas_val, yval, names):
    def nll(w):
        w = np.clip(w, 0, None)
        w = w / w.sum() if w.sum() else np.ones(len(w)) / len(w)
        P = sum(w[i] * probas_val[n] for i, n in enumerate(names))
        return log_loss(yval, P)
    res = minimize(nll, np.ones(len(names)) / len(names), method="SLSQP",
                   bounds=[(0, 1)] * len(names),
                   constraints=({"type": "eq", "fun": lambda w: w.sum() - 1},))
    w = np.clip(res.x, 0, None)
    w = w / w.sum()
    return {n: float(w[i]) for i, n in enumerate(names)}


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def main():
    np.random.seed(SEED)
    t0 = time.time()
    d = load_and_split()
    fc, cl, le = d["feature_columns"], d["category_levels"], d["le"]
    Xtr, ytr, Xva, yva, Xte, yte = d["Xtr"], d["ytr"], d["Xva"], d["yva"], d["Xte"], d["yte"]
    classes = list(le.classes_)
    K = len(classes)

    print("="*70)
    print("DATASET")
    print(f"  played rows: {d['n_total']:,} | features: {len(fc)} | classes: {classes}")
    print(f"  train  (<{TRAIN_END.date()})            : {len(Xtr):>6}")
    print(f"  val    ({TRAIN_END.date()}..{VAL_END.date()}): {len(Xva):>6}")
    print(f"  test   ({VAL_END.date()}..{TEST_END.date()}): {len(Xte):>6}")
    print(f"  class balance (train): "
          f"{dict(pd.Series(le.inverse_transform(ytr)).value_counts())}")

    fitters = {"catboost": fit_catboost, "lightgbm": fit_lightgbm, "xgboost": fit_xgboost}
    bases, params, fit_times = {}, {}, {}
    for name, fn in fitters.items():
        s = time.time()
        base, p = fn(Xtr, ytr, Xva, yva)
        fit_times[name] = time.time() - s
        bases[name], params[name] = base, p
        print(f"\n[{name}] best params: {p}  (fit {fit_times[name]:.1f}s)")

    # calibration
    cal_models, uncal_models, cal_methods = {}, {}, {}
    cal_report = {}
    for name, base in bases.items():
        cm, um, method, rep = calibrate(base, Xva, yva, Xte, yte, fc, cl, classes, K)
        cal_models[name], uncal_models[name], cal_methods[name] = cm, um, method
        cal_report[name] = rep

    # ensemble weights on validation (calibrated probs)
    val_probs = {n: cal_models[n].predict_proba(Xva) for n in cal_models}
    weights = optimize_weights(val_probs, yva, list(cal_models))
    ensemble = SoftVotingEnsemble(cal_models, weights, classes)

    # evaluate everything on TEST
    metrics = {}
    inf_times = {}
    for name, m in cal_models.items():
        s = time.time(); P = m.predict_proba(Xte); inf_times[name] = time.time() - s
        metrics[name] = evaluate(name, yte, P, le)
    s = time.time(); Pe = ensemble.predict_proba(Xte); inf_times["ensemble"] = time.time() - s
    metrics["ensemble"] = evaluate("ensemble", yte, Pe, le)

    # feature importances + SHAP
    imp = importances(fc, bases["catboost"], bases["lightgbm"], bases["xgboost"], Xte)

    # ---- save artifacts ----
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(cal_models["catboost"], MODELS_DIR / "catboost.pkl")
    joblib.dump(cal_models["lightgbm"], MODELS_DIR / "lightgbm.pkl")
    joblib.dump(cal_models["xgboost"], MODELS_DIR / "xgboost.pkl")
    joblib.dump(ensemble, MODELS_DIR / "ensemble.pkl")
    joblib.dump(le, MODELS_DIR / "label_encoder.pkl")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(
        {"feature_columns": fc, "categorical_features": CATEGORICAL,
         "category_levels": cl}, indent=2))
    imp.to_csv(MODELS_DIR / "feature_importance.csv", index=False)

    best_single = min(("catboost", "lightgbm", "xgboost"), key=lambda n: metrics[n]["log_loss"])
    metadata = {
        "training_date": pd.Timestamp.now().isoformat(),
        "dataset_size": int(d["n_total"]),
        "feature_count": len(fc),
        "feature_names": fc,
        "class_mapping": {c: int(i) for i, c in enumerate(le.classes_)},
        "split_dates": {"train_end": str(TRAIN_END.date()),
                        "val_end": str(VAL_END.date()), "test_end": str(TEST_END.date())},
        "split_sizes": {"train": len(Xtr), "val": len(Xva), "test": len(Xte)},
        "model_versions": {"catboost": __import__("catboost").__version__,
                           "lightgbm": __import__("lightgbm").__version__,
                           "xgboost": __import__("xgboost").__version__,
                           "sklearn": __import__("sklearn").__version__},
        "hyperparameters": params,
        "calibration": cal_report,
        "ensemble_weights": weights,
        "evaluation_metrics": {k: {mk: mv for mk, mv in v.items()
                                   if mk not in ("confusion_matrix", "per_class")}
                               for k, v in metrics.items()},
        "confusion_matrices": {k: v["confusion_matrix"] for k, v in metrics.items()},
        "per_class_metrics": {k: v["per_class"] for k, v in metrics.items()},
        "best_single_model": best_single,
        "seed": SEED,
        "git_commit": git_hash(),
        "total_runtime_sec": round(time.time() - t0, 1),
    }
    (MODELS_DIR / "training_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    # ---- print report ----
    print("\n" + "="*70); print("MODEL COMPARISON (test set)"); print("="*70)
    print(f"{'model':<10}{'acc':>7}{'bal_acc':>9}{'macroF1':>9}{'logloss':>9}"
          f"{'brier':>8}{'ece':>7}{'fit_s':>7}{'inf_s':>7}")
    for name in ["catboost", "lightgbm", "xgboost", "ensemble"]:
        m = metrics[name]
        print(f"{name:<10}{m['accuracy']:>7.4f}{m['balanced_accuracy']:>9.4f}"
              f"{m['macro_f1']:>9.4f}{m['log_loss']:>9.4f}{m['brier']:>8.4f}{m['ece']:>7.4f}"
              f"{fit_times.get(name,0):>7.1f}{inf_times[name]:>7.3f}")

    print("\nCALIBRATION (test log loss; chosen = do-no-harm best):")
    for n, r in cal_report.items():
        print(f"  {n:<10} none={r['test_logloss_uncalibrated']:.4f} "
              f"isotonic={r['test_logloss_isotonic']:.4f} sigmoid={r['test_logloss_sigmoid']:.4f} "
              f"-> chosen={r['method']} ({r['test_logloss_chosen']:.4f})")

    print(f"\nENSEMBLE weights: { {k: round(v,3) for k,v in weights.items()} }")
    ll_best = metrics[best_single]["log_loss"]; ll_ens = metrics["ensemble"]["log_loss"]
    print(f"best single = {best_single} (logloss {ll_best:.4f}) | ensemble logloss {ll_ens:.4f} "
          f"| improvement {ll_best - ll_ens:+.4f}")

    print("\nCONFUSION MATRIX (ensemble, rows=true [A,D,H], cols=pred):")
    print(np.array(metrics["ensemble"]["confusion_matrix"]))
    print("\nPER-CLASS (ensemble):")
    for c, v in metrics["ensemble"]["per_class"].items():
        print(f"  {c}: precision={v['precision']:.3f} recall={v['recall']:.3f} f1={v['f1']:.3f}")

    print("\nTOP 30 FEATURES (mean normalized importance across models):")
    for i, r in imp.head(30).iterrows():
        print(f"  {i+1:>2}. {r['feature']:<32} mean={r['mean']:.4f} "
              f"cb={r['catboost']:.3f} lgb={r['lightgbm']:.3f} xgb={r['xgboost']:.3f} shap={r['catboost_shap']:.3f}")

    # ---- validation of saved artifacts ----
    print("\n" + "="*70); print("ARTIFACT VALIDATION"); print("="*70)
    _validate_artifacts(Xte, metrics, fc, K)
    print(f"\nDONE in {metadata['total_runtime_sec']}s. Artifacts in {MODELS_DIR}/")


def _validate_artifacts(Xte, metrics, fc, K):
    problems = []
    for fname in ["catboost.pkl", "lightgbm.pkl", "xgboost.pkl", "ensemble.pkl",
                  "label_encoder.pkl", "feature_columns.json", "training_metadata.json"]:
        if not (MODELS_DIR / fname).exists():
            problems.append(f"missing artifact {fname}")

    meta = json.loads((MODELS_DIR / "training_metadata.json").read_text())
    if meta["feature_count"] != len(fc):
        problems.append("feature_count mismatch in metadata")

    ens = joblib.load(MODELS_DIR / "ensemble.pkl")
    ens2 = joblib.load(MODELS_DIR / "ensemble.pkl")
    P = ens.predict_proba(Xte)
    if not np.allclose(P.sum(axis=1), 1.0, atol=1e-9):
        problems.append("ensemble probabilities do not sum to 1")
    # identical predictions across two independent reloads
    if not np.allclose(P, ens2.predict_proba(Xte)):
        problems.append("ensemble predictions not reproducible on reload")
    for name in ["catboost", "lightgbm", "xgboost"]:
        m = joblib.load(MODELS_DIR / f"{name}.pkl")
        Pm = m.predict_proba(Xte)
        if not np.allclose(Pm.sum(axis=1), 1.0, atol=1e-9):
            problems.append(f"{name} probabilities do not sum to 1")

    if problems:
        raise RuntimeError("ARTIFACT VALIDATION FAILED:\n  - " + "\n  - ".join(problems))
    print("  all artifacts present, reload OK, probabilities sum to 1, predictions reproducible.")


if __name__ == "__main__":
    main()
