"""
Machine-learning module (FR-04): candidate models, evaluation, tuning,
decision threshold, and the saved model bundle used by the dashboard.
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config as cfg
from data_prep import augment_minority, clean_readings, to_feature_frame
from deep_learning import build_mlp


# --------------------------------------------------------------- candidates
def build_candidates(seed: int = cfg.RANDOM_STATE, fast: bool = False) -> dict:
    """Every model compared in the project. 'deployable' = gives probabilities."""
    trees = 100 if fast else 200
    return {
        "Majority baseline": dict(
            model=DummyClassifier(strategy="most_frequent"),
            augment=False, deployable=False),
        "Logistic Regression": dict(
            model=Pipeline([("scale", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))]),
            augment=False, deployable=True),
        "Random Forest": dict(
            model=RandomForestClassifier(n_estimators=trees, min_samples_leaf=3,
                                         class_weight="balanced_subsample",
                                         n_jobs=-1, random_state=seed),
            augment=False, deployable=True),
        "Random Forest + augmentation": dict(
            model=RandomForestClassifier(n_estimators=trees, min_samples_leaf=3,
                                         n_jobs=-1, random_state=seed),
            augment=True, deployable=True),
        "SVM (RBF)": dict(
            model=Pipeline([("scale", StandardScaler()),
                            ("clf", SVC(C=10, gamma="scale", random_state=seed))]),
            augment=False, deployable=False),
        "Gradient Boosting": dict(
            model=HistGradientBoostingClassifier(max_iter=150 if fast else 250,
                                                 learning_rate=0.1, class_weight="balanced",
                                                 early_stopping=False, random_state=seed),
            augment=False, deployable=True),
        "Neural Network (MLP)": dict(
            model=build_mlp(seed), augment=True, deployable=True),
    }


def fit_candidate(spec: dict, X: pd.DataFrame, y, seed: int = cfg.RANDOM_STATE):
    """Fit a fresh copy of a candidate (augmenting the training set if asked)."""
    model = clone(spec["model"])
    if spec.get("augment"):
        X, y = augment_minority(X, pd.Series(np.asarray(y), index=X.index), seed=seed)
    model.fit(X, y)
    return model


# --------------------------------------------------------------- evaluation
def apply_bulb_threshold(proba: np.ndarray, classes, threshold: float,
                         label: int = cfg.BULB_FAILURE) -> np.ndarray:
    """
    Predict Bulb Failure whenever P(bulb failure) >= threshold; otherwise the
    most probable of the other classes. Lets us trade precision for recall on
    the hardest class.
    """
    classes = np.asarray(classes)
    bulb_col = int(np.where(classes == label)[0][0])
    other = [i for i in range(len(classes)) if i != bulb_col]
    preds = classes[other][proba[:, other].argmax(axis=1)]
    preds[proba[:, bulb_col] >= threshold] = label
    return preds


def predict_labels(model, X, threshold=None):
    """Return (labels, proba or None) using argmax or the bulb threshold."""
    proba = model.predict_proba(X) if hasattr(model, "predict_proba") else None
    if threshold is not None and proba is not None:
        return apply_bulb_threshold(proba, model.classes_, threshold), proba
    return np.asarray(model.predict(X)), proba


def metrics_dict(y_true, y_pred, proba=None, classes=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=cfg.FAULT_LABELS, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "per_class": {
            cfg.FAULT_NAMES[k]: {"precision": float(p[i]), "recall": float(r[i]),
                                 "f1": float(f[i]), "support": int(s[i])}
            for i, k in enumerate(cfg.FAULT_LABELS)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=cfg.FAULT_LABELS).tolist(),
    }
    if proba is not None and classes is not None and len(np.unique(y_true)) == len(classes):
        try:
            out["roc_auc_macro_ovr"] = float(roc_auc_score(
                y_true, proba, multi_class="ovr", average="macro", labels=list(classes)))
        except ValueError:
            pass
    return out


def evaluate_model(model, X, y, threshold=None) -> dict:
    labels, proba = predict_labels(model, X, threshold)
    return metrics_dict(y, labels, proba, getattr(model, "classes_", None))


def compare_models(candidates: dict, X_train, y_train, X_val, y_val,
                   seed: int = cfg.RANDOM_STATE, log=print):
    """Fit every candidate on the training split and score on validation."""
    fitted, rows = {}, []
    for name, spec in candidates.items():
        start = time.time()
        model = fit_candidate(spec, X_train, y_train, seed)
        seconds = time.time() - start
        m = evaluate_model(model, X_val, y_val)
        fitted[name] = model
        rows.append({
            "model": name, "fit_seconds": round(seconds, 1),
            "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"],
            "val_macro_f1": m["macro_f1"], "val_mcc": m["mcc"],
            "val_bulb_recall": m["per_class"]["Bulb Failure"]["recall"],
            "val_bulb_precision": m["per_class"]["Bulb Failure"]["precision"],
            "deployable": spec["deployable"],
        })
        log(f"   {name:<30} acc={m['accuracy']:.4f}  macroF1={m['macro_f1']:.4f}  ({seconds:.1f}s)")
    return fitted, pd.DataFrame(rows)


def cross_validate_candidates(candidates: dict, names, X, y, n_splits=5,
                              seed: int = cfg.RANDOM_STATE, log=print) -> pd.DataFrame:
    """Stratified k-fold on train+val: mean and spread of accuracy / macro F1."""
    y = np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name in names:
        accs, f1s = [], []
        for tr, te in skf.split(X, y):
            model = fit_candidate(candidates[name], X.iloc[tr], y[tr], seed)
            pred = model.predict(X.iloc[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        rows.append({"model": name, "cv_folds": n_splits,
                     "cv_accuracy_mean": float(np.mean(accs)), "cv_accuracy_std": float(np.std(accs)),
                     "cv_macro_f1_mean": float(np.mean(f1s)), "cv_macro_f1_std": float(np.std(f1s))})
        log(f"   {name:<30} CV macroF1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    return pd.DataFrame(rows)


def tune_random_forest(X_train, y_train, seed: int = cfg.RANDOM_STATE,
                       n_iter: int = 10, cv: int = 3):
    """Randomised hyper-parameter search scored on macro F1."""
    space = {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 12, 16, 24, None],
        "min_samples_leaf": [1, 2, 3, 5, 8],
        "max_features": ["sqrt", 0.5, 1.0],
        "class_weight": ["balanced", "balanced_subsample",
                         {0: 1, 1: 1, 2: 1, 3: 3, 4: 1}],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, random_state=seed), space, n_iter=n_iter,
        scoring="f1_macro", cv=StratifiedKFold(cv, shuffle=True, random_state=seed),
        random_state=seed, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    results = pd.DataFrame(search.cv_results_)[
        ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    ].sort_values("rank_test_score")
    results["params"] = results["params"].astype(str)
    return search.best_estimator_, search.best_params_, results


# ---------------------------------------------------------------- threshold
def sweep_threshold(model, X_val, y_val, thresholds=None) -> pd.DataFrame:
    thresholds = np.round(np.arange(0.10, 0.61, 0.05), 2) if thresholds is None else thresholds
    proba = model.predict_proba(X_val)
    rows = []
    base = np.asarray(model.predict(X_val))
    for thr in [None] + list(thresholds):
        pred = base if thr is None else apply_bulb_threshold(proba, model.classes_, thr)
        m = metrics_dict(y_val, pred)
        b = m["per_class"]["Bulb Failure"]
        rows.append({"threshold": np.nan if thr is None else float(thr),
                     "rule": "argmax" if thr is None else "bulb threshold",
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "bulb_precision": b["precision"], "bulb_recall": b["recall"],
                     "bulb_f1": b["f1"]})
    return pd.DataFrame(rows)


def select_threshold(sweep: pd.DataFrame, min_gain: float = 0.01, max_accuracy_loss: float = 0.02):
    """
    Use a bulb threshold only if it clearly helps: macro F1 must rise by at
    least `min_gain` AND accuracy may fall by at most `max_accuracy_loss`
    (every extra false bulb alarm sends a crew out for nothing). Otherwise
    return None, meaning plain argmax.
    """
    base = sweep[sweep["rule"] == "argmax"].iloc[0]
    cand = sweep[(sweep["rule"] == "bulb threshold")
                 & (sweep["macro_f1"] >= base["macro_f1"] + min_gain)
                 & (sweep["accuracy"] >= base["accuracy"] - max_accuracy_loss)]
    if cand.empty:
        return None
    return float(cand.sort_values(["macro_f1", "bulb_f1"], ascending=False).iloc[0]["threshold"])


# -------------------------------------------------------------- explanation
def feature_importance(model, X, y, seed: int = cfg.RANDOM_STATE, max_rows: int = 3000):
    y = pd.Series(np.asarray(y), index=X.index)
    if len(X) > max_rows:
        X = X.sample(max_rows, random_state=seed)
        y = y.loc[X.index]
    res = permutation_importance(model, X, y, scoring="f1_macro", n_repeats=5,
                                 random_state=seed, n_jobs=1)
    out = pd.DataFrame({"feature": X.columns, "importance_mean": res.importances_mean,
                        "importance_std": res.importances_std})
    return out.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------ bundle / use
def save_bundle(bundle: dict, path=None) -> float:
    """Compressed save. Returns the file size in MB."""
    path = path or cfg.MODEL_FILE
    bundle = dict(bundle)
    bundle["sklearn_version"] = sklearn.__version__
    bundle["saved_at"] = datetime.now().isoformat(timespec="seconds")
    joblib.dump(bundle, path, compress=3)
    return Path(path).stat().st_size / (1024 * 1024)


def load_bundle(path=None) -> dict:
    bundle = joblib.load(path or cfg.MODEL_FILE)
    for key in ("model", "features", "weather_mapping", "model_name"):
        if key not in bundle:
            raise ValueError(f"Model file is missing '{key}'. Re-run run_pipeline.py.")
    return bundle


def score_readings(raw: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """
    Classify raw readings (weather as text). Returns a tidy results table with
    asset id, readings, predicted fault and model confidence.
    """
    raw = raw.reset_index(drop=True)
    X = to_feature_frame(raw)[bundle["features"]]
    labels, proba = predict_labels(bundle["model"], X, bundle.get("bulb_threshold"))
    classes = list(bundle["model"].classes_)
    if proba is not None:
        conf = np.array([proba[i, classes.index(int(l))] for i, l in enumerate(labels)])
    else:
        conf = np.full(len(labels), np.nan)

    out = pd.DataFrame({
        "asset_id": [cfg.asset_id(n) for n in raw[cfg.COL_ASSET]],
        "bulb_number": raw[cfg.COL_ASSET].astype(int).values,
        "timestamp": raw[cfg.COL_TIME].values if cfg.COL_TIME in raw else pd.NaT,
        "power_w": raw[cfg.COL_POWER].values,
        "voltage_v": raw[cfg.COL_VOLTAGE].values,
        "current_a": raw[cfg.COL_CURRENT].values,
        "temperature_c": raw[cfg.COL_TEMP].values,
        "env_current_a": raw[cfg.COL_ENV_CURRENT].values,
        "weather": raw[cfg.COL_WEATHER].astype(str).values,
        "label": np.asarray(labels).astype(int),
        "prediction": [cfg.FAULT_NAMES[int(l)] for l in labels],
        "confidence": conf,
    })
    truth = cfg.COL_TARGET if cfg.COL_TARGET in raw else ("true_fault" if "true_fault" in raw else None)
    if truth:
        out["true_label"] = [cfg.FAULT_NAMES[int(v)] for v in raw[truth]]
    return out
