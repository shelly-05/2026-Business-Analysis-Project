"""
Pattern anomaly identification (FR-03).

Detectors learn ONLY what healthy telemetry looks like, so they can flag
readings the classifier was never taught (unknown fault types). They are
used next to the supervised classifier as a safety net.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler

import config as cfg
from data_prep import to_feature_frame
from deep_learning import AutoencoderDetector


def anomaly_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """Sensor columns plus one-hot weather (weather is categorical)."""
    out = X[cfg.SENSOR_COLUMNS].copy()
    for name, code in cfg.WEATHER_MAPPING.items():
        out[f"weather_{name}"] = (X[cfg.COL_WEATHER] == code).astype(float)
    return out


class EnvelopeDetector:
    """
    Learns the normal operating envelope of every sensor (0.1th to 99.9th
    percentile of HEALTHY readings). The score is how far the worst sensor
    sits outside its envelope, in envelope-widths; 0 means inside.
    This is the 'deviation from baseline operating curve' check of FR-03.
    """
    name = "Baseline envelope"

    def __init__(self, lower_q: float = 0.001, upper_q: float = 0.999):
        self.lower_q, self.upper_q = lower_q, upper_q
        self.threshold = 0.0

    def fit(self, A_normal):
        A = pd.DataFrame(A_normal)[cfg.SENSOR_COLUMNS]
        self.lo = A.quantile(self.lower_q)
        self.hi = A.quantile(self.upper_q)
        self.width = (self.hi - self.lo).replace(0, 1.0)
        return self

    def score(self, A) -> np.ndarray:
        a = pd.DataFrame(A)[cfg.SENSOR_COLUMNS]
        above = (a - self.hi) / self.width
        below = (self.lo - a) / self.width
        return np.maximum(above, below).clip(lower=0).max(axis=1).values

    def set_threshold(self, A_normal_val, target_fpr: float = 0.02):
        self.threshold = float(np.quantile(self.score(A_normal_val), 1.0 - target_fpr))
        return self

    def predict(self, A) -> np.ndarray:
        return self.score(A) > self.threshold


class IsolationDetector:
    name = "Isolation Forest"

    def __init__(self, seed: int = cfg.RANDOM_STATE):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, random_state=seed, n_jobs=-1)
        self.threshold = None

    def fit(self, X_normal):
        self.model.fit(self.scaler.fit_transform(np.asarray(X_normal, dtype=float)))
        return self

    def score(self, X) -> np.ndarray:
        """Higher = more abnormal."""
        return -self.model.score_samples(self.scaler.transform(np.asarray(X, dtype=float)))

    def set_threshold(self, X_normal_val, target_fpr: float = 0.02):
        self.threshold = float(np.quantile(self.score(X_normal_val), 1.0 - target_fpr))
        return self

    def predict(self, X) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call set_threshold() before predict().")
        return self.score(X) > self.threshold


def train_detectors(X_train, y_train, X_val, y_val, seed=cfg.RANDOM_STATE, target_fpr=0.02):
    """Fit both detectors on healthy training rows; calibrate on healthy val rows."""
    A_train = anomaly_matrix(X_train)[np.asarray(y_train) == cfg.NO_FAULT]
    A_val = anomaly_matrix(X_val)[np.asarray(y_val) == cfg.NO_FAULT]
    detectors = {
        "envelope": EnvelopeDetector().fit(A_train).set_threshold(A_val, target_fpr),
        "iforest": IsolationDetector(seed).fit(A_train).set_threshold(A_val, target_fpr),
        "autoencoder": AutoencoderDetector(seed=seed).fit(A_train).set_threshold(A_val, target_fpr),
    }
    return detectors


def detect(detectors: dict, X: pd.DataFrame) -> pd.DataFrame:
    """Scores and flags from all three detectors plus the deployed combination."""
    A = anomaly_matrix(X)
    out = pd.DataFrame(index=X.index)
    out["envelope_score"] = detectors["envelope"].score(A)
    out["iforest_score"] = detectors["iforest"].score(A)
    out["autoencoder_score"] = detectors["autoencoder"].score(A)
    out["envelope_flag"] = out["envelope_score"] > detectors["envelope"].threshold
    out["iforest_flag"] = out["iforest_score"] > detectors["iforest"].threshold
    out["autoencoder_flag"] = out["autoencoder_score"] > detectors["autoencoder"].threshold
    # Deployed rule: outside the baseline envelope, OR both neural/forest models agree.
    out["anomaly_flag"] = out["envelope_flag"] | (out["iforest_flag"] & out["autoencoder_flag"])
    return out


def evaluate_detectors(detectors: dict, X_test, y_test) -> pd.DataFrame:
    """'Is this reading faulty?' scored for each detector and for their AND."""
    y_test = np.asarray(y_test)
    truth = (y_test != cfg.NO_FAULT).astype(int)
    scored = detect(detectors, X_test)

    variants = {
        "Baseline envelope": ("envelope_score", "envelope_flag"),
        "Isolation Forest": ("iforest_score", "iforest_flag"),
        "Autoencoder": ("autoencoder_score", "autoencoder_flag"),
        "Deployed (envelope OR both ML agree)": (None, "anomaly_flag"),
    }
    rows = []
    for name, (score_col, flag_col) in variants.items():
        flag = scored[flag_col].values.astype(int)
        p, r, f, _ = precision_recall_fscore_support(
            truth, flag, average="binary", zero_division=0
        )
        row = {
            "detector": name,
            "roc_auc": float(roc_auc_score(truth, scored[score_col])) if score_col else np.nan,
            "precision": float(p), "recall": float(r), "f1": float(f),
            "false_positive_rate": float(flag[truth == 0].mean()) if (truth == 0).any() else np.nan,
        }
        for label in cfg.FAULT_LABELS[1:]:
            sel = y_test == label
            row[f"detects_{cfg.FAULT_NAMES[label]}"] = float(flag[sel].mean()) if sel.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_anomaly_columns(results: pd.DataFrame, raw: pd.DataFrame, detectors: dict) -> pd.DataFrame:
    """Attach anomaly scores/flags to a scored-results table (same row order)."""
    scored = detect(detectors, to_feature_frame(raw.reset_index(drop=True)))
    out = results.reset_index(drop=True).copy()
    out["anomaly_score"] = scored["envelope_score"].values
    out["anomaly_flag"] = scored["anomaly_flag"].values
    return out
