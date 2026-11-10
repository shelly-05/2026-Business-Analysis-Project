"""
Deep learning components (scikit-learn neural networks, no GPU needed).

1. build_mlp           - multi-layer perceptron fault CLASSIFIER (supervised).
2. AutoencoderDetector - neural-network AUTOENCODER trained only on healthy
                         telemetry; a large reconstruction error means the
                         reading does not look like normal behaviour
                         (unsupervised anomaly detection, FR-03).
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as cfg


def build_mlp(seed: int = cfg.RANDOM_STATE, hidden=(128, 64, 32)) -> Pipeline:
    """Three hidden layers, ReLU, Adam, early stopping on a held-back slice."""
    return Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-4,                  # L2 regularisation
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )),
    ])


class AutoencoderDetector:
    """Compress then rebuild each reading; error = how abnormal it is."""

    name = "Autoencoder (neural network)"

    def __init__(self, hidden=(12, 4, 12), seed: int = cfg.RANDOM_STATE):
        self.hidden = hidden
        self.seed = seed
        self.scaler = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=1e-5,
            batch_size=256,
            max_iter=400,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=seed,
        )
        self.threshold = None

    def fit(self, X_normal):
        Z = self.scaler.fit_transform(np.asarray(X_normal, dtype=float))
        self.net.fit(Z, Z)               # target = the input itself
        return self

    def score(self, X) -> np.ndarray:
        """Mean squared reconstruction error per row (higher = more abnormal)."""
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        return ((Z - self.net.predict(Z)) ** 2).mean(axis=1)

    def set_threshold(self, X_normal_val, target_fpr: float = 0.02):
        """Flag at most `target_fpr` of healthy validation readings."""
        self.threshold = float(np.quantile(self.score(X_normal_val), 1.0 - target_fpr))
        return self

    def predict(self, X) -> np.ndarray:
        if self.threshold is None:
            raise RuntimeError("Call set_threshold() before predict().")
        return self.score(X) > self.threshold

    @property
    def loss_curve_(self):
        return self.net.loss_curve_
