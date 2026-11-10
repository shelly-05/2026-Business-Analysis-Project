"""
Data preprocessing pipeline (FR-02).

load -> validate/clean -> encode -> (optional temporal features) -> split.
Every cleaning step is counted so the data-quality report can show it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import config as cfg


# ------------------------------------------------------------------ loading
def load_raw(path=None) -> pd.DataFrame:
    path = Path(path or cfg.DATA_FILE)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place street_light_fault_prediction_dataset.csv in the project "
            "folder or pass --data <path>."
        )
    return pd.read_csv(path)


def _out_of_range(d: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=d.index)
    for col, (lo, hi) in cfg.VALID_RANGES.items():
        mask |= (d[col] < lo) | (d[col] > hi)
    return mask


def _parse_time(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().mean() > 0.05:                       # mixed formats? retry
        parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    return parsed


# ----------------------------------------------------------------- cleaning
def validate_and_clean(df: pd.DataFrame):
    """Return (clean_df, report). Rows are dropped, never silently altered."""
    missing = [c for c in cfg.REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    report = {"rows_in": int(len(df))}
    d = df[cfg.REQUIRED_COLUMNS].copy()

    numeric = cfg.SENSOR_COLUMNS + [cfg.COL_ASSET, cfg.COL_TARGET]
    for col in numeric:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d[cfg.COL_WEATHER] = d[cfg.COL_WEATHER].astype("string").str.strip().str.title()
    d[cfg.COL_TIME] = _parse_time(d[cfg.COL_TIME])
    report["unparseable_timestamps_kept"] = int(d[cfg.COL_TIME].isna().sum())

    keep = pd.Series(True, index=d.index)

    bad = keep & d[numeric].isna().any(axis=1)
    report["dropped_missing_values"] = int(bad.sum())
    keep &= ~bad

    bad = keep & ~d[cfg.COL_WEATHER].isin(list(cfg.WEATHER_MAPPING))
    report["dropped_unknown_weather"] = int(bad.sum())
    keep &= ~bad

    bad = keep & ~d[cfg.COL_TARGET].isin(cfg.FAULT_LABELS)
    report["dropped_invalid_fault_label"] = int(bad.sum())
    keep &= ~bad

    bad = keep & _out_of_range(d)
    report["dropped_out_of_range"] = int(bad.sum())
    keep &= ~bad

    d = d[keep]
    dup = d.duplicated(keep="first")
    report["dropped_duplicates"] = int(dup.sum())
    d = d[~dup].reset_index(drop=True)

    d[cfg.COL_ASSET] = d[cfg.COL_ASSET].astype(int)
    d[cfg.COL_TARGET] = d[cfg.COL_TARGET].astype(int)
    d[cfg.COL_WEATHER] = d[cfg.COL_WEATHER].astype(str)

    report["rows_out"] = int(len(d))
    report["rows_removed"] = report["rows_in"] - report["rows_out"]
    report["n_assets"] = int(d[cfg.COL_ASSET].nunique())
    report["class_counts"] = {
        cfg.FAULT_NAMES[k]: int(v)
        for k, v in d[cfg.COL_TARGET].value_counts().sort_index().items()
    }
    if report["rows_out"] == 0:
        raise ValueError("No usable rows remain after cleaning the dataset.")
    return d, report


def clean_readings(readings: pd.DataFrame):
    """
    Validate unlabeled live readings. Returns (valid, rejected); rejected
    rows carry a reject_reason and are never sent to the model.
    """
    d = readings.copy()
    for col in cfg.SENSOR_COLUMNS:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d[cfg.COL_WEATHER] = d[cfg.COL_WEATHER].astype("string").str.strip().str.title()

    missing = d[cfg.SENSOR_COLUMNS].isna().any(axis=1)
    weather_bad = ~d[cfg.COL_WEATHER].isin(list(cfg.WEATHER_MAPPING))
    range_bad = _out_of_range(d)
    reason = np.select(
        [missing, weather_bad, range_bad],
        ["missing sensor value", "unknown weather value", "reading outside physical range"],
        default="",
    )
    ok = reason == ""
    d[cfg.COL_WEATHER] = d[cfg.COL_WEATHER].astype(str)
    valid = d[ok].reset_index(drop=True)
    rejected = d[~ok].assign(reject_reason=reason[~ok]).reset_index(drop=True)
    return valid, rejected


# ----------------------------------------------------------------- features
def to_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """The six model inputs, in training order, with weather encoded."""
    X = df[cfg.BASE_FEATURES].copy()
    X[cfg.COL_WEATHER] = X[cfg.COL_WEATHER].map(cfg.WEATHER_MAPPING).astype(int)
    return X


def add_temporal_features(df: pd.DataFrame, windows=(3, 6)):
    """
    Time-series features per asset, using only the current and PAST readings
    (no look-ahead): previous value, change since previous value, and rolling
    mean / std over the last `windows` readings.
    Returns (df_with_features, feature_names).
    """
    d = df.sort_values([cfg.COL_ASSET, cfg.COL_TIME], kind="stable").copy()
    names = []
    grouped = d.groupby(cfg.COL_ASSET)
    for col in cfg.SENSOR_COLUMNS:
        lag = grouped[col].shift(1).fillna(d[col])
        d[f"{col}__lag1"] = lag
        d[f"{col}__delta1"] = d[col] - lag
        names += [f"{col}__lag1", f"{col}__delta1"]
        for w in windows:
            roll = grouped[col].rolling(w, min_periods=1)
            d[f"{col}__roll{w}_mean"] = roll.mean().reset_index(level=0, drop=True)
            d[f"{col}__roll{w}_std"] = (
                grouped[col].rolling(w, min_periods=2).std()
                .reset_index(level=0, drop=True).fillna(0.0)
            )
            names += [f"{col}__roll{w}_mean", f"{col}__roll{w}_std"]
    return d.sort_index(), names


# ------------------------------------------------------------------- splits
def split_indices(y, test_size=0.15, val_size=0.15, seed=cfg.RANDOM_STATE):
    """Stratified train / validation / test positions (70/15/15 by default)."""
    y = np.asarray(y)
    idx = np.arange(len(y))
    idx_rest, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val = train_test_split(
        idx_rest, test_size=rel_val, stratify=y[idx_rest], random_state=seed
    )
    return idx_train, idx_val, idx_test


def chronological_indices(df: pd.DataFrame, train=0.70, val=0.15):
    """Earliest rows train, next validate, latest test. None if no timestamps."""
    t = df[cfg.COL_TIME]
    if t.isna().mean() > 0.5 or t.nunique() < 10:
        return None
    order = np.argsort(t.fillna(t.max()).values, kind="stable")
    n = len(order)
    a, b = int(n * train), int(n * (train + val))
    return order[:a], order[a:b], order[b:]


def augment_minority(X: pd.DataFrame, y: pd.Series, seed=cfg.RANDOM_STATE,
                     noise_scale=0.05, target=None):
    """
    Raise rare classes up to `target` rows (default: the median class size)
    by resampling with small Gaussian jitter. Training data only; the
    validation and test sets are never augmented.
    """
    rng = np.random.default_rng(seed)
    y = pd.Series(np.asarray(y), index=X.index)
    counts = y.value_counts()
    target = int(counts.median()) if target is None else int(target)
    num_cols = [c for c in X.columns if c != cfg.COL_WEATHER]

    parts_x, parts_y = [X], [y]
    for label, count in counts.items():
        if count >= target:
            continue
        need = target - count
        source = X[y == label]
        extra = source.sample(n=need, replace=True,
                              random_state=int(rng.integers(1_000_000_000))).copy()
        spread = source[num_cols].std().fillna(0.0).values * noise_scale
        extra[num_cols] = extra[num_cols].values + rng.normal(0, 1, (need, len(num_cols))) * spread
        for col in num_cols:
            if col in cfg.VALID_RANGES:
                lo, hi = cfg.VALID_RANGES[col]
                extra[col] = extra[col].clip(lo, hi)
        parts_x.append(extra)
        parts_y.append(pd.Series(label, index=extra.index))

    X_aug = pd.concat(parts_x, ignore_index=True)
    y_aug = pd.concat(parts_y, ignore_index=True)
    return X_aug, y_aug
