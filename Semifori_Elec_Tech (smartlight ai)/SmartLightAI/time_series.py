"""
Time-series analysis of the telemetry.

Covers: daily/hourly patterns, trend + seasonality decomposition, per-asset
EWMA anomaly detection, one-step forecasting, pre-failure signatures and an
asset risk ranking (predictive maintenance).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no timestamp; sort by asset then time; add is_fault."""
    d = df.dropna(subset=[cfg.COL_TIME]).sort_values(
        [cfg.COL_ASSET, cfg.COL_TIME], kind="stable").reset_index(drop=True)
    d["is_fault"] = (d[cfg.COL_TARGET] != cfg.NO_FAULT).astype(int)
    return d


def usable(d: pd.DataFrame) -> bool:
    return len(d) >= 50 and d[cfg.COL_TIME].nunique() >= 10


def choose_rule(d: pd.DataFrame) -> str:
    """Choose daily, weekly, or hourly intervals based on the time span."""
    span = (d[cfg.COL_TIME].max() - d[cfg.COL_TIME].min()).days

    if span >= 10:
        return "D"
    return "h"

def resample_fleet(d: pd.DataFrame, rule: str | None = None) -> pd.DataFrame:
    rule = rule or choose_rule(d)
    g = d.set_index(cfg.COL_TIME).resample(rule)
    out = g[cfg.SENSOR_COLUMNS].mean()
    out["fault_rate"] = g["is_fault"].mean()
    out["readings"] = g["is_fault"].size()
    return out[out["readings"] > 0]


def hourly_profile(d: pd.DataFrame) -> pd.DataFrame:
    cols = cfg.SENSOR_COLUMNS + ["is_fault"]
    return d.groupby(d[cfg.COL_TIME].dt.hour)[cols].mean().rename_axis("hour")


def weekday_profile(d: pd.DataFrame) -> pd.DataFrame:
    cols = cfg.SENSOR_COLUMNS + ["is_fault"]
    return d.groupby(d[cfg.COL_TIME].dt.day_name())[cols].mean()


def decompose(series: pd.Series, period: int):
    """Classical additive decomposition: observed = trend + seasonal + residual."""
    s = series.astype(float)
    if len(s) < 2 * period:
        return None
    trend = s.rolling(period, center=True, min_periods=max(2, period // 2)).mean()
    seasonal = (s - trend).groupby(np.arange(len(s)) % period).transform("mean")
    return pd.DataFrame({"observed": s, "trend": trend, "seasonal": seasonal,
                         "residual": s - trend - seasonal})


def ewma_zscores(d: pd.DataFrame, col: str, span: int = 12, min_history: int = 5) -> pd.Series:
    """
    How unusual is each reading versus the asset's own recent past?
    Uses an exponentially weighted mean/std of PREVIOUS readings only.
    """
    prev = d.groupby(cfg.COL_ASSET)[col].shift(1)
    grouped = prev.groupby(d[cfg.COL_ASSET])
    mean = grouped.ewm(span=span, adjust=False).mean().reset_index(level=0, drop=True)
    std = grouped.ewm(span=span, adjust=False).std().reset_index(level=0, drop=True)
    z = (d[col] - mean) / std.replace(0, np.nan)
    z[d.groupby(cfg.COL_ASSET).cumcount() < min_history] = np.nan
    return z


def ewma_anomaly_evaluation(d: pd.DataFrame, k: float = 3.0, span: int = 12) -> pd.DataFrame:
    """Do |z| > k flags line up with the labelled faults?"""
    rows, any_flag = [], pd.Series(False, index=d.index)
    valid_any = pd.Series(False, index=d.index)
    for col in cfg.SENSOR_COLUMNS:
        z = ewma_zscores(d, col, span)
        ok = z.notna()
        flag = (z.abs() > k) & ok
        any_flag |= flag
        valid_any |= ok
        rows.append(_flag_row(col, flag[ok], d.loc[ok, "is_fault"]))
    rows.append(_flag_row("ANY sensor", any_flag[valid_any], d.loc[valid_any, "is_fault"]))
    return pd.DataFrame(rows)


def _flag_row(name, flag, truth) -> dict:
    flag, truth = flag.astype(bool), truth.astype(bool)
    tp = int((flag & truth).sum())
    precision = tp / flag.sum() if flag.sum() else 0.0
    recall = tp / truth.sum() if truth.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"signal": name, "flagged": int(flag.sum()), "precision": precision,
            "recall": recall, "f1": f1}


def forecast_evaluation(d: pd.DataFrame, col: str = cfg.COL_VOLTAGE, alpha: float = 0.3) -> dict:
    """One-step-ahead forecast of a healthy reading: EWMA vs 'same as last time'."""
    grouped = d.groupby(cfg.COL_ASSET)[col]
    naive = grouped.shift(1)
    ewma = naive.groupby(d[cfg.COL_ASSET]).ewm(alpha=alpha, adjust=False).mean() \
        .reset_index(level=0, drop=True)
    ok = naive.notna() & ewma.notna() & (d[cfg.COL_TARGET] == cfg.NO_FAULT)
    return {
        "column": col, "alpha": alpha, "n": int(ok.sum()),
        "mae_naive_persistence": float((d.loc[ok, col] - naive[ok]).abs().mean()) if ok.any() else None,
        "mae_ewma": float((d.loc[ok, col] - ewma[ok]).abs().mean()) if ok.any() else None,
    }


def prefailure_signatures(d: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    """
    Do readings look different just BEFORE a fault starts? Compares the mean
    of the `lookback` healthy readings preceding each fault onset with the
    overall healthy mean (difference in healthy standard deviations).
    """
    grouped = d.groupby(cfg.COL_ASSET)
    healthy_before = pd.Series(True, index=d.index)
    for k in range(1, lookback + 1):
        shifted = grouped[cfg.COL_TARGET].shift(k)
        healthy_before &= shifted.eq(cfg.NO_FAULT)
    onset = (d[cfg.COL_TARGET] != cfg.NO_FAULT) & healthy_before
    healthy = d[d[cfg.COL_TARGET] == cfg.NO_FAULT]

    rows = []
    for col in cfg.SENSOR_COLUMNS:
        if onset.sum() == 0:
            break
        before = pd.concat([grouped[col].shift(k)[onset] for k in range(1, lookback + 1)], axis=1).mean(axis=1)
        sd = healthy[col].std()
        rows.append({
            "sensor": col, "n_onsets": int(onset.sum()),
            "healthy_mean": float(healthy[col].mean()),
            "pre_fault_mean": float(before.mean()),
            "effect_size_sd": float((before.mean() - healthy[col].mean()) / sd) if sd else 0.0,
        })
    return pd.DataFrame(rows, columns=["sensor", "n_onsets", "healthy_mean",
                                       "pre_fault_mean", "effect_size_sd"])


def asset_risk_ranking(d: pd.DataFrame, recent_fraction: float = 0.25) -> pd.DataFrame:
    """
    Predictive-maintenance view: assets whose recent fault rate is high or
    rising compared with their own history rank first.
    """
    a = d.copy()
    n = a.groupby(cfg.COL_ASSET)["is_fault"].transform("size")
    position_from_end = a.groupby(cfg.COL_ASSET).cumcount(ascending=False)
    a["recent"] = position_from_end < np.maximum(3, (n * recent_fraction).astype(int))

    out = pd.DataFrame({
        "readings": a.groupby(cfg.COL_ASSET).size(),
        "fault_rate_overall": a.groupby(cfg.COL_ASSET)["is_fault"].mean(),
        "fault_rate_recent": a[a["recent"]].groupby(cfg.COL_ASSET)["is_fault"].mean(),
        "fault_rate_earlier": a[~a["recent"]].groupby(cfg.COL_ASSET)["is_fault"].mean(),
    }).fillna(0.0)
    out["trend"] = out["fault_rate_recent"] - out["fault_rate_earlier"]
    out["risk_score"] = (0.7 * out["fault_rate_recent"] + 0.3 * out["trend"].clip(lower=0)).round(4)

    faults = pd.crosstab(a[cfg.COL_ASSET], a[cfg.COL_TARGET]).drop(columns=[cfg.NO_FAULT], errors="ignore")
    if faults.shape[1]:
        dominant = faults.idxmax(axis=1).map(cfg.FAULT_NAMES)
        dominant[faults.sum(axis=1) == 0] = "-"
        out["most_common_fault"] = dominant
    else:
        out["most_common_fault"] = "-"
    out["most_common_fault"] = out["most_common_fault"].fillna("-")

    out["risk_level"] = pd.cut(out["risk_score"], [-0.01, 0.15, 0.30, 10],
                               labels=["Low", "Medium", "High"]).astype(str)
    out = out.reset_index().rename(columns={cfg.COL_ASSET: "bulb_number"})
    out.insert(0, "asset_id", out["bulb_number"].map(cfg.asset_id))
    return out.sort_values("risk_score", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------- plots
def make_plots(d: pd.DataFrame, outdir: Path) -> list:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(fig, name):
        p = outdir / name
        fig.tight_layout()
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(str(p))

    rule = choose_rule(d)
    fleet = resample_fleet(d, rule)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(fleet.index, fleet["fault_rate"] * 100, color="#c0392b")
    ax.set(title=f"Share of readings with a fault over time (resampled '{rule}')",
           ylabel="% of readings", xlabel="")
    save(fig, "ts_fault_rate.png")

    hp = hourly_profile(d)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    axes[0].plot(hp.index, hp[cfg.COL_VOLTAGE], marker="o"); axes[0].set_title("Voltage by hour (V)")
    axes[1].plot(hp.index, hp[cfg.COL_CURRENT], marker="o"); axes[1].set_title("Current by hour (A)")
    axes[2].plot(hp.index, hp["is_fault"] * 100, marker="o", color="#c0392b"); axes[2].set_title("Fault rate by hour (%)")
    for ax in axes:
        ax.set_xlabel("hour of day")
    save(fig, "ts_hourly_profile.png")

    period = {"h": 24, "D": 7, "W": 4}[rule]
    dec = decompose(fleet[cfg.COL_VOLTAGE], period)
    if dec is not None:
        fig, axes = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
        for ax, col in zip(axes, ["observed", "trend", "seasonal", "residual"]):
            ax.plot(dec.index, dec[col]); ax.set_ylabel(col)
        axes[0].set_title(f"Fleet average voltage: trend / seasonality / residual (period={period})")
        save(fig, "ts_decomposition.png")

    counts = d.groupby(cfg.COL_ASSET).size()
    sample_asset = counts.idxmax()
    a = d[d[cfg.COL_ASSET] == sample_asset].tail(200).copy()
    full = d.copy()
    zv = ewma_zscores(full, cfg.COL_VOLTAGE)
    zc = ewma_zscores(full, cfg.COL_CURRENT)
    a["zv"], a["zc"] = zv.loc[a.index], zc.loc[a.index]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    for ax, col, z in zip(axes, [cfg.COL_VOLTAGE, cfg.COL_CURRENT], ["zv", "zc"]):
        ax.plot(a[cfg.COL_TIME], a[col], lw=1)
        flagged = a[a[z].abs() > 3]
        ax.scatter(flagged[cfg.COL_TIME], flagged[col], color="red", s=22, label="EWMA anomaly (|z|>3)")
        faulty = a[a["is_fault"] == 1]
        ax.scatter(faulty[cfg.COL_TIME], faulty[col], facecolors="none", edgecolors="orange",
                   s=45, label="labelled fault")
        ax.set_ylabel(col.split(" (")[0])
    axes[0].set_title(f"{cfg.asset_id(sample_asset)}: readings with anomalies marked")
    axes[0].legend(loc="upper right", fontsize=8)
    save(fig, "ts_asset_anomalies.png")

    risk = asset_risk_ranking(d).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(risk["asset_id"], risk["risk_score"], color="#e67e22")
    ax.set(title="Top 10 assets by risk score (recent fault rate and trend)", xlabel="risk score")
    save(fig, "ts_asset_risk.png")
    return paths


def run_time_series_analysis(df: pd.DataFrame, outdir=None, plots: bool = True) -> dict:
    """Run everything, write CSV/JSON/PNG outputs, return a summary dict."""
    outdir = Path(outdir or cfg.OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    d = prepare(df)
    if not usable(d):
        return {"status": "skipped", "reason": "not enough valid timestamps"}

    fleet = resample_fleet(d)
    fleet.to_csv(outdir / "ts_fleet_series.csv")
    anomaly_eval = ewma_anomaly_evaluation(d)
    anomaly_eval.to_csv(outdir / "ts_anomaly_eval.csv", index=False)
    prefault = prefailure_signatures(d)
    prefault.to_csv(outdir / "ts_prefailure.csv", index=False)
    risk = asset_risk_ranking(d)
    risk.to_csv(outdir / "ts_asset_risk.csv", index=False)
    forecast = forecast_evaluation(d)
    hourly = hourly_profile(d)

    plot_files = []
    if plots:
        try:
            plot_files = make_plots(d, outdir / "plots")
        except Exception as exc:                      # plots must never stop the run
            plot_files = [f"plots failed: {exc}"]

    summary = {
        "status": "ok",
        "span_start": str(d[cfg.COL_TIME].min()), "span_end": str(d[cfg.COL_TIME].max()),
        "resample_rule": choose_rule(d), "n_assets": int(d[cfg.COL_ASSET].nunique()),
        "median_readings_per_asset": float(d.groupby(cfg.COL_ASSET).size().median()),
        "fault_rate_by_hour_max": float(hourly["is_fault"].max()),
        "fault_rate_by_hour_min": float(hourly["is_fault"].min()),
        "ewma_any_sensor": anomaly_eval.iloc[-1].to_dict(),
        "forecast": forecast,
        "n_fault_onsets": int(prefault["n_onsets"].iloc[0]) if len(prefault) else 0,
        "top_risk_assets": risk.head(5)[["asset_id", "risk_score", "risk_level"]].to_dict("records"),
        "plots": plot_files,
    }
    (outdir / "time_series_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
