"""Charts for the report: data exploration, model evaluation, anomaly detection."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import config as cfg

NAMES = [cfg.FAULT_NAMES[k] for k in cfg.FAULT_LABELS]


def _save(fig, outdir, name) -> str:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return str(path)


def class_distribution(df, outdir):
    counts = df[cfg.COL_TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar([cfg.FAULT_NAMES[k] for k in counts.index], counts.values, color="#2980b9")
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", fontsize=8)
    ax.set(title="Class distribution (imbalanced)", ylabel="records")
    ax.set_ylim(0, counts.max() * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=15)
    return _save(fig, outdir, "eda_class_distribution.png")


def sensor_boxplots(df, outdir):
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    labels = df[cfg.COL_TARGET].map(cfg.FAULT_NAMES)
    for ax, col in zip(axes, cfg.SENSOR_COLUMNS):
        sns.boxplot(x=labels, y=df[col], ax=ax, order=NAMES, color="#aed6f1", fliersize=1)
        ax.set_title(col.split(" (")[0], fontsize=9)
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=7)
    return _save(fig, outdir, "eda_sensor_boxplots.png")


def correlation_heatmap(df, outdir):
    d = df[cfg.SENSOR_COLUMNS + [cfg.COL_TARGET]].copy()
    d[cfg.COL_TARGET] = (d[cfg.COL_TARGET] != 0).astype(int)
    d = d.rename(columns={cfg.COL_TARGET: "is_fault", **{c: c.split(" (")[0] for c in cfg.SENSOR_COLUMNS}})
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(d.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation of sensor readings with 'is a fault'")
    return _save(fig, outdir, "eda_correlation.png")


def confusion_matrices(cm, outdir, title="Confusion matrix (test set)"):
    cm = np.asarray(cm)
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=NAMES, yticklabels=NAMES, ax=axes[0])
    axes[0].set(title=f"{title} - counts", xlabel="predicted", ylabel="actual")
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=NAMES, yticklabels=NAMES, ax=axes[1])
    axes[1].set(title="Row-normalised (recall per class)", xlabel="predicted", ylabel="actual")
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    return _save(fig, outdir, "model_confusion_matrix.png")


def model_comparison(comp: pd.DataFrame, outdir):
    d = comp.sort_values("val_macro_f1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(d))
    ax.barh(y - 0.2, d["val_accuracy"], height=0.4, label="accuracy", color="#5dade2")
    ax.barh(y + 0.2, d["val_macro_f1"], height=0.4, label="macro F1", color="#e67e22")
    ax.set_yticks(y, d["model"])
    ax.set(title="Model comparison on the validation set", xlim=(0, 1))
    ax.legend(loc="lower right")
    return _save(fig, outdir, "model_comparison.png")


def per_class_f1(metrics: dict, outdir):
    pc = metrics["per_class"]
    x = np.arange(len(pc))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.27, [v["precision"] for v in pc.values()], 0.27, label="precision")
    ax.bar(x, [v["recall"] for v in pc.values()], 0.27, label="recall")
    ax.bar(x + 0.27, [v["f1"] for v in pc.values()], 0.27, label="F1")
    ax.set_xticks(x, list(pc), rotation=15)
    ax.set(ylim=(0, 1.05), title="Per-class precision / recall / F1 (test set)")
    ax.legend(loc="lower right")
    return _save(fig, outdir, "model_per_class.png")


def feature_importance_plot(imp: pd.DataFrame, outdir):
    d = imp.sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(d["feature"].str.split(" \\(").str[0], d["importance_mean"],
            xerr=d["importance_std"], color="#27ae60")
    ax.set(title="Permutation importance (macro-F1 lost when shuffled)", xlabel="importance")
    return _save(fig, outdir, "model_feature_importance.png")


def roc_curves(model, X, y, outdir):
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    Y = label_binarize(np.asarray(y), classes=classes)
    fig, ax = plt.subplots(figsize=(6.2, 5))
    for i, c in enumerate(classes):
        fpr, tpr, _ = roc_curve(Y[:, i], proba[:, i])
        ax.plot(fpr, tpr, label=f"{cfg.FAULT_NAMES[c]} (AUC {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title="One-vs-rest ROC curves (test set)", xlabel="false positive rate", ylabel="true positive rate")
    ax.legend(fontsize=8)
    return _save(fig, outdir, "model_roc.png")


def threshold_sweep_plot(sweep: pd.DataFrame, selected, outdir):
    d = sweep[sweep["rule"] == "bulb threshold"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for col, lab in [("bulb_precision", "precision"), ("bulb_recall", "recall"),
                     ("bulb_f1", "F1"), ("macro_f1", "overall macro F1")]:
        ax.plot(d["threshold"], d[col], marker="o", label=f"bulb {lab}" if col.startswith("bulb") else lab)
    if selected is not None:
        ax.axvline(selected, color="k", ls="--", lw=0.8, label=f"selected = {selected:.2f}")
    ax.set(title="Bulb-failure probability threshold (validation set)", xlabel="threshold", ylim=(0, 1))
    ax.legend(fontsize=8)
    return _save(fig, outdir, "model_threshold_sweep.png")


def loss_curve(curve, outdir, title, name):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(curve)
    ax.set(title=title, xlabel="epoch", ylabel="training loss")
    return _save(fig, outdir, name)


def anomaly_score_hist(scores, y, threshold, outdir, title, name):
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    hi = np.quantile(scores, 0.995)
    ax.hist(scores[y == 0].clip(max=hi), bins=60, alpha=0.6, label="healthy", density=True)
    ax.hist(scores[y != 0].clip(max=hi), bins=60, alpha=0.6, label="faulty", density=True)
    ax.axvline(threshold, color="k", ls="--", label="alarm threshold")
    ax.set(title=title, xlabel="anomaly score")
    ax.legend()
    return _save(fig, outdir, name)
