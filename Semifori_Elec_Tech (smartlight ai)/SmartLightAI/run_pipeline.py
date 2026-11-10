#!/usr/bin/env python3
"""
SmartLight AI - run the whole machine-learning pipeline.

    python run_pipeline.py                 # full run (a few minutes)
    python run_pipeline.py --fast          # quick run: fewer trees, no tuning
    python run_pipeline.py --data other.csv

Steps: clean data -> explore -> time-series analysis -> compare models ->
cross-validate -> tune -> pick best -> bulb threshold -> test once ->
temporal-feature experiment -> anomaly detectors -> explain -> save model,
metrics, charts, sample tickets and outputs/RESULTS.md.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
import sklearn

import config as cfg
import plots
from anomaly import add_anomaly_columns, evaluate_detectors, train_detectors
from chatbot import IntentClassifier
from data_prep import (add_temporal_features, chronological_indices, load_raw,
                       split_indices, to_feature_frame, validate_and_clean)
from models import (build_candidates, compare_models, cross_validate_candidates,
                    evaluate_model, feature_importance, fit_candidate, save_bundle,
                    score_readings, select_threshold, sweep_threshold, tune_random_forest)
from ticketing import add_priority, build_tickets, tickets_to_dataframe, verify_ticket
from time_series import run_time_series_analysis


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(cfg.DATA_FILE), help="path to the CSV dataset")
    p.add_argument("--model-out", default=str(cfg.MODEL_FILE), help="where to save the model")
    p.add_argument("--fast", action="store_true", help="fewer trees, no tuning, 3-fold CV")
    p.add_argument("--skip-tune", action="store_true", help="skip the random-forest search")
    p.add_argument("--skip-cv", action="store_true", help="skip cross-validation")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--tune-iter", type=int, default=10)
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    return o


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        cells = [f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    args = parse_args()
    warnings.filterwarnings("ignore")
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    def safe_plot(fn, *a, **k):
        if args.no_plots:
            return None
        try:
            return fn(*a, **k)
        except Exception as exc:               # a chart must never stop training
            log(f"   (chart skipped: {exc})")

    metrics = {"sklearn_version": sklearn.__version__}

    # ---------------------------------------------------------- 1. data
    log("1/11 Loading and cleaning data")
    df, quality = validate_and_clean(load_raw(args.data))
    X, y = to_feature_frame(df), df[cfg.COL_TARGET]
    quality["majority_class_share"] = float(y.value_counts(normalize=True).max())
    metrics["data_quality"] = quality
    log(f"   {quality['rows_in']:,} rows in, {quality['rows_out']:,} usable "
        f"({quality['rows_removed']} removed); majority class = {quality['majority_class_share']:.1%}")

    stats = df.groupby(df[cfg.COL_TARGET].map(cfg.FAULT_NAMES))[cfg.SENSOR_COLUMNS].agg(["mean", "std", "min", "max"])
    stats.to_csv(cfg.OUTPUT_DIR / "eda_sensor_stats_by_fault.csv")
    safe_plot(plots.class_distribution, df, cfg.PLOTS_DIR)
    safe_plot(plots.sensor_boxplots, df, cfg.PLOTS_DIR)
    safe_plot(plots.correlation_heatmap, df, cfg.PLOTS_DIR)

    # ------------------------------------------------ 2. time series
    log("2/11 Time-series analysis")
    ts = run_time_series_analysis(df, cfg.OUTPUT_DIR, plots=not args.no_plots)
    metrics["time_series"] = ts
    log(f"   status: {ts['status']}")

    # ----------------------------------------------------- 3. split
    idx_tr, idx_va, idx_te = split_indices(y)
    Xtr, ytr, Xva, yva, Xte, yte = (X.iloc[idx_tr], y.iloc[idx_tr], X.iloc[idx_va],
                                    y.iloc[idx_va], X.iloc[idx_te], y.iloc[idx_te])
    log(f"3/11 Split: train {len(Xtr):,} / validation {len(Xva):,} / test {len(Xte):,}")

    # ------------------------------------------------ 4. compare models
    log("4/11 Comparing models on the validation set")
    candidates = build_candidates(fast=args.fast)
    fitted, comp = compare_models(candidates, Xtr, ytr, Xva, yva, log=log)

    # ----------------------------------------------------- 5. CV
    cv_table = None
    if not args.skip_cv:
        folds = 3 if args.fast else args.cv_folds
        log(f"5/11 {folds}-fold cross-validation on train+validation")
        cv_names = ["Logistic Regression", "Random Forest", "Gradient Boosting", "Neural Network (MLP)"]
        X_tv, y_tv = pd.concat([Xtr, Xva]), pd.concat([ytr, yva])
        cv_table = cross_validate_candidates(candidates, cv_names, X_tv, y_tv, n_splits=folds, log=log)
        cv_table.to_csv(cfg.OUTPUT_DIR / "cross_validation.csv", index=False)
        metrics["cross_validation"] = cv_table
    else:
        log("5/11 Cross-validation skipped")

    # ---------------------------------------------------- 6. tuning
    specs = dict(candidates)
    if not (args.fast or args.skip_tune):
        log(f"6/11 Tuning the random forest ({args.tune_iter} random settings, 3-fold CV)")
        best_rf, best_params, tune_table = tune_random_forest(Xtr, ytr, n_iter=args.tune_iter)
        tune_table.to_csv(cfg.OUTPUT_DIR / "rf_tuning_results.csv", index=False)
        spec = dict(model=best_rf, augment=False, deployable=True)
        f2, c2 = compare_models({"Random Forest (tuned)": spec}, Xtr, ytr, Xva, yva, log=log)
        fitted.update(f2)
        comp = pd.concat([comp, c2], ignore_index=True)
        specs["Random Forest (tuned)"] = spec
        metrics["rf_best_params"] = {k: str(v) for k, v in best_params.items()}
    else:
        log("6/11 Tuning skipped")
    comp.to_csv(cfg.OUTPUT_DIR / "model_comparison.csv", index=False)
    metrics["model_comparison"] = comp
    safe_plot(plots.model_comparison, comp, cfg.PLOTS_DIR)

        # --------------------------------------------- 7. select + test
    
    deployable = comp[comp["deployable"]].copy()

    if deployable.empty:
        raise RuntimeError("No deployable models are available.")

    top_macro_f1 = deployable["val_macro_f1"].max()

    near_ties = deployable[
        deployable["val_macro_f1"] >= top_macro_f1 - 0.005
    ]

    best_row = near_ties.sort_values(
        ["val_mcc", "val_macro_f1"],
        ascending=False
    ).iloc[0]

    best_name = str(best_row["model"])
    final = fitted[best_name]

    log(
        f"7/11 Selected '{best_name}' "
        "(highest MCC among models within 0.005 of best validation macro-F1)"
    )

    sweep = sweep_threshold(final, Xva, yva)
    threshold = select_threshold(sweep)
    sweep.to_csv(cfg.OUTPUT_DIR / "bulb_threshold_sweep.csv", index=False)
    safe_plot(plots.threshold_sweep_plot, sweep, threshold, cfg.PLOTS_DIR)
    log("   bulb threshold: " + ("none (plain argmax is as good)" if threshold is None else f"{threshold:.2f}"))

    test_baseline = evaluate_model(fitted["Majority baseline"], Xte, yte)
    test_argmax = evaluate_model(final, Xte, yte)
    test_final = evaluate_model(final, Xte, yte, threshold=threshold)
    metrics.update(selected_model=best_name, bulb_threshold=threshold, test_baseline=test_baseline,
                   test_argmax=test_argmax, test_final=test_final)
    log(f"   TEST accuracy {test_final['accuracy']:.4f} | macro-F1 {test_final['macro_f1']:.4f} "
        f"(majority-class baseline accuracy {test_baseline['accuracy']:.4f})")

    safe_plot(plots.confusion_matrices, test_final["confusion_matrix"], cfg.PLOTS_DIR)
    safe_plot(plots.per_class_f1, test_final, cfg.PLOTS_DIR)
    safe_plot(plots.roc_curves, final, Xte, yte, cfg.PLOTS_DIR)

    # ------------------------------------ 8. temporal-feature experiment
    if ts["status"] == "ok":
        log("8/11 Experiment: do time-series features help the classifier?")
        d_t, tnames = add_temporal_features(df)
        Xt = pd.concat([X, d_t[tnames]], axis=1)
        rf_spec = candidates["Random Forest"]
        m_t = evaluate_model(fit_candidate(rf_spec, Xt.iloc[idx_tr], ytr), Xt.iloc[idx_va], yva)
        m_b = evaluate_model(fitted["Random Forest"], Xva, yva)
        delta = m_t["macro_f1"] - m_b["macro_f1"]
        metrics["temporal_experiment"] = {
            "n_temporal_features": len(tnames),
            "val_macro_f1_base": m_b["macro_f1"], "val_macro_f1_with_temporal": m_t["macro_f1"],
            "val_accuracy_base": m_b["accuracy"], "val_accuracy_with_temporal": m_t["accuracy"],
            "delta_macro_f1": delta,
            "conclusion": ("temporal context helped" if delta > 0.005 else
                           "temporal context did not help meaningfully; the single-reading model is kept"),
        }
        log(f"   macro-F1 {m_b['macro_f1']:.4f} -> {m_t['macro_f1']:.4f} ({delta:+.4f}): "
            f"{metrics['temporal_experiment']['conclusion']}")
    else:
        log("8/11 Temporal experiment skipped (no usable timestamps)")

    # ------------------------------------- 9. chronological check
    chron = chronological_indices(df)
    if chron is not None:
        tr_c, _, te_c = chron
        spec = specs[best_name]
        m_c = evaluate_model(fit_candidate(spec, X.iloc[tr_c], y.iloc[tr_c]),
                             X.iloc[te_c], y.iloc[te_c], threshold=threshold)
        metrics["chronological_check"] = {"accuracy": m_c["accuracy"], "macro_f1": m_c["macro_f1"],
                                          "note": "trained on the earliest 70% of time, tested on the latest 15%"}
        log(f"9/11 Chronological hold-out: accuracy {m_c['accuracy']:.4f}, macro-F1 {m_c['macro_f1']:.4f}")
    else:
        log("9/11 Chronological check skipped (no usable timestamps)")

    # ------------------------------------------ 10. anomaly detection
    log("10/11 Anomaly detection and deep learning")
    detectors = train_detectors(Xtr, ytr, Xva, yva)
    anom = evaluate_detectors(detectors, Xte, yte)
    anom.to_csv(cfg.OUTPUT_DIR / "anomaly_detection_results.csv", index=False)
    metrics["anomaly_detection"] = anom
    ae = detectors["autoencoder"]
    safe_plot(plots.loss_curve, ae.loss_curve_, cfg.PLOTS_DIR, "Autoencoder training loss", "dl_autoencoder_loss.png")
    if "Neural Network (MLP)" in fitted:
        mlp = fitted["Neural Network (MLP)"].named_steps["mlp"]
        safe_plot(plots.loss_curve, mlp.loss_curve_, cfg.PLOTS_DIR, "MLP classifier training loss", "dl_mlp_loss.png")
        metrics["mlp_epochs_run"] = int(mlp.n_iter_)
    from anomaly import anomaly_matrix
    A_te = anomaly_matrix(Xte)
    safe_plot(plots.anomaly_score_hist, ae.score(A_te), yte, ae.threshold, cfg.PLOTS_DIR,
              "Autoencoder reconstruction error (test set)", "dl_autoencoder_scores.png")
    for _, r in anom.iterrows():
        log(f"   {r['detector']:<24} AUC={r['roc_auc']:.3f}  recall={r['recall']:.3f}  "
            f"FPR={r['false_positive_rate']:.3f}")

    # ------------------------------------------- 11. explain + save
    log("11/11 Feature importance, chatbot check, saving")
    importance = feature_importance(final, Xva, yva)
    importance.to_csv(cfg.OUTPUT_DIR / "feature_importance.csv", index=False)
    metrics["feature_importance"] = importance
    safe_plot(plots.feature_importance_plot, importance, cfg.PLOTS_DIR)

    metrics["chatbot_intents"] = IntentClassifier().evaluate()

    bundle = {
        "model": final, "features": list(X.columns), "weather_mapping": cfg.WEATHER_MAPPING,
        "model_name": best_name, "bulb_threshold": threshold,
        "classes": [int(c) for c in final.classes_], "metrics": test_final,
        "test_accuracy": test_final["accuracy"], "test_macro_f1": test_final["macro_f1"],
        "anomaly": detectors, "confidence_threshold": cfg.CONFIDENCE_THRESHOLD,
    }
    size_mb = save_bundle(bundle, args.model_out)
    metrics["model_file_mb"] = round(size_mb, 2)
    log(f"   saved {args.model_out} ({size_mb:.1f} MB)")
    if size_mb > 90:
        log("   WARNING: file is close to GitHub's 100 MB limit; re-run with --fast or add min_samples_leaf.")

    raw_te = df.iloc[idx_te].reset_index(drop=True)
    results = add_priority(add_anomaly_columns(score_readings(raw_te, bundle), raw_te, detectors))
    tickets = build_tickets(results)
    tdf = tickets_to_dataframe(tickets)
    tdf.to_csv(cfg.OUTPUT_DIR / "sample_tickets.csv", index=False)
    (cfg.OUTPUT_DIR / "sample_tickets.json").write_text(json.dumps(tickets[:20], indent=2))
    metrics["sample_tickets"] = {
        "tickets_from_test_set": len(tickets),
        "high_priority_dispatch_now": int((tdf["dispatch_decision"] == "Dispatch crew now").sum()) if len(tdf) else 0,
        "needing_verification": int(tdf["needs_verification"].sum()) if len(tdf) else 0,
        "signatures_valid": all(verify_ticket(t) for t in tickets),
    }

    metrics["runtime_seconds"] = round(time.time() - t0)
    (cfg.OUTPUT_DIR / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2))
    write_results_md(metrics, comp, cv_table, anom, importance, sweep, threshold)
    log(f"Done in {time.time() - t0:.0f}s. Results: {cfg.OUTPUT_DIR / 'RESULTS.md'}")
    log("Next: streamlit run app.py   |   python live_monitor.py")


def write_results_md(m, comp, cv, anom, imp, sweep, threshold):
    q, tf, tb, ta = m["data_quality"], m["test_final"], m["test_baseline"], m["test_argmax"]
    L = ["# SmartLight AI - results summary", "",
         "Generated automatically by `run_pipeline.py`. All test-set numbers come from data the",
         "model never saw during training, tuning or model selection.", "",
         "## Data", f"- Rows in: {q['rows_in']:,}; usable: {q['rows_out']:,}; assets: {q['n_assets']}",
         f"- Removed: missing {q['dropped_missing_values']}, unknown weather {q['dropped_unknown_weather']}, "
         f"out of range {q['dropped_out_of_range']}, duplicates {q['dropped_duplicates']}",
         f"- Class counts: {q['class_counts']}",
         f"- Majority-class share: {q['majority_class_share']:.1%} (accuracy of always guessing 'No Fault')", "",
         "## Model comparison (validation set)", md_table(comp.drop(columns=["deployable"])), ""]
    if cv is not None:
        L += ["## Cross-validation (train + validation)", md_table(cv), ""]
    L += ["## Selected model", f"- **{m['selected_model']}** (highest MCC among models within 0.005 of best validation macro-F1)",
          f"- Bulb-failure threshold: {'none (argmax)' if threshold is None else f'{threshold:.2f}'}", ""]
    if "rf_best_params" in m:
        L += [f"- Tuned random-forest settings: {m['rf_best_params']}", ""]
    L += ["## Test-set performance (final model)",
          f"- Accuracy **{tf['accuracy']:.4f}**, balanced accuracy {tf['balanced_accuracy']:.4f}, "
          f"macro-F1 **{tf['macro_f1']:.4f}**, MCC {tf['mcc']:.4f}",
          f"- Majority-class baseline: accuracy {tb['accuracy']:.4f}, macro-F1 {tb['macro_f1']:.4f}",
          f"- Same model with plain argmax (no bulb threshold): accuracy {ta['accuracy']:.4f}, macro-F1 {ta['macro_f1']:.4f}", "",
          "| class | precision | recall | F1 | support |", "|---|---|---|---|---|"]
    for k, v in tf["per_class"].items():
        L.append(f"| {k} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | {v['support']} |")
    L += ["", "## Bulb-failure threshold sweep (validation)", md_table(sweep), ""]
    if "temporal_experiment" in m:
        t = m["temporal_experiment"]
        L += ["## Time-series features experiment (random forest, validation)",
              f"- {t['n_temporal_features']} lag / delta / rolling features added",
              f"- macro-F1 {t['val_macro_f1_base']:.4f} -> {t['val_macro_f1_with_temporal']:.4f} "
              f"({t['delta_macro_f1']:+.4f}); {t['conclusion']}", ""]
    if "chronological_check" in m:
        c = m["chronological_check"]
        L += ["## Chronological hold-out", f"- accuracy {c['accuracy']:.4f}, macro-F1 {c['macro_f1']:.4f} ({c['note']})", ""]
    L += ["## Anomaly detection (test set, 'is this reading faulty?')", md_table(anom), "",
          "## Feature importance (permutation, validation)", md_table(imp), ""]
    ts = m["time_series"]
    if ts.get("status") == "ok":
        L += ["## Time-series analysis",
              f"- Span {ts['span_start']} to {ts['span_end']}; {ts['n_assets']} assets; "
              f"median {ts['median_readings_per_asset']:.0f} readings per asset",
              f"- EWMA z-score anomaly flags vs labelled faults: {ts['ewma_any_sensor']}",
              f"- One-step forecast MAE ({ts['forecast']['column']}): EWMA {ts['forecast']['mae_ewma']}, "
              f"naive {ts['forecast']['mae_naive_persistence']}",
              f"- Fault onsets found for pre-failure analysis: {ts['n_fault_onsets']}",
              f"- Top risk assets: {ts['top_risk_assets']}", ""]
    ch = m["chatbot_intents"]
    L += ["## Chatbot", f"- Intent classifier: {ch['accuracy']:.1%} accuracy on {ch['n_test']} held-out phrasings "
          f"(trained on {ch['n_train']})", ""]
    st = m["sample_tickets"]
    L += ["## Tickets (generated from the test set)", f"- {st['tickets_from_test_set']} tickets; "
          f"{st['high_priority_dispatch_now']} dispatch-now; {st['needing_verification']} need verification; "
          f"signatures valid: {st['signatures_valid']}", "",
          f"Model file: {m['model_file_mb']} MB. Runtime: {m['runtime_seconds']} s. scikit-learn {m['sklearn_version']}."]
    (cfg.OUTPUT_DIR / "RESULTS.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
