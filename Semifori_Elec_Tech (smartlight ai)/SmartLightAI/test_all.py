#!/usr/bin/env python3
"""
SmartLight AI - one-command test suite.

    python test_all.py

Uses simulated data in a temporary folder, so it never touches your real
dataset, trained model or outputs. Exit code 0 = everything passed.
(The dashboard is exercised against a stand-in for Streamlit, so it checks the
app's own logic; open it with `streamlit run app.py` to see the real page.)
"""
import runpy
import sys
import tempfile
import time
import types
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import config as cfg

RESULTS = []


def check(name, fn):
    t = time.time()
    try:
        detail = fn()
        RESULTS.append((name, True, ""))
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else "") + f"  ({time.time() - t:.1f}s)")
    except Exception as exc:                                   # noqa: BLE001
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"  FAIL  {name}  ->  {type(exc).__name__}: {exc}")


TMP = Path(tempfile.mkdtemp(prefix="smartlight_test_"))
cfg.OUTPUT_DIR = TMP / "outputs"
cfg.PLOTS_DIR = cfg.OUTPUT_DIR / "plots"
cfg.PORTAL_DIR = cfg.OUTPUT_DIR / "portal_outbox"
cfg.MODEL_FILE = TMP / "model.joblib"
cfg.DATA_FILE = TMP / "data.csv"

from simulator import HeartbeatMonitor, SensorStream, generate_dataset          # noqa: E402
from data_prep import (add_temporal_features, augment_minority, clean_readings,   # noqa: E402
                       split_indices, to_feature_frame, validate_and_clean)

S = {}          # shared fixtures


def fixtures():
    raw = generate_dataset(n_rows=5000, n_assets=25, seed=7)
    raw.to_csv(cfg.DATA_FILE, index=False)
    S["raw"] = raw
    S["df"], S["quality"] = validate_and_clean(raw)
    S["X"], S["y"] = to_feature_frame(S["df"]), S["df"][cfg.COL_TARGET]
    return f"{len(raw)} simulated rows"


# ----------------------------------------------------------------- data prep
def t_cleaning():
    raw = S["raw"].copy()
    raw.loc[0, cfg.COL_VOLTAGE] = np.nan                       # missing
    raw.loc[1, cfg.COL_WEATHER] = "Snowy"                      # unknown weather
    raw.loc[2, cfg.COL_CURRENT] = 999.0                        # out of range
    raw.loc[3, cfg.COL_TARGET] = 9                             # bad label
    raw = pd.concat([raw, raw.iloc[[10]]], ignore_index=True)  # duplicate
    clean, rep = validate_and_clean(raw)
    assert rep["dropped_missing_values"] == 1, rep
    assert rep["dropped_unknown_weather"] == 1, rep
    assert rep["dropped_out_of_range"] == 1, rep
    assert rep["dropped_invalid_fault_label"] == 1, rep
    assert rep["dropped_duplicates"] == 1, rep
    assert len(clean) == len(S["raw"]) - 4
    return "5 kinds of bad rows removed and counted"


def t_missing_columns():
    try:
        validate_and_clean(S["raw"].drop(columns=[cfg.COL_VOLTAGE]))
    except ValueError as exc:
        assert "missing required columns" in str(exc)
        return "clear error for a missing column"
    raise AssertionError("no error raised")


def t_live_cleaning():
    batch = S["raw"].head(6).copy()
    batch.loc[0, cfg.COL_POWER] = np.nan
    batch.loc[1, cfg.COL_VOLTAGE] = 9999
    valid, rejected = clean_readings(batch)
    assert len(valid) == 4 and len(rejected) == 2
    assert set(rejected["reject_reason"]) == {"missing sensor value", "reading outside physical range"}


def t_no_lookahead():
    df = S["df"]
    full, names = add_temporal_features(df)
    cut = df[df[cfg.COL_TIME] <= df[cfg.COL_TIME].sort_values().iloc[2000]]
    part, _ = add_temporal_features(cut)
    a = full.loc[part.index, names].values
    assert np.allclose(a, part[names].values, equal_nan=True), "features changed when future rows were removed"
    return f"{len(names)} features use only past readings"


def t_split_and_augment():
    X, y = S["X"], S["y"]
    tr, va, te = split_indices(y)
    assert len(set(tr) & set(va)) == len(set(tr) & set(te)) == len(set(va) & set(te)) == 0
    assert abs(len(te) / len(y) - 0.15) < 0.01
    share = lambda idx: y.iloc[idx].value_counts(normalize=True).sort_index().values  # noqa: E731
    assert np.allclose(share(tr), share(te), atol=0.02), "split is not stratified"
    Xa, ya = augment_minority(X.iloc[tr], y.iloc[tr])
    counts = ya.value_counts()
    assert counts.min() >= int(y.iloc[tr].value_counts().median())
    assert len(Xa) == len(ya) > len(tr)
    assert Xa[cfg.COL_VOLTAGE].between(*cfg.VALID_RANGES[cfg.COL_VOLTAGE]).all()


# ----------------------------------------------------------------- simulator
def t_simulator():
    s = SensorStream(n_assets=10, seed=1, dropout_rate=0.3, corruption_rate=0.3)
    batches = [s.next_batch() for _ in range(8)]
    assert all(set(cfg.BASE_FEATURES) <= set(b.columns) for b in batches)
    assert min(len(b) for b in batches) < 10, "no node ever went offline"
    valid, rejected = clean_readings(pd.concat(batches, ignore_index=True))
    assert len(rejected) > 0 and len(valid) > 0
    return f"{len(rejected)} corrupted readings caught"


def t_heartbeat():
    hb = HeartbeatMonitor(timeout_minutes=100)
    t0 = pd.Timestamp("2026-01-01 00:00")
    hb.register([1, 2], t0)
    hb.update(pd.DataFrame({cfg.COL_ASSET: [1, 2], cfg.COL_TIME: [t0, t0]}))
    later = t0 + pd.Timedelta(minutes=180)
    hb.update(pd.DataFrame({cfg.COL_ASSET: [1], cfg.COL_TIME: [later]}))
    off = hb.offline_assets(later)
    assert list(off["bulb_number"]) == [2], off
    assert "not a hardware fault" in off.iloc[0]["status"]


# -------------------------------------------------------------------- models
def t_threshold_logic():
    from models import apply_bulb_threshold, select_threshold
    proba = np.array([[0.5, 0.0, 0.0, 0.3, 0.2], [0.2, 0.0, 0.0, 0.6, 0.2]])
    classes = np.array([0, 1, 2, 3, 4])
    assert list(apply_bulb_threshold(proba, classes, 0.25)) == [3, 3]
    assert list(apply_bulb_threshold(proba, classes, 0.50)) == [0, 3]
    sweep = pd.DataFrame({"rule": ["argmax", "bulb threshold"], "threshold": [np.nan, 0.3],
                          "macro_f1": [0.70, 0.705], "accuracy": [0.85, 0.80], "bulb_f1": [0.3, 0.3]})
    assert select_threshold(sweep) is None, "a tiny gain that costs accuracy must be rejected"
    sweep.loc[1, ["macro_f1", "accuracy"]] = [0.75, 0.84]
    assert select_threshold(sweep) == 0.3


def t_train_save_score():
    from anomaly import train_detectors
    from models import (build_candidates, evaluate_model, fit_candidate, load_bundle,
                        save_bundle, score_readings)
    X, y = S["X"], S["y"]
    tr, va, te = split_indices(y)
    spec = build_candidates(fast=True)["Random Forest"]
    spec["model"].set_params(n_estimators=40)
    model = fit_candidate(spec, X.iloc[tr], y.iloc[tr])
    m = evaluate_model(model, X.iloc[te], y.iloc[te])
    base = evaluate_model(fit_candidate(build_candidates()["Majority baseline"], X.iloc[tr], y.iloc[tr]),
                          X.iloc[te], y.iloc[te])
    assert m["accuracy"] > base["accuracy"] + 0.03, (m["accuracy"], base["accuracy"])
    assert 0 <= m["macro_f1"] <= 1 and len(m["confusion_matrix"]) == 5

    det = train_detectors(X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va])
    bundle = {"model": model, "features": list(X.columns), "weather_mapping": cfg.WEATHER_MAPPING,
              "model_name": "Test RF", "bulb_threshold": None, "classes": [0, 1, 2, 3, 4],
              "metrics": m, "test_accuracy": m["accuracy"], "test_macro_f1": m["macro_f1"],
              "anomaly": det, "confidence_threshold": cfg.CONFIDENCE_THRESHOLD}
    size = save_bundle(bundle)
    loaded = load_bundle()
    raw_te = S["df"].iloc[te].reset_index(drop=True)
    res = score_readings(raw_te, loaded)
    assert len(res) == len(raw_te) and res["confidence"].between(0, 1).all()
    assert set(res["prediction"]) <= set(cfg.FAULT_NAMES.values())
    assert (res["prediction"] == res["true_label"]).mean() > 0.7
    S["bundle"], S["raw_te"] = loaded, raw_te
    return f"accuracy {m['accuracy']:.3f} vs baseline {base['accuracy']:.3f}; file {size:.1f} MB"


def t_wrong_model_file():
    import joblib
    from models import load_bundle
    bad = TMP / "bad.joblib"
    joblib.dump({"just": "a dict"}, bad)
    try:
        load_bundle(bad)
    except ValueError as exc:
        assert "missing" in str(exc)
        return
    raise AssertionError("bad model file was accepted")


# ----------------------------------------------------------- deep learning
def t_deep_learning():
    from deep_learning import AutoencoderDetector, build_mlp
    from anomaly import anomaly_matrix
    X, y = S["X"], S["y"]
    tr, va, te = split_indices(y)
    mlp = build_mlp().fit(X.iloc[tr], y.iloc[tr])
    assert mlp.predict(X.iloc[te]).shape == (len(te),)
    A = anomaly_matrix(X)
    ae = AutoencoderDetector().fit(A.iloc[tr][y.iloc[tr] == 0]).set_threshold(A.iloc[va][y.iloc[va] == 0])
    assert len(ae.loss_curve_) > 3
    s = ae.score(A.iloc[te])
    assert s.shape == (len(te),) and (s >= 0).all()


# ------------------------------------------------------------------- anomaly
def t_envelope():
    from anomaly import EnvelopeDetector, anomaly_matrix
    X, y = S["X"], S["y"]
    A = anomaly_matrix(X)
    det = EnvelopeDetector().fit(A[y == 0])
    inside, outside = A[y == 0].iloc[[0]].copy(), A[y == 0].iloc[[0]].copy()
    outside[cfg.COL_CURRENT] = 9.0
    assert not det.set_threshold(A[y == 0]).predict(inside)[0]
    assert det.predict(outside)[0]


def t_anomaly_eval():
    from anomaly import evaluate_detectors
    res = evaluate_detectors(S["bundle"]["anomaly"], S["X"], S["y"])
    dep = res[res["detector"].str.startswith("Deployed")].iloc[0]
    assert dep["false_positive_rate"] < 0.05 and dep["recall"] > 0.2, dep.to_dict()
    return f"deployed detector: recall {dep['recall']:.2f}, false-alarm rate {dep['false_positive_rate']:.3f}"


# ------------------------------------------------------------------- tickets
def t_priority_and_tickets():
    from ticketing import add_priority, build_tickets, dispatch_to_portal, tickets_to_dataframe, verify_ticket
    from anomaly import add_anomaly_columns
    res = add_anomaly_columns(__import__("models").score_readings(S["raw_te"], S["bundle"]),
                              S["raw_te"], S["bundle"]["anomaly"])
    res = add_priority(res)
    toy = res.iloc[:3].copy()
    toy["label"], toy["confidence"] = [1, 1, 3], [0.95, 0.40, 0.90]
    toy["prediction"] = ["Short Circuit", "Short Circuit", "Bulb Failure"]
    toy["anomaly_flag"] = False
    out = add_priority(toy)
    assert list(out["dispatch_decision"]) == ["Dispatch crew now", "Verify remotely before dispatch", "Schedule repair"]
    assert out.loc[out.index[1], "priority"] == "Medium", "doubtful High alert must be downgraded"

    tickets = build_tickets(res)
    assert tickets and all(verify_ticket(t) for t in tickets)
    assert tickets[0]["priority"] in ("High", "Medium"), "highest priority first"
    tampered = dict(tickets[0], asset_id="ASSET-SL-2026-999")
    assert not verify_ticket(tampered), "tampering went undetected"
    paths = dispatch_to_portal(tickets[:3])
    assert len(paths) == 3 and all(Path(p).exists() for p in paths)
    df = tickets_to_dataframe(tickets)
    assert {"ticket_id", "asset_id", "voltage_v"} <= set(df.columns)
    S["results"] = res
    return f"{len(tickets)} signed tickets; tampering detected"


# ------------------------------------------------------------------- chatbot
def t_chatbot_nlp():
    from chatbot import IntentClassifier, extract_asset, extract_fault
    ev = IntentClassifier().evaluate()
    assert ev["accuracy"] >= 0.85, ev
    assert extract_asset("what is the status of bulb 12")[0] == 12
    assert extract_asset("check ASSET-SL-2026-045 please")[0] == 45
    assert extract_asset("is SL-007 ok")[0] == 7
    assert extract_asset("how many faults")[0] is None
    assert extract_fault("how many short circuits") == cfg.SHORT_CIRCUIT
    assert extract_fault("explain voltage surge") == cfg.VOLTAGE_SURGE
    assert extract_fault("why is it flickering") == cfg.LIGHT_FLICKERING
    return f"intent accuracy {ev['accuracy']:.0%} on {ev['n_test']} unseen phrasings"


def t_chatbot_answers():
    from chatbot import ChatState, SmartLightChatbot
    from time_series import asset_risk_ranking, prepare
    state = ChatState(results=S["results"], risk=asset_risk_ranking(prepare(S["df"])),
                      metrics=S["bundle"]["metrics"], model_name="Test RF")
    bot = SmartLightChatbot(state)
    res = S["results"]
    n_sc = int((res["label"] == cfg.SHORT_CIRCUIT).sum())
    assert str(n_sc) in bot.reply("how many short circuits are there?")
    some = int(res["bulb_number"].iloc[0])
    assert cfg.asset_id(some) in bot.reply(f"what is the status of bulb {some}")
    assert "voltage" in bot.reply("what is a voltage surge").lower()
    assert "risk" in bot.reply("which assets are at risk").lower()
    assert "accuracy" in bot.reply("how accurate is the model")
    assert "understand" in bot.reply("zxqv plorp").lower()
    empty = SmartLightChatbot(ChatState()).reply("how is the network doing")
    assert "no monitoring data" in empty.lower()


# --------------------------------------------------------------- time series
def t_time_series():
    from time_series import (asset_risk_ranking, decompose, ewma_zscores, prepare,
                             prefailure_signatures, run_time_series_analysis)
    d = prepare(S["df"])
    risk = asset_risk_ranking(d)
    assert risk["risk_score"].is_monotonic_decreasing and len(risk) == d[cfg.COL_ASSET].nunique()
    z = ewma_zscores(d, cfg.COL_VOLTAGE)
    assert z.notna().sum() > 0.5 * len(d)
    dec = decompose(pd.Series(np.sin(np.arange(96) * 2 * np.pi / 24) + np.arange(96) * 0.01), 24)
    assert dec is not None and abs(dec["seasonal"].iloc[:24].mean()) < 0.3
    assert isinstance(prefailure_signatures(d), pd.DataFrame)
    summary = run_time_series_analysis(S["df"], TMP / "ts", plots=True)
    assert summary["status"] == "ok"
    assert (TMP / "ts" / "plots" / "ts_fault_rate.png").exists()
    skipped = run_time_series_analysis(S["df"].assign(**{cfg.COL_TIME: pd.NaT}), TMP / "ts2", plots=False)
    assert skipped["status"] == "skipped"
    return "analysis, charts and graceful skip all work"


# ----------------------------------------------------------- live monitoring
def t_live_monitor():
    import live_monitor
    res, tickets, offline, rejected = live_monitor.run(assets=15, steps=6, seed=5, dispatch=True, echo=False)
    assert len(res) > 0 and "dispatch_decision" in res
    assert (res["prediction"] == res["true_label"]).mean() > 0.6
    assert len(list(cfg.PORTAL_DIR.glob("TKT-*.json"))) >= len(tickets) > 0
    return f"{len(res)} readings, {len(tickets)} tickets, {len(rejected)} rejected, {len(offline)} offline"


# ---------------------------------------------------------- dashboard (stub)
class _Ctx:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, name): return getattr(_STUB, name)


class _SS(dict):
    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v): self[k] = v


_STUB = types.ModuleType("streamlit")
_CALLS = {"error": [], "dataframes": 0, "html": 0, "press": set(), "type": None}
_STATE = _SS()


def _install_stub():
    st = _STUB
    st.session_state = _STATE
    noop = lambda *a, **k: None                                          # noqa: E731
    for n in ("set_page_config", "title", "caption", "subheader", "write", "info", "success",
              "warning", "divider", "bar_chart", "line_chart", "image", "metric", "markdown", "header"):
        setattr(st, n, noop)
    st.error = lambda msg, *a, **k: _CALLS["error"].append(msg)
    st.stop = lambda: (_ for _ in ()).throw(SystemExit("st.stop() called"))
    st.dataframe = lambda *a, **k: _CALLS.__setitem__("dataframes", _CALLS["dataframes"] + 1)
    st.columns = lambda spec, **k: [_Ctx() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [_Ctx() for _ in labels]
    st.expander = lambda *a, **k: _Ctx()
    st.chat_message = lambda *a, **k: _Ctx()
    st.sidebar = _Ctx()
    st.radio = lambda label, options, **k: options[0]
    st.slider = lambda label, lo, hi, value=None, step=None: value if value is not None else lo
    st.checkbox = lambda label, value=False, **k: value
    st.button = lambda label, **k: label in _CALLS["press"]
    st.download_button = lambda *a, **k: False
    st.cache_data = st.cache_resource = lambda *a, **k: (a[0] if a and callable(a[0]) else (lambda f: f))

    def text_input(label, key=None, on_change=None, **k):
        if _CALLS["type"] and key:
            _STATE[key] = _CALLS["type"]
            on_change and on_change()
        return _STATE.get(key, "")
    st.text_input = text_input

    comps = types.ModuleType("streamlit.components")
    v1 = types.ModuleType("streamlit.components.v1")
    v1.html = lambda *a, **k: _CALLS.__setitem__("html", _CALLS["html"] + 1)
    comps.v1 = v1
    st.components = comps
    sys.modules.update({"streamlit": st, "streamlit.components": comps, "streamlit.components.v1": v1})


def _run_app(press=(), type_text=None):
    _CALLS["press"], _CALLS["type"] = set(press), type_text
    runpy.run_path(str(Path(__file__).resolve().parent / "app.py"), run_name="__main__")


def t_dashboard():
    _install_stub()
    _run_app()                                                              # first visit
    assert not _CALLS["error"], _CALLS["error"]
    _run_app(press=["▶ Run AI monitoring"])                                 # run monitoring
    run = _STATE["run"]
    assert run and len(run["results"]) > 0 and not _CALLS["error"], _CALLS["error"]
    before = len(list(cfg.PORTAL_DIR.glob("TKT-*.json")))
    _run_app(press=["📨 Send to maintenance portal (simulated)"], type_text="how is the network doing")
    assert _STATE["run"]["dispatched"] > 0
    assert len(list(cfg.PORTAL_DIR.glob("TKT-*.json"))) > before
    assert _STATE["chat"][-2][0] == "user" and "readings" in _STATE["chat"][-1][1]
    _run_app(press=["What is most urgent?"])
    assert len(_STATE["chat"]) >= 4 and not _CALLS["error"]
    return f"{_CALLS['dataframes']} tables drawn, chat and portal dispatch work"


def t_dashboard_missing_model():
    keep = cfg.MODEL_FILE
    cfg.MODEL_FILE = TMP / "nope.joblib"
    _CALLS["error"].clear()
    try:
        _run_app()
    except SystemExit:
        pass
    finally:
        cfg.MODEL_FILE = keep
    assert _CALLS["error"] and "run_pipeline.py" in _CALLS["error"][0]
    _CALLS["error"].clear()


def main():
    print("=" * 66)
    print("  SMARTLIGHT AI - FULL TEST SUITE")
    print("=" * 66)
    check("build simulated test data", fixtures)
    print("\n[data preparation]")
    check("cleaning removes and counts bad rows", t_cleaning)
    check("missing column gives a clear error", t_missing_columns)
    check("live readings: corrupted ones are rejected", t_live_cleaning)
    check("temporal features never look ahead", t_no_lookahead)
    check("stratified split and training-only augmentation", t_split_and_augment)
    print("\n[IoT simulation]")
    check("sensor stream with outages and corruption", t_simulator)
    check("heartbeat separates silence from faults", t_heartbeat)
    print("\n[machine learning]")
    check("bulb threshold and selection rule", t_threshold_logic)
    check("train, beat baseline, save, reload, score", t_train_save_score)
    check("invalid model file is rejected", t_wrong_model_file)
    check("deep learning: MLP and autoencoder", t_deep_learning)
    print("\n[anomaly detection]")
    check("baseline envelope flags out-of-range readings", t_envelope)
    check("deployed detector: useful recall, few false alarms", t_anomaly_eval)
    print("\n[tickets, assistant, time series]")
    check("priority rules, signed tickets, portal dispatch", t_priority_and_tickets)
    check("chatbot NLP: intents and entities", t_chatbot_nlp)
    check("chatbot answers from live results", t_chatbot_answers)
    check("time-series analysis", t_time_series)
    print("\n[end to end]")
    check("live monitor: stream -> model -> tickets -> portal", t_live_monitor)
    check("dashboard runs through all tabs and actions", t_dashboard)
    check("dashboard explains a missing model", t_dashboard_missing_model)

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    print("\n" + "=" * 66)
    print(f"  SUMMARY: {passed}/{len(RESULTS)} passed")
    print("=" * 66)
    for name, ok, detail in RESULTS:
        if not ok:
            print(f"  FAILED: {name}: {detail}")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
