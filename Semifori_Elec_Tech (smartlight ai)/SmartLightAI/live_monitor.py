#!/usr/bin/env python3
"""
End-to-end prototype pipeline:

  IoT data stream -> Python preprocessing -> ML evaluation ->
  fault classification -> maintenance notification

    python live_monitor.py --assets 20 --steps 6 --seed 1
"""
from __future__ import annotations

import argparse
import warnings

import pandas as pd

import config as cfg
from anomaly import add_anomaly_columns
from data_prep import clean_readings
from models import load_bundle, score_readings
from simulator import HeartbeatMonitor, SensorStream
from ticketing import add_priority, build_tickets, dispatch_to_portal


def run(assets=20, steps=6, seed=None, model_file=None, dispatch=True, echo=True):
    """Run the simulated pipeline. Returns (results, tickets, offline, rejected)."""
    bundle = load_bundle(model_file or cfg.MODEL_FILE)
    stream = SensorStream(n_assets=assets, seed=seed)
    heartbeat = HeartbeatMonitor()
    heartbeat.register(stream.assets, stream.now)

    all_valid, all_rejected, offline = [], [], pd.DataFrame()
    say = print if echo else (lambda *a, **k: None)

    for step in range(1, steps + 1):
        batch = stream.next_batch()                              # 1. IoT stream
        heartbeat.update(batch)
        valid, rejected = clean_readings(batch)                  # 2. preprocessing
        all_valid.append(valid)
        all_rejected.append(rejected)
        offline = heartbeat.offline_assets(stream.now)
        say(f"step {step}: {len(batch)} readings, {len(rejected)} rejected, {len(offline)} node(s) silent")

    valid = pd.concat(all_valid, ignore_index=True)
    rejected = pd.concat(all_rejected, ignore_index=True)

    results = score_readings(valid, bundle)                      # 3-4. ML + classification
    if bundle.get("anomaly"):
        results = add_anomaly_columns(results, valid, bundle["anomaly"])
    results = add_priority(results, bundle.get("confidence_threshold", cfg.CONFIDENCE_THRESHOLD))
    tickets = build_tickets(results)                             # 5. report generation
    if dispatch and tickets:
        dispatch_to_portal(tickets)                              # 6. notification

    say("\n=== MAINTENANCE SUMMARY ===")
    say(f"Readings analysed: {len(results)} | faults predicted: {int((results['label'] != 0).sum())} "
        f"| tickets raised: {len(tickets)}")
    for t in tickets[:8]:
        say(f"[{t['priority'].upper():6}] {t['summary']}")
    if len(tickets) > 8:
        say(f"... and {len(tickets) - 8} more (see {cfg.PORTAL_DIR})")
    if len(rejected):
        say(f"\n{len(rejected)} reading(s) rejected as corrupted: " + ", ".join(sorted(set(rejected['reject_reason']))))
    if len(offline):
        say("\nCommunication loss (NOT hardware faults): " + ", ".join(offline["asset_id"]))
    return results, tickets, offline, rejected


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, default=20)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-dispatch", action="store_true", help="do not write tickets to the portal folder")
    a = ap.parse_args()
    run(a.assets, a.steps, a.seed, dispatch=not a.no_dispatch)
