"""
Automated report generation (FR-06) and municipal alert dispatch (FR-07).

* add_priority   - urgency + dispatch decision, with a confidence gate so a
                   doubtful prediction is verified before a crew is sent.
* build_tickets  - structured diagnostic record per confirmed fault.
* dispatch_to_portal - writes tickets to a simulated maintenance portal
                   (a live municipal database is outside the project scope).
Tickets carry an HMAC signature so tampering can be detected (security NFR).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg


def _secret() -> bytes:
    return os.environ.get("SMARTLIGHT_SECRET", "smartlight-dev-secret").encode()


def add_priority(results: pd.DataFrame, conf_threshold: float = cfg.CONFIDENCE_THRESHOLD) -> pd.DataFrame:
    """Add priority, verification flag, dispatch decision and recommended action."""
    r = results.copy()
    base = r["label"].map(lambda k: cfg.FAULT_INFO[int(k)]["priority"])
    faulty = r["label"] != cfg.NO_FAULT
    doubtful = faulty & (r["confidence"] < conf_threshold)

    priority = base.copy()
    priority[doubtful & (base == "High")] = "Medium"       # do not rush a crew on a doubtful call
    r["priority"] = priority
    r["needs_verification"] = doubtful

    r["dispatch_decision"] = np.select(
        [~faulty, doubtful, priority == "High"],
        ["No action", "Verify remotely before dispatch", "Dispatch crew now"],
        default="Schedule repair",
    )
    r["recommended_action"] = r["label"].map(lambda k: cfg.FAULT_INFO[int(k)]["action"])

    if "anomaly_flag" in r.columns:
        review = (~faulty) & r["anomaly_flag"].astype(bool)
        r["review_anomaly"] = review
        r.loc[review, "dispatch_decision"] = "Review remotely (unclassified anomaly)"
        r.loc[review, "priority"] = "Low"
    return r


def _sign(ticket: dict) -> str:
    body = {k: v for k, v in ticket.items() if k != "signature"}
    payload = json.dumps(body, sort_keys=True, default=str).encode()
    return hmac.new(_secret(), payload, hashlib.sha256).hexdigest()


def verify_ticket(ticket: dict) -> bool:
    """True if the ticket has not been altered since it was created."""
    return hmac.compare_digest(ticket.get("signature", ""), _sign(ticket))


def _summary(row) -> str:
    return (
        f"{row['asset_id']}: {row['prediction']} detected "
        f"(confidence {row['confidence']:.0%}). Voltage {row['voltage_v']:.1f} V, "
        f"current {row['current_a']:.2f} A, power {row['power_w']:.1f} W. "
        f"Decision: {row['dispatch_decision']}. {row['recommended_action']}"
    )


def build_tickets(results: pd.DataFrame, created_at: datetime | None = None) -> list:
    """One signed ticket per predicted fault (needs add_priority first)."""
    created_at = created_at or datetime.now()
    stamp = created_at.strftime("%Y%m%d-%H%M%S")
    faults = results[results["label"] != cfg.NO_FAULT]
    tickets = []
    for i, (_, row) in enumerate(faults.iterrows(), start=1):
        ticket = {
            "ticket_id": f"TKT-{stamp}-{i:04d}",
            "created_at": created_at.isoformat(timespec="seconds"),
            "asset_id": str(row["asset_id"]),
            "fault_type": str(row["prediction"]),
            "confidence": round(float(row["confidence"]), 3),
            "priority": str(row["priority"]),
            "dispatch_decision": str(row["dispatch_decision"]),
            "needs_verification": bool(row["needs_verification"]),
            "reading_timestamp": str(row["timestamp"]),
            "diagnostics": {
                "power_w": round(float(row["power_w"]), 2),
                "voltage_v": round(float(row["voltage_v"]), 2),
                "current_a": round(float(row["current_a"]), 2),
                "temperature_c": round(float(row["temperature_c"]), 2),
                "environmental_current_a": round(float(row["env_current_a"]), 2),
                "weather": str(row["weather"]),
            },
            "recommended_action": str(row["recommended_action"]),
            "summary": _summary(row),
            "status": "OPEN",
        }
        ticket["signature"] = _sign(ticket)
        tickets.append(ticket)
    order = {"High": 0, "Medium": 1, "Low": 2}
    return sorted(tickets, key=lambda t: (order.get(t["priority"], 3), -t["confidence"]))


def tickets_to_dataframe(tickets: list) -> pd.DataFrame:
    rows = []
    for t in tickets:
        flat = {k: v for k, v in t.items() if k not in ("diagnostics", "signature")}
        flat.update(t["diagnostics"])
        rows.append(flat)
    return pd.DataFrame(rows)


def dispatch_to_portal(tickets: list, outdir=None) -> list:
    """Push tickets to the (simulated) central maintenance portal."""
    outdir = Path(outdir or cfg.PORTAL_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for t in tickets:
        path = outdir / f"{t['ticket_id']}_{t['asset_id']}.json"
        path.write_text(json.dumps(t, indent=2))
        paths.append(str(path))
    return paths
