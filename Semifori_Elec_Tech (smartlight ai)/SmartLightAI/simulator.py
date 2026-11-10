"""
IoT telemetry simulator (FR-01) and heartbeat monitor.

The brief forbids installing hardware on real poles, so sensor feeds are
simulated. This module provides:

* generate_reading / generate_dataset - readings that follow the fault
  signatures found in street_light_fault_prediction_dataset.csv (used for
  testing and for demos when the real CSV is not to hand).
* SensorStream - a live feed for many assets, with random node outages and
  occasional corrupted readings.
* HeartbeatMonitor - tells a communication drop apart from a hardware fault.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

import config as cfg

# Class mix and weather mix observed in the real dataset's diagnostic report.
CLASS_COUNTS = {0: 24111, 1: 1706, 2: 1720, 3: 3359, 4: 3414}
WEATHER_PROBS = {
    0: {"Clear": 0.3255, "Cloudy": 0.3336, "Rainy": 0.3409},
    1: {"Clear": 0.3007, "Rainy": 0.6993},
    2: {"Clear": 0.3198, "Cloudy": 0.6802},
    3: {"Clear": 0.6943, "Cloudy": 0.3057},
    4: {"Clear": 0.3049, "Rainy": 0.6951},
}


def _weather(rng: np.random.Generator, label: int) -> str:
    names = list(WEATHER_PROBS[label])
    probs = np.array([WEATHER_PROBS[label][n] for n in names], dtype=float)
    return str(rng.choice(names, p=probs / probs.sum()))


def generate_reading(rng: np.random.Generator, label: int) -> dict:
    """One sensor reading for a node that is in fault state `label`."""
    power = rng.uniform(50, 200)
    voltage = rng.uniform(210, 240)
    current = rng.uniform(0, 5)
    temperature = rng.uniform(10, 30)
    env_current = rng.uniform(0, 5)

    if label == cfg.BULB_FAILURE:                       # lamp draws less power
        power = max(20.0, rng.uniform(30, 180) + rng.normal(0, 3))
    elif label == cfg.VOLTAGE_SURGE:                    # supply above 240 V
        voltage = float(np.clip(rng.normal(240, 9), 220, 260))
    elif label == cfg.SHORT_CIRCUIT:                    # very high line current
        current = float(np.clip(rng.normal(6, 1.7), 2.1, 10))
    elif label == cfg.LIGHT_FLICKERING:                 # unstable env current
        env_current = float(np.clip(rng.normal(6, 1.7), 2.1, 10))

    return {
        cfg.COL_POWER: round(float(power), 2),
        cfg.COL_VOLTAGE: round(float(voltage), 2),
        cfg.COL_CURRENT: round(float(current), 2),
        cfg.COL_TEMP: round(float(temperature), 2),
        cfg.COL_ENV_CURRENT: round(float(env_current), 2),
        cfg.COL_WEATHER: _weather(rng, label),
    }


def generate_dataset(
    n_rows: int = 34310,
    n_assets: int = 60,
    seed: int = cfg.RANDOM_STATE,
    start: str = "2025-01-01",
    freq_minutes: int = 60,
    persistence: float = 0.0,
) -> pd.DataFrame:
    """
    Build a dataset with the same schema as the real CSV.

    persistence > 0 makes a fault tend to continue on the same asset
    (useful for testing time-series features); 0 gives independent rows.
    """
    rng = np.random.default_rng(seed)
    labels = np.array(list(CLASS_COUNTS))
    probs = np.array(list(CLASS_COUNTS.values()), dtype=float)
    probs /= probs.sum()

    asset_numbers = (np.arange(n_rows) % n_assets) + 1
    step_index = np.arange(n_rows) // n_assets
    t0 = pd.Timestamp(start)
    timestamps = t0 + pd.to_timedelta(step_index * freq_minutes, unit="m")

    last_label: dict[int, int] = {}
    rows = []
    for i in range(n_rows):
        asset = int(asset_numbers[i])
        if persistence > 0 and asset in last_label and rng.random() < persistence:
            label = last_label[asset]
        else:
            label = int(rng.choice(labels, p=probs))
        last_label[asset] = label
        row = {cfg.COL_ASSET: asset, cfg.COL_TIME: timestamps[i]}
        row.update(generate_reading(rng, label))
        row[cfg.COL_TARGET] = label
        rows.append(row)

    columns = [cfg.COL_ASSET, cfg.COL_TIME] + cfg.BASE_FEATURES + [cfg.COL_TARGET]
    return pd.DataFrame(rows)[columns]


# ---------------------------------------------------------------------------
class SensorStream:
    """Simulated live telemetry for a group of streetlights."""

    def __init__(
        self,
        n_assets: int = 20,
        seed: int | None = None,
        fault_rate: float = 0.25,
        dropout_rate: float = 0.05,
        corruption_rate: float = 0.03,
        start: str | pd.Timestamp | None = None,
        step_minutes: int = 60,
    ):
        self.rng = np.random.default_rng(seed)
        self.assets = list(range(1, n_assets + 1))
        self.fault_rate = fault_rate
        self.dropout_rate = dropout_rate
        self.corruption_rate = corruption_rate
        self.step = timedelta(minutes=step_minutes)
        self.now = pd.Timestamp(start) if start is not None else pd.Timestamp.now().floor("h")
        self.offline_ticks: dict[int, int] = {a: 0 for a in self.assets}

        fault_labels = [1, 2, 3, 4]
        weights = np.array([CLASS_COUNTS[k] for k in fault_labels], dtype=float)
        self._fault_labels = fault_labels
        self._fault_probs = weights / weights.sum()

    def next_batch(self) -> pd.DataFrame:
        """Advance one time step and return the readings that arrived."""
        self.now = self.now + self.step
        rows = []
        for asset in self.assets:
            if self.offline_ticks[asset] > 0:                 # node is silent
                self.offline_ticks[asset] -= 1
                continue
            if self.rng.random() < self.dropout_rate:         # new outage begins
                self.offline_ticks[asset] = int(self.rng.integers(2, 5))
                continue

            if self.rng.random() < self.fault_rate:
                label = int(self.rng.choice(self._fault_labels, p=self._fault_probs))
            else:
                label = cfg.NO_FAULT
            reading = generate_reading(self.rng, label)

            if self.rng.random() < self.corruption_rate:      # noisy / broken sensor
                column = str(self.rng.choice(cfg.SENSOR_COLUMNS))
                reading[column] = float(self.rng.choice([np.nan, -999.0, 9999.0]))

            row = {cfg.COL_ASSET: asset, cfg.COL_TIME: self.now}
            row.update(reading)
            row["true_fault"] = label      # hidden truth, only for the demo
            rows.append(row)

        columns = [cfg.COL_ASSET, cfg.COL_TIME] + cfg.BASE_FEATURES + ["true_fault"]
        return pd.DataFrame(rows, columns=columns)


class HeartbeatMonitor:
    """Distinguishes a silent node (communication loss) from a faulty one."""

    def __init__(self, timeout_minutes: int = cfg.HEARTBEAT_TIMEOUT_MINUTES):
        self.timeout = timedelta(minutes=timeout_minutes)
        self.last_seen: dict[int, pd.Timestamp] = {}

    def register(self, asset_numbers, now) -> None:
        for a in asset_numbers:
            self.last_seen.setdefault(int(a), pd.Timestamp(now))

    def update(self, readings: pd.DataFrame) -> None:
        for asset, ts in zip(readings[cfg.COL_ASSET], readings[cfg.COL_TIME]):
            ts = pd.Timestamp(ts)
            asset = int(asset)
            if asset not in self.last_seen or ts > self.last_seen[asset]:
                self.last_seen[asset] = ts

    def offline_assets(self, now) -> pd.DataFrame:
        now = pd.Timestamp(now)
        rows = []
        for asset, seen in self.last_seen.items():
            silent = now - seen
            if silent > self.timeout:
                rows.append({
                    "asset_id": cfg.asset_id(asset),
                    "bulb_number": asset,
                    "last_seen": seen,
                    "minutes_silent": int(silent.total_seconds() // 60),
                    "status": "Communication loss (not a hardware fault)",
                })
        return pd.DataFrame(
            rows,
            columns=["asset_id", "bulb_number", "last_seen", "minutes_silent", "status"],
        )
