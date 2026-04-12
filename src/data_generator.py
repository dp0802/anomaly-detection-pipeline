"""Synthetic IoT sensor stream.

Generates a continuous stream of records mimicking industrial sensor
telemetry (temperature, pressure, vibration, power consumption) and
periodically injects one of five anomaly archetypes:

* **point**         — single-record spike on one feature
* **contextual**    — value normal in absolute terms, abnormal for the hour
* **collective**    — short run of slightly elevated readings
* **drift**         — slow gradient applied over many records
* **multivariate**  — each feature individually fine, the combination is not

Records emitted by :meth:`SensorDataGenerator.stream` carry an
``is_true_anomaly`` ground-truth label that downstream evaluation uses
to compute precision/recall.
"""
from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SensorState:
    """Per-sensor mutable state used to model drift and collective bursts."""
    sensor_id: str
    drift_offset: float = 0.0
    drift_remaining: int = 0
    collective_remaining: int = 0
    collective_bias: float = 0.0


@dataclass
class GeneratorStats:
    total_emitted: int = 0
    total_anomalies: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)


class SensorDataGenerator:
    """Yield realistic sensor records with controllable anomaly injection."""

    ANOMALY_TYPES = ("point", "contextual", "collective", "drift", "multivariate")

    def __init__(self, config: Dict[str, Any], seed: Optional[int] = None) -> None:
        self.cfg = config
        self.num_sensors: int = config["num_sensors"]
        self.anomaly_rate: float = config["anomaly_rate"]
        self.emit_delay: float = config.get("emit_delay_seconds", 0.05)
        self.feature_ranges: Dict[str, Dict[str, float]] = config["feature_ranges"]
        self.seasonal = config.get("seasonal", {})
        self.weights = config.get("anomaly_types", {})

        # RNG — exposed for reproducibility
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        self.sensors: List[SensorState] = [
            SensorState(sensor_id=f"sensor_{i:02d}") for i in range(self.num_sensors)
        ]
        self.stats = GeneratorStats(by_type={t: 0 for t in self.ANOMALY_TYPES})
        self._sensor_index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def stream(self, max_records: Optional[int] = None) -> Generator[Dict[str, Any], None, None]:
        """Yield one sensor reading at a time.

        Args:
            max_records: Stop after this many records. ``None`` ⇒ infinite.
        """
        emitted = 0
        while max_records is None or emitted < max_records:
            sensor = self.sensors[self._sensor_index]
            self._sensor_index = (self._sensor_index + 1) % self.num_sensors

            record = self._build_normal_record(sensor)

            if self._should_inject_anomaly(sensor):
                record = self._inject_anomaly(record, sensor)

            self.stats.total_emitted += 1
            if record["is_true_anomaly"]:
                self.stats.total_anomalies += 1
                self.stats.by_type[record["anomaly_type"]] += 1

            emitted += 1
            yield record

            if self.emit_delay > 0:
                time.sleep(self.emit_delay)

    def generate_batch(self, n: int) -> List[Dict[str, Any]]:
        """Convenience: collect ``n`` records into a list (no sleep)."""
        original_delay = self.emit_delay
        self.emit_delay = 0.0
        try:
            return list(self.stream(max_records=n))
        finally:
            self.emit_delay = original_delay

    # ------------------------------------------------------------------
    # Normal-data generation
    # ------------------------------------------------------------------
    def _build_normal_record(self, sensor: SensorState) -> Dict[str, Any]:
        ts = time.time()
        seasonal_temp = self._seasonal_component(ts)

        temp = self._sample("temperature") + seasonal_temp + sensor.drift_offset
        pressure = self._sample("pressure")
        vibration = self._sample("vibration")
        power = self._sample("power")

        record: Dict[str, Any] = {
            "timestamp": ts,
            "sensor_id": sensor.sensor_id,
            "temperature": float(round(temp, 4)),
            "pressure": float(round(pressure, 4)),
            "vibration": float(round(vibration, 4)),
            "power_consumption": float(round(power, 4)),
            "is_true_anomaly": 0,
            "anomaly_type": None,
        }

        # Decay drift counter
        if sensor.drift_remaining > 0:
            sensor.drift_remaining -= 1
            if sensor.drift_remaining == 0:
                sensor.drift_offset = 0.0
            else:
                # mark drift records as anomalies for ground truth
                record["is_true_anomaly"] = 1
                record["anomaly_type"] = "drift"

        # Decay collective burst
        if sensor.collective_remaining > 0:
            record["temperature"] += sensor.collective_bias
            sensor.collective_remaining -= 1
            record["is_true_anomaly"] = 1
            record["anomaly_type"] = "collective"

        return record

    def _sample(self, feature: str) -> float:
        rng = self.feature_ranges[feature]
        v = self.rng.normal(rng["mean"], rng["std"])
        return float(np.clip(v, rng["min"], rng["max"]))

    def _seasonal_component(self, ts: float) -> float:
        amp = float(self.seasonal.get("daily_amplitude", 0.0))
        period = float(self.seasonal.get("period_seconds", 86400.0))
        if amp == 0.0 or period == 0.0:
            return 0.0
        return amp * math.sin(2 * math.pi * (ts % period) / period)

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------
    def _should_inject_anomaly(self, sensor: SensorState) -> bool:
        # Don't double-inject while a collective/drift event is active.
        if sensor.collective_remaining > 0 or sensor.drift_remaining > 0:
            return False
        return self.rng.random() < self.anomaly_rate

    def _pick_type(self) -> str:
        types = list(self.weights.keys()) or list(self.ANOMALY_TYPES)
        weights = [self.weights.get(t, 1.0) for t in types]
        total = sum(weights)
        probs = [w / total for w in weights]
        return str(self.rng.choice(types, p=probs))

    def _inject_anomaly(self, record: Dict[str, Any],
                        sensor: SensorState) -> Dict[str, Any]:
        kind = self._pick_type()
        record["is_true_anomaly"] = 1
        record["anomaly_type"] = kind

        if kind == "point":
            feature = self.rng.choice([
                "temperature", "pressure", "vibration", "power_consumption"
            ])
            rng = self.feature_ranges[
                "power" if feature == "power_consumption" else feature
            ]
            spike = rng["std"] * self.rng.uniform(6, 10)
            sign = self.rng.choice([-1, 1])
            record[feature] = float(round(record[feature] + sign * spike, 4))

        elif kind == "contextual":
            # Force a "night-time" temperature value during a "day-time" hour
            mean = self.feature_ranges["temperature"]["mean"]
            std = self.feature_ranges["temperature"]["std"]
            record["temperature"] = float(round(mean - 4 * std, 4))

        elif kind == "collective":
            sensor.collective_remaining = int(self.rng.integers(5, 12))
            sensor.collective_bias = float(self.rng.uniform(2.5, 4.0))
            record["temperature"] = float(round(
                record["temperature"] + sensor.collective_bias, 4
            ))

        elif kind == "drift":
            sensor.drift_remaining = int(self.rng.integers(40, 80))
            sensor.drift_offset = float(self.rng.uniform(2.0, 4.0))
            record["temperature"] = float(round(
                record["temperature"] + sensor.drift_offset, 4
            ))

        elif kind == "multivariate":
            # Push every feature toward the edge of its envelope without
            # any single feature looking individually extreme.
            for feature, key in (
                ("temperature", "temperature"),
                ("pressure", "pressure"),
                ("vibration", "vibration"),
                ("power_consumption", "power"),
            ):
                rng = self.feature_ranges[key]
                shift = rng["std"] * self.rng.uniform(2.0, 2.5)
                sign = self.rng.choice([-1, 1])
                record[feature] = float(round(record[feature] + sign * shift, 4))

        return record

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        return {
            "total_emitted": self.stats.total_emitted,
            "total_anomalies": self.stats.total_anomalies,
            "anomaly_rate_actual": (
                self.stats.total_anomalies / self.stats.total_emitted
                if self.stats.total_emitted else 0.0
            ),
            "by_type": dict(self.stats.by_type),
        }
