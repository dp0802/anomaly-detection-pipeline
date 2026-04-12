"""Tests for the AlertManager: severity, deduplication, persistence."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

import pytest

from src.alert_manager import AlertManager


@dataclass
class FakeResult:
    is_anomaly: int
    score: float = 1.0


@pytest.fixture
def alert_cfg(tmp_path):
    return {
        "cooldown_seconds": 0.5,
        "log_path": str(tmp_path / "alerts.jsonl"),
        "console_color": False,
        "severity_rules": {"low": 1, "medium": 2, "high": 3},
    }


def _make_results(votes):
    """Build a results dict from a tuple ``(if, dbscan, zscore, ensemble)``."""
    return {
        "isolation_forest": FakeResult(is_anomaly=int(votes[0])),
        "dbscan":           FakeResult(is_anomaly=int(votes[1])),
        "zscore":           FakeResult(is_anomaly=int(votes[2])),
        "ensemble":         FakeResult(is_anomaly=int(votes[3]), score=0.9),
    }


class TestSeverity:
    def test_low_when_one_detector(self, alert_cfg):
        am = AlertManager(alert_cfg)
        alert = am.evaluate("sensor_01", _make_results((1, 0, 0, 0)))
        assert alert is not None
        assert alert.severity == "LOW"

    def test_medium_when_two_detectors(self, alert_cfg):
        am = AlertManager(alert_cfg)
        alert = am.evaluate("sensor_01", _make_results((1, 1, 0, 0)))
        assert alert is not None
        assert alert.severity == "MEDIUM"

    def test_high_requires_three_plus_ensemble(self, alert_cfg):
        am = AlertManager(alert_cfg)
        alert = am.evaluate("sensor_01", _make_results((1, 1, 1, 1)))
        assert alert is not None
        assert alert.severity == "HIGH"

    def test_no_alert_when_nothing_flags(self, alert_cfg):
        am = AlertManager(alert_cfg)
        alert = am.evaluate("sensor_01", _make_results((0, 0, 0, 0)))
        assert alert is None


class TestDeduplication:
    def test_duplicate_within_cooldown_suppressed(self, alert_cfg):
        alert_cfg["cooldown_seconds"] = 5
        am = AlertManager(alert_cfg)
        first = am.evaluate("sensor_01", _make_results((1, 1, 0, 0)))
        second = am.evaluate("sensor_01", _make_results((1, 1, 0, 0)))
        assert first is not None
        assert second is None

    def test_different_severity_not_deduped(self, alert_cfg):
        alert_cfg["cooldown_seconds"] = 5
        am = AlertManager(alert_cfg)
        first = am.evaluate("sensor_01", _make_results((1, 0, 0, 0)))   # LOW
        second = am.evaluate("sensor_01", _make_results((1, 1, 1, 1)))  # HIGH
        assert first is not None
        assert second is not None

    def test_after_cooldown_alert_fires_again(self, alert_cfg):
        alert_cfg["cooldown_seconds"] = 0.2
        am = AlertManager(alert_cfg)
        first = am.evaluate("sensor_01", _make_results((1, 1, 0, 0)))
        time.sleep(0.3)
        second = am.evaluate("sensor_01", _make_results((1, 1, 0, 0)))
        assert first is not None
        assert second is not None


class TestPersistence:
    def test_writes_jsonl_log(self, alert_cfg):
        am = AlertManager(alert_cfg)
        am.evaluate("sensor_01", _make_results((1, 1, 1, 1)))
        log_path = alert_cfg["log_path"]
        with open(log_path) as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        assert len(lines) == 1
        assert lines[0]["severity"] == "HIGH"
        assert "ensemble" in lines[0]["detectors"]
