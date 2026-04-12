"""Integration tests for the streaming pipeline."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.alert_manager import AlertManager
from src.anomaly_detector import AnomalyDetectorPipeline
from src.data_generator import SensorDataGenerator
from src.database import Database
from src.stream_processor import StreamProcessor


@pytest.fixture
def fast_config(config):
    """Tweak the loaded config so tests run quickly."""
    cfg = {**config}
    cfg["data_generator"] = {**config["data_generator"]}
    cfg["data_generator"]["emit_delay_seconds"] = 0.0
    cfg["data_generator"]["anomaly_rate"] = 0.10
    cfg["data_generator"]["num_sensors"] = 2
    cfg["stream"] = {**config["stream"], "metrics_interval_seconds": 1.0}
    cfg["detectors"] = {**config["detectors"]}
    cfg["detectors"]["isolation_forest"] = {
        **config["detectors"]["isolation_forest"],
        "initial_train_size": 100, "retrain_every": 200, "n_estimators": 30,
    }
    cfg["detectors"]["dbscan"] = {**config["detectors"]["dbscan"], "window_size": 80}
    cfg["detectors"]["zscore"] = {**config["detectors"]["zscore"], "window_size": 80}
    return cfg


def test_data_generator_yields_records(fast_config):
    gen = SensorDataGenerator(fast_config["data_generator"], seed=42)
    batch = gen.generate_batch(50)
    assert len(batch) == 50
    assert all("temperature" in r for r in batch)
    assert all("sensor_id" in r for r in batch)
    summary = gen.summary()
    assert summary["total_emitted"] == 50


def test_data_generator_injects_anomalies(fast_config):
    gen = SensorDataGenerator(fast_config["data_generator"], seed=7)
    batch = gen.generate_batch(500)
    n_anom = sum(1 for r in batch if r["is_true_anomaly"])
    # With anomaly_rate=0.10 over 500 records, drift/collective bursts span
    # many records each, so the realised anomaly count is higher than the
    # injection rate alone would suggest. Just assert "non-trivial".
    assert 20 < n_anom < 500
    types = {r["anomaly_type"] for r in batch if r["is_true_anomaly"]}
    assert types  # at least one type observed


def test_pipeline_end_to_end(fast_config, tmp_path):
    db_path = tmp_path / "pipeline.db"
    db = Database(db_path, batch_size=10, flush_interval_seconds=0.5)
    gen = SensorDataGenerator(fast_config["data_generator"], seed=11)
    detector = AnomalyDetectorPipeline(fast_config)
    alerts = AlertManager({**fast_config["alerts"], "log_path": str(tmp_path / "alerts.jsonl")},
                          database=db)

    proc = StreamProcessor(fast_config, gen, detector, alerts, db)
    proc.start(max_records=600, warmup_size=150)

    # Wait for processing to finish
    deadline = time.time() + 30
    while proc.is_running() and time.time() < deadline:
        time.sleep(0.1)
    proc.stop()

    snap = proc.metrics.snapshot()
    assert snap["records_processed"] >= 500
    # At least one detector should have caught some anomalies
    metrics = detector.metrics()
    assert any(m["tp"] + m["fp"] > 0 for m in metrics)
    # And the database should hold the events
    assert db.event_count() >= snap["records_processed"]
    db.close()


def test_throughput_metrics_present(fast_config, tmp_path):
    db = Database(tmp_path / "p.db", batch_size=20, flush_interval_seconds=0.5)
    gen = SensorDataGenerator(fast_config["data_generator"], seed=13)
    detector = AnomalyDetectorPipeline(fast_config)
    alerts = AlertManager({**fast_config["alerts"], "log_path": str(tmp_path / "a.jsonl")},
                          database=db)
    proc = StreamProcessor(fast_config, gen, detector, alerts, db)
    proc.start(max_records=300, warmup_size=100)
    while proc.is_running():
        time.sleep(0.1)
    proc.stop()
    snap = proc.metrics.snapshot()
    assert snap["throughput_rps"] > 0
    db.close()
