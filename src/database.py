"""SQLite persistence layer.

Stores raw events, detected anomalies, fired alerts, and rolling model
metrics. All inserts are batched behind a thread-safe lock so the
streaming consumer never blocks on disk I/O for individual records.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    sensor_id       TEXT    NOT NULL,
    temperature     REAL,
    pressure        REAL,
    vibration       REAL,
    power_consumption REAL,
    is_true_anomaly INTEGER DEFAULT 0,
    anomaly_type    TEXT
);
CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_sensor ON raw_events(sensor_id);

CREATE TABLE IF NOT EXISTS anomalies (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id        INTEGER,
    timestamp       REAL    NOT NULL,
    sensor_id       TEXT    NOT NULL,
    detector        TEXT    NOT NULL,
    score           REAL,
    is_anomaly      INTEGER NOT NULL,
    payload         TEXT,
    FOREIGN KEY(event_id) REFERENCES raw_events(id)
);
CREATE INDEX IF NOT EXISTS idx_anom_ts ON anomalies(timestamp);

CREATE TABLE IF NOT EXISTS alerts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    sensor_id       TEXT    NOT NULL,
    severity        TEXT    NOT NULL,
    detectors       TEXT    NOT NULL,
    score           REAL,
    message         TEXT,
    payload         TEXT
);
CREATE INDEX IF NOT EXISTS idx_alert_ts ON alerts(timestamp);

CREATE TABLE IF NOT EXISTS model_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       REAL    NOT NULL,
    detector        TEXT    NOT NULL,
    precision       REAL,
    recall          REAL,
    f1              REAL,
    tp              INTEGER,
    fp              INTEGER,
    tn              INTEGER,
    fn              INTEGER
);
CREATE INDEX IF NOT EXISTS idx_metrics_detector ON model_metrics(detector);
"""


class Database:
    """Thread-safe SQLite wrapper with batched writes."""

    def __init__(self, path: str | Path, batch_size: int = 50,
                 flush_interval_seconds: float = 2.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.flush_interval = flush_interval_seconds

        self._lock = threading.Lock()
        self._event_buffer: List[Tuple[Any, ...]] = []
        self._anomaly_buffer: List[Tuple[Any, ...]] = []
        self._last_flush = time.time()

        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, isolation_level=None
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()
        logger.info("Database initialized at %s", self.path)

    # ------------------------------------------------------------------
    # Schema / lifecycle
    # ------------------------------------------------------------------
    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self.flush()
        with self._lock:
            self._conn.close()
        logger.info("Database closed")

    @contextmanager
    def _cursor(self):
        with self._lock:
            cur = self._conn.cursor()
            try:
                yield cur
            finally:
                cur.close()

    # ------------------------------------------------------------------
    # Buffered writes
    # ------------------------------------------------------------------
    def insert_event(self, event: Dict[str, Any]) -> None:
        """Buffer one raw event for batched insert."""
        row = (
            event["timestamp"],
            event["sensor_id"],
            event.get("temperature"),
            event.get("pressure"),
            event.get("vibration"),
            event.get("power_consumption"),
            int(event.get("is_true_anomaly", 0)),
            event.get("anomaly_type"),
        )
        with self._lock:
            self._event_buffer.append(row)
            should_flush = (
                len(self._event_buffer) >= self.batch_size
                or (time.time() - self._last_flush) >= self.flush_interval
            )
        if should_flush:
            self.flush()

    def insert_anomaly(self, record: Dict[str, Any]) -> None:
        """Buffer a detector hit (one row per detector per record)."""
        row = (
            record.get("event_id"),
            record["timestamp"],
            record["sensor_id"],
            record["detector"],
            record.get("score"),
            int(record.get("is_anomaly", 0)),
            json.dumps(record.get("payload", {}), default=str),
        )
        with self._lock:
            self._anomaly_buffer.append(row)

    def flush(self) -> None:
        """Persist all buffered events and anomalies."""
        with self._lock:
            events = self._event_buffer
            anomalies = self._anomaly_buffer
            self._event_buffer = []
            self._anomaly_buffer = []
            self._last_flush = time.time()

            if events:
                self._conn.executemany(
                    """INSERT INTO raw_events
                       (timestamp, sensor_id, temperature, pressure,
                        vibration, power_consumption, is_true_anomaly,
                        anomaly_type)
                       VALUES (?,?,?,?,?,?,?,?)""",
                    events,
                )
            if anomalies:
                self._conn.executemany(
                    """INSERT INTO anomalies
                       (event_id, timestamp, sensor_id, detector,
                        score, is_anomaly, payload)
                       VALUES (?,?,?,?,?,?,?)""",
                    anomalies,
                )
        if events or anomalies:
            logger.debug("Flushed %d events / %d anomalies",
                         len(events), len(anomalies))

    # ------------------------------------------------------------------
    # Direct (non-buffered) inserts
    # ------------------------------------------------------------------
    def insert_alert(self, alert: Dict[str, Any]) -> int:
        """Insert one alert immediately and return its row id."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO alerts
                   (timestamp, sensor_id, severity, detectors,
                    score, message, payload)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    alert["timestamp"],
                    alert["sensor_id"],
                    alert["severity"],
                    ",".join(alert.get("detectors", [])),
                    alert.get("score"),
                    alert.get("message", ""),
                    json.dumps(alert.get("payload", {}), default=str),
                ),
            )
            return cur.lastrowid or -1

    def insert_metrics(self, metrics: Dict[str, Any]) -> None:
        """Snapshot precision/recall/F1 for one detector."""
        with self._cursor() as cur:
            cur.execute(
                """INSERT INTO model_metrics
                   (timestamp, detector, precision, recall, f1,
                    tp, fp, tn, fn)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    metrics["timestamp"],
                    metrics["detector"],
                    metrics.get("precision"),
                    metrics.get("recall"),
                    metrics.get("f1"),
                    metrics.get("tp", 0),
                    metrics.get("fp", 0),
                    metrics.get("tn", 0),
                    metrics.get("fn", 0),
                ),
            )

    # ------------------------------------------------------------------
    # Queries (used by dashboard)
    # ------------------------------------------------------------------
    def fetch_recent_events(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM raw_events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def fetch_recent_anomalies(self, limit: int = 500) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM anomalies ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def fetch_recent_alerts(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def fetch_latest_metrics(self) -> List[Dict[str, Any]]:
        """Return the most recent metric row per detector."""
        with self._cursor() as cur:
            rows = cur.execute(
                """SELECT detector, precision, recall, f1, tp, fp, tn, fn,
                          MAX(timestamp) AS timestamp
                   FROM model_metrics GROUP BY detector"""
            ).fetchall()
        return [dict(r) for r in rows]

    def event_count(self) -> int:
        with self._cursor() as cur:
            return cur.execute("SELECT COUNT(*) FROM raw_events").fetchone()[0]

    def truncate_all(self) -> None:
        """Wipe every table — used by tests and ``--reset``."""
        with self._cursor() as cur:
            for tbl in ("raw_events", "anomalies", "alerts", "model_metrics"):
                cur.execute(f"DELETE FROM {tbl}")
        logger.info("Database truncated")
