"""Threaded producer/consumer pipeline.

Replaces a real Kafka topic with a bounded in-memory ``queue.Queue``.

* The **producer** thread pulls records from the data generator and
  pushes them onto the queue.
* The **consumer** thread(s) drain the queue, run the detector pipeline,
  buffer results into the database, and push alerts through the
  ``AlertManager``.

The processor exposes throughput metrics and a clean :meth:`stop`
method that drains in-flight work and joins every thread.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from src.alert_manager import AlertManager
from src.anomaly_detector import AnomalyDetectorPipeline
from src.data_generator import SensorDataGenerator
from src.database import Database

logger = logging.getLogger(__name__)


_SENTINEL = object()


@dataclass
class StreamMetrics:
    started_at: float = field(default_factory=time.time)
    records_processed: int = 0
    anomalies_flagged: int = 0
    alerts_fired: int = 0
    last_throughput: float = 0.0  # records / second
    queue_high_watermark: int = 0

    def snapshot(self) -> Dict[str, Any]:
        elapsed = max(time.time() - self.started_at, 1e-6)
        return {
            "elapsed_seconds": round(elapsed, 2),
            "records_processed": self.records_processed,
            "anomalies_flagged": self.anomalies_flagged,
            "alerts_fired": self.alerts_fired,
            "throughput_rps": round(self.records_processed / elapsed, 2),
            "last_throughput_rps": round(self.last_throughput, 2),
            "queue_high_watermark": self.queue_high_watermark,
        }


class StreamProcessor:
    """Glue layer between generator → detectors → database/alerts."""

    def __init__(
        self,
        config: Dict[str, Any],
        generator: SensorDataGenerator,
        detector: AnomalyDetectorPipeline,
        alert_manager: AlertManager,
        database: Database,
    ) -> None:
        stream_cfg = config["stream"]
        self.queue: "queue.Queue[Any]" = queue.Queue(maxsize=int(stream_cfg["queue_max_size"]))
        self.window_size: int = int(stream_cfg["window_size"])
        self.metrics_interval: float = float(stream_cfg.get("metrics_interval_seconds", 5))

        self.generator = generator
        self.detector = detector
        self.alert_manager = alert_manager
        self.database = database

        self.metrics = StreamMetrics()
        self.window: deque = deque(maxlen=self.window_size)

        self._stop_event = threading.Event()
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._metrics_thread: Optional[threading.Thread] = None
        self._max_records: Optional[int] = None
        self._records_in_window = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, max_records: Optional[int] = None,
              warmup_size: int = 300) -> None:
        """Spin up producer + consumer threads.

        Args:
            max_records: Stop the producer after this many records.
            warmup_size: Pre-fit detectors on this many synthetic records
                before the live stream starts. Set to ``0`` to skip.
        """
        if warmup_size > 0:
            self._warmup(warmup_size)

        self._max_records = max_records
        self._stop_event.clear()
        self.metrics = StreamMetrics()

        self._producer_thread = threading.Thread(
            target=self._producer_loop, name="producer", daemon=True
        )
        self._consumer_thread = threading.Thread(
            target=self._consumer_loop, name="consumer", daemon=True
        )
        self._metrics_thread = threading.Thread(
            target=self._metrics_loop, name="metrics", daemon=True
        )
        self._producer_thread.start()
        self._consumer_thread.start()
        self._metrics_thread.start()
        logger.info("StreamProcessor started")

    def stop(self, timeout: float = 5.0) -> None:
        """Drain the queue and join every thread."""
        self._stop_event.set()
        # Unblock the consumer if it's currently waiting on get()
        try:
            self.queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass

        for t in (self._producer_thread, self._consumer_thread, self._metrics_thread):
            if t and t.is_alive():
                t.join(timeout=timeout)

        # Final flush
        self.database.flush()
        logger.info("StreamProcessor stopped: %s", self.metrics.snapshot())

    def is_running(self) -> bool:
        return any(
            t and t.is_alive()
            for t in (self._producer_thread, self._consumer_thread)
        )

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------
    def _warmup(self, n: int) -> None:
        logger.info("Generating warmup batch of %d records", n)
        warmup = self.generator.generate_batch(n)
        # Strip ground-truth anomalies from training data so the unsupervised
        # detectors don't learn to model anomalies as normal.
        normal_only = [r for r in warmup if not r.get("is_true_anomaly")]
        self.detector.warm_up(normal_only)

    def _producer_loop(self) -> None:
        try:
            for record in self.generator.stream(max_records=self._max_records):
                if self._stop_event.is_set():
                    break
                while True:
                    try:
                        self.queue.put(record, timeout=0.5)
                        break
                    except queue.Full:
                        if self._stop_event.is_set():
                            return
                        logger.debug("Queue full, retrying...")
        except Exception:
            logger.exception("Producer crashed")
        finally:
            try:
                self.queue.put_nowait(_SENTINEL)
            except queue.Full:
                pass
            logger.info("Producer finished")

    def _consumer_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    item = self.queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                if item is _SENTINEL:
                    self.queue.task_done()
                    break

                self._handle_record(item)
                self.queue.task_done()
        except Exception:
            logger.exception("Consumer crashed")
        finally:
            logger.info("Consumer finished")

    def _handle_record(self, record: Dict[str, Any]) -> None:
        # Track queue pressure
        qsize = self.queue.qsize()
        if qsize > self.metrics.queue_high_watermark:
            self.metrics.queue_high_watermark = qsize

        # Buffer raw event
        self.database.insert_event(record)

        # Detection
        try:
            results = self.detector.process(record)
        except Exception:
            logger.exception("Detector failure on record %s", record.get("sensor_id"))
            return

        # Persist anomaly hits
        any_flag = False
        for name, result in results.items():
            if result.is_anomaly:
                any_flag = True
            self.database.insert_anomaly(
                {
                    "timestamp": record["timestamp"],
                    "sensor_id": record["sensor_id"],
                    "detector": name,
                    "score": result.score,
                    "is_anomaly": result.is_anomaly,
                    "payload": result.payload,
                }
            )

        with self._lock:
            self.metrics.records_processed += 1
            self._records_in_window += 1
            if any_flag:
                self.metrics.anomalies_flagged += 1

        # Alerts
        alert = self.alert_manager.evaluate(record["sensor_id"], results)
        if alert is not None:
            with self._lock:
                self.metrics.alerts_fired += 1

        # Sliding context window for downstream consumers (dashboard etc.)
        self.window.append(record)

    def _metrics_loop(self) -> None:
        last_count = 0
        last_ts = time.time()
        while not self._stop_event.is_set():
            time.sleep(self.metrics_interval)
            now = time.time()
            with self._lock:
                count = self.metrics.records_processed
            delta = count - last_count
            elapsed = now - last_ts
            self.metrics.last_throughput = delta / elapsed if elapsed > 0 else 0.0
            last_count = count
            last_ts = now

            # Periodic metric snapshots persisted for the dashboard
            for snap in self.detector.metrics():
                self.database.insert_metrics(
                    {
                        "timestamp": now,
                        "detector": snap["detector"],
                        "precision": snap["precision"],
                        "recall": snap["recall"],
                        "f1": snap["f1"],
                        "tp": snap["tp"],
                        "fp": snap["fp"],
                        "tn": snap["tn"],
                        "fn": snap["fn"],
                    }
                )
            logger.info("Throughput: %.1f rps | %s", self.metrics.last_throughput, self.metrics.snapshot())
