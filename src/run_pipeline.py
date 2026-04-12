"""Command-line entry point for the streaming pipeline.

Examples
--------
Run forever, writing to the configured SQLite database::

    python -m src.run_pipeline

Run for 5 000 records and exit (useful for benchmarks)::

    python -m src.run_pipeline --max-records 5000

Reset the database before starting::

    python -m src.run_pipeline --reset
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path and CWD points there, so both
# `python -m src.run_pipeline` and `python src/run_pipeline.py` work
# regardless of which directory the process starts in.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import argparse
import logging
import signal
import sys
import time
from typing import Optional

from src.alert_manager import AlertManager
from src.anomaly_detector import AnomalyDetectorPipeline
from src.config import load_config, setup_logging
from src.data_generator import SensorDataGenerator
from src.database import Database
from src.stream_processor import StreamProcessor

logger = logging.getLogger("run_pipeline")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time anomaly detection pipeline")
    parser.add_argument("--config", default=None, help="Optional config path override")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Stop after this many records")
    parser.add_argument("--warmup", type=int, default=300,
                        help="Records used for initial detector training")
    parser.add_argument("--reset", action="store_true",
                        help="Truncate the database before starting")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    setup_logging(cfg["pipeline"].get("log_level", "INFO"))

    db = Database(
        path=cfg["database"]["path"],
        batch_size=int(cfg["database"]["batch_size"]),
        flush_interval_seconds=float(cfg["database"]["flush_interval_seconds"]),
    )
    if args.reset:
        db.truncate_all()

    generator = SensorDataGenerator(cfg["data_generator"], seed=cfg["pipeline"]["random_seed"])
    detector = AnomalyDetectorPipeline(cfg)
    alert_manager = AlertManager(cfg["alerts"], database=db)

    processor = StreamProcessor(
        config=cfg,
        generator=generator,
        detector=detector,
        alert_manager=alert_manager,
        database=db,
    )

    def _shutdown(signum, _frame):  # noqa: ANN001
        logger.info("Received signal %s, shutting down...", signum)
        processor.stop()
        db.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    processor.start(max_records=args.max_records, warmup_size=args.warmup)

    # Keep main thread alive while consumer/producer run.
    try:
        while processor.is_running():
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping")
    finally:
        processor.stop()
        db.close()

    print("\n--- Final pipeline metrics ---")
    print(processor.metrics.snapshot())
    print("\n--- Detector metrics ---")
    for snap in detector.metrics():
        print(snap)
    print(f"\nTotal alerts fired: {alert_manager.total()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
