"""Alert lifecycle: severity assignment, deduplication, persistence.

The alert manager translates raw detector results into actionable alerts.
It enforces a per-(sensor, severity) cooldown so a single noisy event
doesn't generate hundreds of duplicates, color-codes the console output,
and writes a JSONL audit log for downstream tooling.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    _HAS_COLOR = True
except ImportError:  # pragma: no cover
    _HAS_COLOR = False

    class _Stub:
        def __getattr__(self, _: str) -> str:  # type: ignore[override]
            return ""

    Fore = Style = _Stub()  # type: ignore[assignment]

from src.database import Database

logger = logging.getLogger(__name__)


SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3}


@dataclass
class Alert:
    timestamp: float
    sensor_id: str
    severity: str
    detectors: List[str]
    score: float
    message: str
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "severity": self.severity,
            "detectors": self.detectors,
            "score": self.score,
            "message": self.message,
            "payload": self.payload,
        }


class AlertManager:
    """Generate, deduplicate, persist and announce alerts."""

    def __init__(self, config: Dict[str, Any], database: Optional[Database] = None) -> None:
        self.cfg = config
        self.cooldown: float = float(config.get("cooldown_seconds", 30))
        self.console_color: bool = bool(config.get("console_color", True)) and _HAS_COLOR
        self.severity_rules = config.get("severity_rules", {"low": 1, "medium": 2, "high": 3})

        log_path = config.get("log_path", "logs/alerts.jsonl")
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self.database = database
        self._lock = threading.Lock()
        self._last_fired: Dict[Tuple[str, str], float] = {}
        self._alerts: List[Alert] = []

    # ------------------------------------------------------------------
    def evaluate(self, sensor_id: str, results: Dict[str, Any]) -> Optional[Alert]:
        """Inspect detector results for one record. Return an Alert if any.

        Args:
            sensor_id: Source sensor identifier.
            results: Mapping detector_name -> DetectorResult-like object.
        """
        member_names = [n for n in results if n != "ensemble"]
        flagged_members = [n for n in member_names if results[n].is_anomaly]
        ensemble_flagged = bool(results.get("ensemble") and results["ensemble"].is_anomaly)

        n_flagged = len(flagged_members)
        if n_flagged == 0 and not ensemble_flagged:
            return None

        severity = self._classify_severity(n_flagged, ensemble_flagged, len(member_names))
        if severity is None:
            return None

        alert = Alert(
            timestamp=time.time(),
            sensor_id=sensor_id,
            severity=severity,
            detectors=flagged_members + (["ensemble"] if ensemble_flagged else []),
            score=float(results["ensemble"].score) if "ensemble" in results else 0.0,
            message=self._format_message(sensor_id, severity, flagged_members),
            payload={
                name: {
                    "is_anomaly": results[name].is_anomaly,
                    "score": results[name].score,
                }
                for name in results
            },
        )

        if self._is_duplicate(alert):
            return None

        self._record(alert)
        return alert

    # ------------------------------------------------------------------
    def _classify_severity(self, n_flagged: int, ensemble_flagged: bool,
                           n_members: int) -> Optional[str]:
        high = int(self.severity_rules.get("high", 3))
        medium = int(self.severity_rules.get("medium", 2))
        low = int(self.severity_rules.get("low", 1))

        if n_flagged >= high and ensemble_flagged:
            return "HIGH"
        if n_flagged >= medium:
            return "MEDIUM"
        if n_flagged >= low:
            return "LOW"
        return None

    def _format_message(self, sensor_id: str, severity: str,
                        detectors: List[str]) -> str:
        det = ", ".join(detectors) if detectors else "n/a"
        return f"[{severity}] Anomaly on {sensor_id} flagged by: {det}"

    def _is_duplicate(self, alert: Alert) -> bool:
        key = (alert.sensor_id, alert.severity)
        now = alert.timestamp
        with self._lock:
            last = self._last_fired.get(key, 0.0)
            if now - last < self.cooldown:
                return True
            self._last_fired[key] = now
        return False

    def _record(self, alert: Alert) -> None:
        with self._lock:
            self._alerts.append(alert)

        # Console
        self._print_alert(alert)

        # JSONL log
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(alert.to_dict(), default=str) + "\n")
        except OSError as exc:
            logger.warning("Failed writing alert log: %s", exc)

        # Database
        if self.database is not None:
            try:
                self.database.insert_alert(alert.to_dict())
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed persisting alert: %s", exc)

    def _print_alert(self, alert: Alert) -> None:
        color = ""
        reset = ""
        if self.console_color:
            reset = Style.RESET_ALL
            color = {
                "LOW": Fore.YELLOW,
                "MEDIUM": Fore.MAGENTA,
                "HIGH": Fore.RED + Style.BRIGHT,
            }.get(alert.severity, "")
        ts = time.strftime("%H:%M:%S", time.localtime(alert.timestamp))
        print(f"{color}[{ts}] {alert.message} (score={alert.score:.3f}){reset}")

    # ------------------------------------------------------------------
    def recent(self, n: int = 20) -> List[Alert]:
        with self._lock:
            return list(self._alerts[-n:])

    def total(self) -> int:
        with self._lock:
            return len(self._alerts)
