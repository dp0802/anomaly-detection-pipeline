"""Base class shared by every anomaly detector.

The contract is intentionally minimal:

* :meth:`fit` — train (or warm-start) on a window of records.
* :meth:`predict` — return ``1`` if a record is anomalous, ``0`` otherwise.
* :meth:`get_score` — return a continuous anomaly score (higher = more anomalous).
* :meth:`update_metrics` — accumulate confusion-matrix counts using the
  ground-truth label injected by the data generator.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectorResult:
    """Standard return type for ``predict``."""
    is_anomaly: int
    score: float
    detector: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfusionCounts:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    def update(self, predicted: int, truth: int) -> None:
        if predicted == 1 and truth == 1:
            self.tp += 1
        elif predicted == 1 and truth == 0:
            self.fp += 1
        elif predicted == 0 and truth == 0:
            self.tn += 1
        else:
            self.fn += 1

    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    def f1(self) -> float:
        p, r = self.precision(), self.recall()
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "precision": round(self.precision(), 4),
            "recall": round(self.recall(), 4),
            "f1": round(self.f1(), 4),
        }


class BaseDetector(ABC):
    """Abstract base class for streaming anomaly detectors."""

    name: str = "base"

    def __init__(self, features: List[str], config: Dict[str, Any]) -> None:
        self.features = features
        self.config = config
        self.is_fitted: bool = False
        self.metrics = ConfusionCounts()
        self.weight: float = float(config.get("weight", 1.0))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def vectorize(self, record: Dict[str, Any]) -> np.ndarray:
        """Convert a record dict to a feature vector in the configured order."""
        return np.array([float(record[f]) for f in self.features], dtype=float)

    def vectorize_batch(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Stack a list of records into a 2-D array."""
        if not records:
            return np.empty((0, len(self.features)))
        return np.vstack([self.vectorize(r) for r in records])

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(self, records: List[Dict[str, Any]]) -> None:
        """Train (or refresh) the detector on a list of records."""

    @abstractmethod
    def predict(self, record: Dict[str, Any]) -> DetectorResult:
        """Score a single record and decide if it's anomalous."""

    def get_score(self, record: Dict[str, Any]) -> float:
        """Return only the continuous anomaly score (no classification)."""
        return self.predict(record).score

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def update_metrics(self, prediction: int, truth: int) -> None:
        self.metrics.update(prediction, truth)

    def metrics_snapshot(self) -> Dict[str, Any]:
        snap = self.metrics.to_dict()
        snap["detector"] = self.name
        return snap

    def reset_metrics(self) -> None:
        self.metrics = ConfusionCounts()
