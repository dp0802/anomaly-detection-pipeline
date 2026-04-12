"""Isolation Forest detector with periodic retraining."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


class IsolationForestDetector(BaseDetector):
    """Tree-based anomaly detector that re-fits on a sliding training window.

    The model is fitted lazily — :meth:`predict` returns a neutral score
    until ``initial_train_size`` records have accumulated.
    """

    name = "isolation_forest"

    def __init__(self, features: List[str], config: Dict[str, Any]) -> None:
        super().__init__(features, config)
        self.contamination: float = float(config.get("contamination", 0.05))
        self.n_estimators: int = int(config.get("n_estimators", 150))
        self.max_samples: int = int(config.get("max_samples", 256))
        self.initial_train_size: int = int(config.get("initial_train_size", 500))
        self.retrain_every: int = int(config.get("retrain_every", 1000))
        self.score_threshold: float = float(config.get("score_threshold", 0.0))

        self.training_window: Deque[np.ndarray] = deque(
            maxlen=max(self.initial_train_size * 4, self.retrain_every * 2)
        )
        self.scaler: StandardScaler = StandardScaler()
        self.model: IsolationForest | None = None
        self._records_since_retrain: int = 0

    # ------------------------------------------------------------------
    def fit(self, records: List[Dict[str, Any]]) -> None:
        if not records:
            return
        X = self.vectorize_batch(records)
        for row in X:
            self.training_window.append(row)
        self._refit_if_needed(force=True)

    def _refit_if_needed(self, force: bool = False) -> None:
        n = len(self.training_window)
        # When force=True (explicit fit() call), train on whatever we have as
        # long as it's enough to be statistically meaningful. This lets
        # callers warm up with smaller batches than ``initial_train_size``.
        min_required = 50 if force else self.initial_train_size
        if n < min_required:
            return
        if not force and self._records_since_retrain < self.retrain_every:
            return

        X = np.array(self.training_window)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, n),
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(Xs)
        self.is_fitted = True
        self._records_since_retrain = 0
        logger.debug("IsolationForest re-fitted on %d samples", n)

    # ------------------------------------------------------------------
    def predict(self, record: Dict[str, Any]) -> DetectorResult:
        x = self.vectorize(record).reshape(1, -1)
        self.training_window.append(x.flatten())
        self._records_since_retrain += 1

        # Warm-up: not enough data yet
        if not self.is_fitted or self.model is None:
            self._refit_if_needed()
            return DetectorResult(
                is_anomaly=0, score=0.0, detector=self.name,
                payload={"warmup": True},
            )

        Xs = self.scaler.transform(x)
        # decision_function: higher = more normal; flip so higher = anomaly
        raw = float(self.model.decision_function(Xs)[0])
        score = -raw
        is_anom = int(raw < self.score_threshold)

        # Lazy retrain after threshold
        if self._records_since_retrain >= self.retrain_every:
            self._refit_if_needed()

        return DetectorResult(
            is_anomaly=is_anom,
            score=round(score, 6),
            detector=self.name,
            payload={"raw_decision": round(raw, 6)},
        )
