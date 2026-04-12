"""DBSCAN-based detector that flags noise points in the rolling window."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


class DBSCANDetector(BaseDetector):
    """Density-based clustering. Records that fall outside any cluster
    (label ``-1``) are reported as anomalies.

    DBSCAN does not natively support online inference, so on every call we
    refit the clustering on the current sliding window — this is cheap as
    long as ``window_size`` stays modest (≤ ~1k records).
    """

    name = "dbscan"

    def __init__(self, features: List[str], config: Dict[str, Any]) -> None:
        super().__init__(features, config)
        self.eps: float = float(config.get("eps", 0.6))
        self.min_samples: int = int(config.get("min_samples", 8))
        self.window_size: int = int(config.get("window_size", 200))
        self.window: Deque[np.ndarray] = deque(maxlen=self.window_size)

    def fit(self, records: List[Dict[str, Any]]) -> None:
        for r in records[-self.window_size:]:
            self.window.append(self.vectorize(r))
        self.is_fitted = len(self.window) >= self.min_samples

    def predict(self, record: Dict[str, Any]) -> DetectorResult:
        x = self.vectorize(record)
        self.window.append(x)

        if len(self.window) < self.min_samples * 2:
            return DetectorResult(
                is_anomaly=0, score=0.0, detector=self.name,
                payload={"warmup": True},
            )

        X = np.array(self.window)
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples, n_jobs=-1)
        labels = clusterer.fit_predict(Xs)

        # Last point in the window is the record we just appended.
        last_label = int(labels[-1])
        is_anom = int(last_label == -1)

        # Score = distance to nearest core point (rough proxy)
        if is_anom:
            core_mask = labels != -1
            if core_mask.any():
                diffs = Xs[core_mask] - Xs[-1]
                score = float(np.min(np.linalg.norm(diffs, axis=1)))
            else:
                score = 1.0
        else:
            score = 0.0

        self.is_fitted = True
        return DetectorResult(
            is_anomaly=is_anom,
            score=round(score, 6),
            detector=self.name,
            payload={"label": last_label, "n_clusters": len(set(labels)) - (1 if -1 in labels else 0)},
        )
