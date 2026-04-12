"""Rolling-window Z-score detector."""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, List

import numpy as np

from .base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


class ZScoreDetector(BaseDetector):
    """Flag any record whose value on *any* feature exceeds N sigma from
    the rolling mean. Cheap, interpretable, and a strong baseline.
    """

    name = "zscore"

    def __init__(self, features: List[str], config: Dict[str, Any]) -> None:
        super().__init__(features, config)
        self.window_size: int = int(config.get("window_size", 150))
        self.threshold: float = float(config.get("sigma_threshold", 3.0))
        self.windows: Dict[str, Deque[float]] = {
            f: deque(maxlen=self.window_size) for f in self.features
        }

    def fit(self, records: List[Dict[str, Any]]) -> None:
        for r in records:
            for f in self.features:
                self.windows[f].append(float(r[f]))
        self.is_fitted = all(
            len(w) >= max(20, self.window_size // 2) for w in self.windows.values()
        )

    def predict(self, record: Dict[str, Any]) -> DetectorResult:
        per_feature_z: Dict[str, float] = {}
        max_abs_z = 0.0
        any_anom = False

        for f in self.features:
            window = self.windows[f]
            value = float(record[f])

            if len(window) >= 20:
                mu = float(np.mean(window))
                sigma = float(np.std(window))
                z = (value - mu) / sigma if sigma > 1e-9 else 0.0
            else:
                z = 0.0

            per_feature_z[f] = round(z, 4)
            if abs(z) > max_abs_z:
                max_abs_z = abs(z)
            if abs(z) > self.threshold:
                any_anom = True

            window.append(value)

        self.is_fitted = True
        return DetectorResult(
            is_anomaly=int(any_anom),
            score=round(max_abs_z, 6),
            detector=self.name,
            payload={"z_scores": per_feature_z, "threshold": self.threshold},
        )
