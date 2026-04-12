"""Unit tests for each anomaly detector and the ensemble."""
from __future__ import annotations

import numpy as np
import pytest

from models.base import ConfusionCounts
from models.dbscan_detector import DBSCANDetector
from models.ensemble import EnsembleDetector
from models.isolation_forest import IsolationForestDetector
from models.zscore_detector import ZScoreDetector


def _make_normal(n: int, rng: np.random.Generator):
    return [
        {
            "temperature": float(rng.normal(65, 1.5)),
            "pressure": float(rng.normal(101, 0.5)),
            "vibration": float(rng.normal(0.5, 0.05)),
            "power_consumption": float(rng.normal(220, 3)),
        }
        for _ in range(n)
    ]


def _make_anomaly(temp: float = 90.0):
    return {
        "temperature": temp,
        "pressure": 101.0,
        "vibration": 0.5,
        "power_consumption": 220.0,
    }


# ----------------------------------------------------------------------
# Z-score
# ----------------------------------------------------------------------
class TestZScoreDetector:
    def test_warm_up_then_flag(self, features):
        det = ZScoreDetector(features, {"window_size": 100, "sigma_threshold": 3.0})
        rng = np.random.default_rng(0)
        det.fit(_make_normal(100, rng))

        # Normal record should not flag
        assert det.predict(_make_normal(1, rng)[0]).is_anomaly == 0

        # 90 °C with std≈1.5 ⇒ z ≈ 16 ⇒ flagged
        result = det.predict(_make_anomaly(90.0))
        assert result.is_anomaly == 1
        assert result.score > 3.0

    def test_score_returns_max_z(self, features):
        det = ZScoreDetector(features, {"window_size": 50, "sigma_threshold": 3.0})
        rng = np.random.default_rng(1)
        det.fit(_make_normal(50, rng))
        score = det.predict(_make_anomaly(95.0)).score
        assert score > 5.0


# ----------------------------------------------------------------------
# Isolation Forest
# ----------------------------------------------------------------------
class TestIsolationForest:
    def test_warmup_returns_neutral_score(self, features):
        det = IsolationForestDetector(features, {
            "contamination": 0.05,
            "n_estimators": 50,
            "max_samples": 64,
            "initial_train_size": 200,
            "retrain_every": 200,
        })
        rng = np.random.default_rng(2)
        # Before initial_train_size, predict returns warmup result
        result = det.predict(_make_normal(1, rng)[0])
        assert result.is_anomaly == 0
        assert result.payload.get("warmup") is True

    def test_flags_extreme_value(self, features):
        det = IsolationForestDetector(features, {
            "contamination": 0.05,
            "n_estimators": 60,
            "max_samples": 64,
            "initial_train_size": 200,
            "retrain_every": 1000,
        })
        rng = np.random.default_rng(3)
        det.fit(_make_normal(300, rng))
        # Now predict an extreme outlier
        result = det.predict({
            "temperature": 200.0,
            "pressure": 50.0,
            "vibration": 5.0,
            "power_consumption": 400.0,
        })
        assert result.is_anomaly == 1


# ----------------------------------------------------------------------
# DBSCAN
# ----------------------------------------------------------------------
class TestDBSCAN:
    def test_flags_outlier_outside_cluster(self, features):
        det = DBSCANDetector(features, {"eps": 0.6, "min_samples": 5, "window_size": 100})
        rng = np.random.default_rng(4)
        det.fit(_make_normal(100, rng))
        # Push some normals through to populate the live window
        for r in _make_normal(20, rng):
            det.predict(r)
        # Then a strong outlier
        result = det.predict({
            "temperature": 150.0,
            "pressure": 70.0,
            "vibration": 4.0,
            "power_consumption": 350.0,
        })
        assert result.is_anomaly == 1


# ----------------------------------------------------------------------
# Ensemble
# ----------------------------------------------------------------------
class TestEnsemble:
    def test_combines_member_votes(self, features):
        z = ZScoreDetector(features, {"window_size": 50, "sigma_threshold": 3.0, "weight": 0.4})
        rng = np.random.default_rng(5)
        z.fit(_make_normal(50, rng))

        ens = EnsembleDetector([z], features, {"vote_threshold": 0.5})
        result = ens.predict(_make_anomaly(95.0))
        assert result.is_anomaly == 1

    def test_threshold_blocks_single_weak_detector(self, features):
        z = ZScoreDetector(features, {"window_size": 50, "sigma_threshold": 3.0, "weight": 0.2})
        z2 = ZScoreDetector(features, {"window_size": 50, "sigma_threshold": 3.0, "weight": 0.2})
        z3 = ZScoreDetector(features, {"window_size": 50, "sigma_threshold": 3.0, "weight": 0.6})
        rng = np.random.default_rng(6)
        for d in (z, z2, z3):
            d.fit(_make_normal(50, rng))

        ens = EnsembleDetector([z, z2, z3], features, {"vote_threshold": 0.7})
        # All members will flag the extreme record, vote = 1.0 > 0.7
        result = ens.predict(_make_anomaly(95.0))
        assert result.is_anomaly == 1


# ----------------------------------------------------------------------
# Confusion counts
# ----------------------------------------------------------------------
class TestConfusionCounts:
    def test_metrics(self):
        cc = ConfusionCounts()
        for pred, truth in [(1, 1), (1, 1), (1, 0), (0, 0), (0, 1)]:
            cc.update(pred, truth)
        assert cc.tp == 2
        assert cc.fp == 1
        assert cc.tn == 1
        assert cc.fn == 1
        assert pytest.approx(cc.precision(), rel=1e-3) == 2 / 3
        assert pytest.approx(cc.recall(), rel=1e-3) == 2 / 3
        assert pytest.approx(cc.f1(), rel=1e-3) == 2 / 3
