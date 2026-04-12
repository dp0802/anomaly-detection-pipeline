"""Coordinator for the model layer.

Wires the four detectors (Isolation Forest, DBSCAN, Z-Score and the
Ensemble) into a single object the stream processor can call.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from models.base import BaseDetector, DetectorResult
from models.dbscan_detector import DBSCANDetector
from models.ensemble import EnsembleDetector
from models.isolation_forest import IsolationForestDetector
from models.zscore_detector import ZScoreDetector

logger = logging.getLogger(__name__)


class AnomalyDetectorPipeline:
    """One-stop shop the consumer thread talks to."""

    def __init__(self, config: Dict[str, Any]) -> None:
        ens_cfg = config["ensemble"]
        det_cfg = config["detectors"]
        self.features: List[str] = ens_cfg["features"]

        self.members: List[BaseDetector] = []
        if det_cfg["isolation_forest"].get("enabled", True):
            self.members.append(IsolationForestDetector(self.features, det_cfg["isolation_forest"]))
        if det_cfg["dbscan"].get("enabled", True):
            self.members.append(DBSCANDetector(self.features, det_cfg["dbscan"]))
        if det_cfg["zscore"].get("enabled", True):
            self.members.append(ZScoreDetector(self.features, det_cfg["zscore"]))

        if not self.members:
            raise ValueError("No detectors enabled in configuration")

        self.ensemble = EnsembleDetector(self.members, self.features, ens_cfg)
        logger.info(
            "Initialized pipeline with detectors: %s",
            [m.name for m in self.members],
        )

    # ------------------------------------------------------------------
    def warm_up(self, records: List[Dict[str, Any]]) -> None:
        """Pre-fit detectors with an initial training window."""
        if not records:
            return
        logger.info("Warming up detectors on %d records", len(records))
        self.ensemble.fit(records)

    def process(self, record: Dict[str, Any]) -> Dict[str, DetectorResult]:
        """Score a record with every member and the ensemble.

        Returns a dict keyed by detector name. The function also updates
        per-detector confusion-matrix counts using the ground truth label.
        """
        results: Dict[str, DetectorResult] = {}

        # Run each member individually so each gets its window updated.
        for member in self.members:
            r = member.predict(record)
            results[member.name] = r
            member.update_metrics(r.is_anomaly, int(record.get("is_true_anomaly", 0)))

        # Build the ensemble result by reusing already-computed member outputs
        weighted_vote = sum(
            (m.weight / sum(x.weight for x in self.members))
            * results[m.name].is_anomaly
            for m in self.members
        )
        is_anom = int(weighted_vote >= self.ensemble.vote_threshold)
        flagged_by = [name for name, r in results.items() if r.is_anomaly]

        ens_result = DetectorResult(
            is_anomaly=is_anom,
            score=round(weighted_vote, 6),
            detector=self.ensemble.name,
            payload={
                "vote": round(weighted_vote, 4),
                "flagged_by": flagged_by,
                "member_results": [
                    {
                        "detector": r.detector,
                        "is_anomaly": r.is_anomaly,
                        "score": r.score,
                    }
                    for r in results.values()
                ],
            },
        )
        results[self.ensemble.name] = ens_result
        self.ensemble.update_metrics(is_anom, int(record.get("is_true_anomaly", 0)))

        return results

    # ------------------------------------------------------------------
    def metrics(self) -> List[Dict[str, Any]]:
        """Return a metrics snapshot for every detector + the ensemble."""
        return [d.metrics_snapshot() for d in [*self.members, self.ensemble]]
