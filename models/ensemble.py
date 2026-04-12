"""Weighted-vote ensemble that combines all base detectors."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .base import BaseDetector, DetectorResult

logger = logging.getLogger(__name__)


class EnsembleDetector(BaseDetector):
    """Combine multiple base detectors with normalised weighted voting.

    Each member detector is queried independently. Their predictions are
    combined into a single weighted vote in ``[0, 1]``; if the vote
    exceeds ``vote_threshold``, the ensemble flags the record.
    """

    name = "ensemble"

    def __init__(self, members: List[BaseDetector], features: List[str],
                 config: Dict[str, Any]) -> None:
        super().__init__(features, config)
        if not members:
            raise ValueError("EnsembleDetector requires at least one member")
        self.members = members
        self.vote_threshold: float = float(config.get("vote_threshold", 0.5))

        total_weight = sum(m.weight for m in self.members) or 1.0
        self._normalized_weights = [m.weight / total_weight for m in self.members]

    def fit(self, records: List[Dict[str, Any]]) -> None:
        for m in self.members:
            m.fit(records)
        self.is_fitted = all(m.is_fitted for m in self.members)

    def predict(self, record: Dict[str, Any]) -> DetectorResult:
        member_results: List[DetectorResult] = [m.predict(record) for m in self.members]

        weighted_vote = sum(
            w * r.is_anomaly for w, r in zip(self._normalized_weights, member_results)
        )
        # Score = max raw score across members (already on different scales,
        # but useful as a single magnitude indicator).
        max_score = max((r.score for r in member_results), default=0.0)

        is_anom = int(weighted_vote >= self.vote_threshold)

        flagged_by = [r.detector for r in member_results if r.is_anomaly]
        return DetectorResult(
            is_anomaly=is_anom,
            score=round(weighted_vote, 6),
            detector=self.name,
            payload={
                "vote": round(weighted_vote, 4),
                "max_member_score": round(max_score, 4),
                "flagged_by": flagged_by,
                "member_results": [
                    {
                        "detector": r.detector,
                        "is_anomaly": r.is_anomaly,
                        "score": r.score,
                    }
                    for r in member_results
                ],
            },
        )

    @property
    def all_detectors(self) -> List[BaseDetector]:
        """Members + the ensemble itself, useful for metric reporting."""
        return [*self.members, self]
