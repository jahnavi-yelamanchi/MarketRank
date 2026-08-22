"""Bounded decision logging plus lightweight feature-drift checks.

Production deployments should forward :class:`DecisionEvent` to durable telemetry.
The in-process log is intentionally bounded, useful for local serving and tests.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log
from threading import Lock


@dataclass(frozen=True)
class DecisionEvent:
    request_id: str
    policy: str
    fallback: str | None
    candidate_count: int
    selected_provider_id: str | None
    selected_score: float | None
    latency_ms: float
    created_at: datetime


@dataclass(frozen=True)
class DecisionQualityReport:
    decisions: int
    no_match_rate: float
    fallback_rate: float
    mean_candidate_count: float
    mean_latency_ms: float
    top_provider_exposure_share: float


@dataclass(frozen=True)
class FeatureDriftReport:
    feature_name: str
    population_stability_index: float
    drift_detected: bool
    baseline_count: int
    current_count: int


class DecisionLog:
    """Thread-safe ring buffer that never stores request PII beyond its identifier."""

    def __init__(self, max_events: int = 1_000) -> None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        self._events: deque[DecisionEvent] = deque(maxlen=max_events)
        self._lock = Lock()

    def record(self, event: DecisionEvent) -> None:
        with self._lock:
            self._events.append(event)

    def snapshot(self) -> list[DecisionEvent]:
        with self._lock:
            return list(self._events)


def summarize_decision_quality(events: list[DecisionEvent]) -> DecisionQualityReport:
    """Surface serving health before delayed conversion labels arrive."""
    if not events:
        raise ValueError("At least one decision event is required")
    providers: dict[str, int] = {}
    for event in events:
        if event.selected_provider_id:
            providers[event.selected_provider_id] = providers.get(event.selected_provider_id, 0) + 1
    total = len(events)
    return DecisionQualityReport(
        decisions=total,
        no_match_rate=sum(event.candidate_count == 0 for event in events) / total,
        fallback_rate=sum(event.fallback is not None for event in events) / total,
        mean_candidate_count=sum(event.candidate_count for event in events) / total,
        mean_latency_ms=sum(event.latency_ms for event in events) / total,
        top_provider_exposure_share=max(providers.values(), default=0) / total,
    )


def detect_feature_drift(
    feature_name: str,
    baseline: list[float],
    current: list[float],
    *,
    bins: int = 10,
    threshold: float = 0.2,
) -> FeatureDriftReport:
    """Calculate population stability index (PSI) against a training baseline.

    PSI >= 0.2 is a practical investigation threshold rather than an automatic
    rollback signal. Quantile bins are calculated only from the baseline to
    preserve the reference distribution.
    """
    if not baseline or not current:
        raise ValueError("baseline and current values are required")
    if bins < 2:
        raise ValueError("bins must be at least 2")
    if threshold <= 0:
        raise ValueError("threshold must be positive")

    edges = _quantile_edges(baseline, bins)
    baseline_counts = _histogram(baseline, edges)
    current_counts = _histogram(current, edges)
    baseline_total, current_total = len(baseline), len(current)
    smoothing = 1e-6
    psi = 0.0
    for baseline_count, current_count in zip(baseline_counts, current_counts, strict=True):
        baseline_share = (baseline_count / baseline_total) + smoothing
        current_share = (current_count / current_total) + smoothing
        psi += (current_share - baseline_share) * log(current_share / baseline_share)
    return FeatureDriftReport(
        feature_name=feature_name,
        population_stability_index=psi,
        drift_detected=psi >= threshold,
        baseline_count=baseline_total,
        current_count=current_total,
    )


def decision_event(
    *,
    request_id: str,
    policy: str,
    fallback: str | None,
    candidate_count: int,
    selected_provider_id: str | None,
    selected_score: float | None,
    latency_ms: float,
) -> DecisionEvent:
    """Create a timestamped event without leaking request attributes."""
    return DecisionEvent(
        request_id=request_id,
        policy=policy,
        fallback=fallback,
        candidate_count=candidate_count,
        selected_provider_id=selected_provider_id,
        selected_score=selected_score,
        latency_ms=latency_ms,
        created_at=datetime.now(UTC),
    )


def _quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    edges = []
    for index in range(1, bins):
        quantile_index = round(index * (len(ordered) - 1) / bins)
        edge = ordered[quantile_index]
        if not edges or edge > edges[-1]:
            edges.append(edge)
    return edges


def _histogram(values: list[float], edges: list[float]) -> list[int]:
    counts = [0] * (len(edges) + 1)
    for value in values:
        index = next((index for index, edge in enumerate(edges) if value <= edge), len(edges))
        counts[index] += 1
    return counts
