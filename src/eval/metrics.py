"""Aggregate metrics from reasoning traces."""

from __future__ import annotations
from dataclasses import dataclass
from ..pipeline.reasoning_pipeline import ReasoningTrace


@dataclass
class ExperimentMetrics:
    method: str
    benchmark: str
    n_problems: int
    n_correct: int
    accuracy: float
    avg_tokens: float
    avg_extra_tokens: float
    avg_steps: float
    avg_conflicts: float
    total_tokens: int

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}


def compute_metrics(traces: list[ReasoningTrace], method: str, benchmark: str) -> ExperimentMetrics:
    n = len(traces)
    if n == 0:
        return ExperimentMetrics(method=method, benchmark=benchmark,
                                 n_problems=0, n_correct=0, accuracy=0.0,
                                 avg_tokens=0.0, avg_extra_tokens=0.0,
                                 avg_steps=0.0, avg_conflicts=0.0, total_tokens=0)
    nc = sum(1 for t in traces if t.is_correct)
    tt = sum(t.total_tokens for t in traces)
    return ExperimentMetrics(
        method=method, benchmark=benchmark, n_problems=n, n_correct=nc,
        accuracy=nc / n, avg_tokens=tt / n,
        avg_extra_tokens=sum(t.total_extra_tokens for t in traces) / n,
        avg_steps=sum(len(t.steps) for t in traces) / n,
        avg_conflicts=sum(t.num_conflicts for t in traces) / n,
        total_tokens=tt,
    )
