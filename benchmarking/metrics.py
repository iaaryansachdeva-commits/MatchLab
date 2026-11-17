"""Metrics data structures for benchmarking pattern matching algorithms."""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class TrialMetric:
    """Metrics collected from a single benchmark execution trial."""

    algorithm_name: str
    text_length: int
    pattern_length: int
    preprocessing_comparisons: int
    search_comparisons: int
    total_comparisons: int
    execution_time_seconds: float
    match_count: int
    scenario: str
    trial_index: int


@dataclass
class AggregatedMetric:
    """Statistical summary over multiple benchmark trials for a specific configuration."""

    algorithm_name: str
    scenario: str
    text_length: int
    pattern_length: int
    trial_count: int

    # Comparison metrics
    mean_prep_comparisons: float
    mean_search_comparisons: float
    mean_total_comparisons: float
    std_total_comparisons: float
    median_total_comparisons: float

    # Timing metrics
    mean_time_seconds: float
    std_time_seconds: float
    median_time_seconds: float
    min_time_seconds: float
    max_time_seconds: float

    # Throughput (characters of text searched per second)
    mean_throughput_chars_per_sec: float

    metadata: Dict[str, Any] = field(default_factory=dict)
