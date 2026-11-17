"""Experiment configuration classes for parameter sweeps and empirical benchmarks."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class BenchmarkScenario(str, Enum):
    """Predefined benchmark scenarios for comparative analysis."""

    RANDOM_DNA = "random_dna"
    RANDOM_ASCII = "random_ascii"
    RANDOM_BINARY = "random_binary"
    REPETITIVE = "repetitive"
    OVERLAPPING = "overlapping"
    WORST_CASE_NAIVE = "worst_case_naive"
    BEST_CASE_BOYER_MOORE = "best_case_boyer_moore"


@dataclass
class ExperimentConfig:
    """Configuration for an empirical benchmark experiment.

    Attributes:
        scenario: BenchmarkScenario enum or string name.
        text_lengths: List of corpus text sizes to evaluate.
        pattern_lengths: List of pattern lengths to evaluate.
        trials_per_config: Number of independent runs per parameter point.
        warmup_runs: Number of discard runs prior to measurement.
        algorithms: Names of algorithms to evaluate (e.g. ['Naive', 'KMP', 'Boyer-Moore']).
        seed: Random seed for deterministic reproducibility.
    """

    scenario: BenchmarkScenario = BenchmarkScenario.RANDOM_ASCII
    text_lengths: List[int] = field(default_factory=lambda: [500, 1000, 2500, 5000])
    pattern_lengths: List[int] = field(default_factory=lambda: [10, 25, 50])
    trials_per_config: int = 5
    warmup_runs: int = 1
    algorithms: List[str] = field(
        default_factory=lambda: ["Naive", "KMP", "Boyer-Moore"]
    )
    seed: Optional[int] = 42
