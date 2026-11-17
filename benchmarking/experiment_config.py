"""Experiment configuration classes for parameter sweeps and empirical benchmarks."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid
from typing import List, Optional

from generators.input_generator import InputDistribution


@dataclass
class ExperimentConfig:
    """Configuration governing reproducible benchmarking sweeps across algorithms.

    Attributes:
        experiment_id: Unique identifier for tracking this experiment batch.
        name: Human-readable experiment description.
        distributions: Input distributions to evaluate.
        text_lengths: Corpus text sizes (n) to sweep across.
        pattern_lengths: Pattern lengths (m) to sweep across.
        alphabet_sizes: Alphabet cardinalities (|Σ|) to sweep across.
        algorithms: Algorithm names to evaluate (e.g. ['Naive', 'KMP', 'Boyer-Moore']).
        repetitions: Number of independent timed repetitions per configuration point.
                     Defaults to 30 for statistical rigor.
        warmup_runs: Number of unmeasured discard runs to warm caches/JIT.
        seed: Deterministic pseudo-random seed.
        created_at: ISO 8601 timestamp of configuration generation.
    """

    experiment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "empirical_comparison"
    distributions: List[InputDistribution] = field(
        default_factory=lambda: [InputDistribution.RANDOM]
    )
    text_lengths: List[int] = field(default_factory=lambda: [500, 1000, 2500, 5000])
    pattern_lengths: List[int] = field(default_factory=lambda: [10, 25, 50])
    alphabet_sizes: List[int] = field(default_factory=lambda: [26])
    algorithms: List[str] = field(
        default_factory=lambda: ["Naive", "KMP", "Boyer-Moore"]
    )
    repetitions: int = 30  # Default 30 repetitions for statistical rigor
    warmup_runs: int = 3
    seed: Optional[int] = 42
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
