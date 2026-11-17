"""Statistical utility functions for empirical analysis."""

import math
import statistics as py_stats
from typing import Dict, List, Sequence


def compute_summary_stats(values: Sequence[float]) -> Dict[str, float]:
    """Compute comprehensive descriptive statistics on a sequence of numerical values.

    Args:
        values: Sequence of numeric values.

    Returns:
        Dictionary containing count, mean, median, stdev, variance, min, max, iqr.
    """
    if not values:
        raise ValueError("Cannot compute statistics on empty sequence")

    n = len(values)
    sorted_vals = sorted(values)

    mean_val = py_stats.mean(values)
    median_val = py_stats.median(values)
    stdev_val = py_stats.stdev(values) if n > 1 else 0.0
    var_val = py_stats.variance(values) if n > 1 else 0.0
    min_val = sorted_vals[0]
    max_val = sorted_vals[-1]

    # Percentiles
    p25_idx = int(0.25 * (n - 1))
    p75_idx = int(0.75 * (n - 1))
    p25_val = sorted_vals[p25_idx]
    p75_val = sorted_vals[p75_idx]
    iqr_val = p75_val - p25_val

    # 95% Confidence Interval for the mean using standard error
    stderr = stdev_val / math.sqrt(n) if n > 0 else 0.0
    ci_95_margin = 1.96 * stderr

    return {
        "count": float(n),
        "mean": mean_val,
        "median": median_val,
        "stdev": stdev_val,
        "variance": var_val,
        "min": min_val,
        "max": max_val,
        "p25": p25_val,
        "p75": p75_val,
        "iqr": iqr_val,
        "ci_95_lower": mean_val - ci_95_margin,
        "ci_95_upper": mean_val + ci_95_margin,
    }
