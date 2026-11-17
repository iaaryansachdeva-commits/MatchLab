"""Empirical time complexity estimator based on scaling analysis and regression.

Fits power-law curves y = a * x^b to empirical comparison or execution time measurements
to estimate the empirical growth exponent b.
"""

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ComplexityFit:
    """Results of empirical complexity curve fitting."""

    exponent: float  # Empirical exponent b in y = a * x^b
    constant: float  # Empirical prefactor a
    r_squared: float  # Goodness-of-fit coefficient of determination
    classified_complexity: str  # Human-readable interpretation (e.g. O(n), O(n^2))


class ComplexityEstimator:
    """Estimates empirical computational complexity from series of (n, comparisons/time)."""

    @staticmethod
    def fit_power_law(
        x_values: Sequence[float], y_values: Sequence[float]
    ) -> ComplexityFit:
        """Fit y = a * x^b via log-log ordinary least squares regression.

        log(y) = log(a) + b * log(x)
        """
        if len(x_values) != len(y_values):
            raise ValueError("x_values and y_values must have identical length")
        if len(x_values) < 2:
            raise ValueError("At least 2 data points required for curve fitting")

        # Filter out non-positive entries
        filtered_points: List[Tuple[float, float]] = [
            (math.log(x), math.log(y))
            for x, y in zip(x_values, y_values)
            if x > 0 and y > 0
        ]

        if len(filtered_points) < 2:
            return ComplexityFit(
                exponent=0.0,
                constant=0.0,
                r_squared=0.0,
                classified_complexity="Insufficient positive data points",
            )

        log_x = [p[0] for p in filtered_points]
        log_y = [p[1] for p in filtered_points]
        k = len(log_x)

        mean_lx = sum(log_x) / k
        mean_ly = sum(log_y) / k

        numerator = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
        denominator = sum((lx - mean_lx) ** 2 for lx in log_x)

        if abs(denominator) < 1e-12:
            exponent = 0.0
            log_a = mean_ly
            r_squared = 0.0
        else:
            exponent = numerator / denominator
            log_a = mean_ly - exponent * mean_lx

            # R^2 calculation
            pred_ly = [log_a + exponent * lx for lx in log_x]
            ss_tot = sum((ly - mean_ly) ** 2 for ly in log_y)
            ss_res = sum((ly - py) ** 2 for ly, py in zip(log_y, pred_ly))
            r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0

        constant = math.exp(log_a)

        # Classification heuristic
        if exponent < 0.2:
            classified = "O(1) [Sublinear / Constant]"
        elif 0.8 <= exponent <= 1.2:
            classified = "O(n) [Linear]"
        elif 1.2 < exponent <= 1.4:
            classified = "O(n log n) [Superlinear]"
        elif 1.8 <= exponent <= 2.2:
            classified = "O(n^2) [Quadratic]"
        else:
            classified = f"O(n^{exponent:.2f}) [Polynomial]"

        return ComplexityFit(
            exponent=exponent,
            constant=constant,
            r_squared=r_squared,
            classified_complexity=classified,
        )
