"""Empirical time complexity estimator based on log-log scaling regression.

Implements empirical power-law regression:
    log(T(n)) = log(c) + k * log(n)

Estimates the empirical scaling exponent k using ordinary least squares with strict
safeguards against zero/negative values, degenerate domains, and small samples.

Distinguishes observed empirical scaling behavior from theoretical asymptotic bounds.
"""

from dataclasses import dataclass
import math
from typing import List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ComplexityFit:
    """Results of empirical complexity curve fitting.

    Attributes:
        exponent: Estimated empirical scaling exponent k in T(n) ≈ c * n^k.
        constant: Estimated constant prefactor c.
        r_squared: Coefficient of determination (goodness of fit, in [0, 1]).
        standard_error: Standard error of the estimated exponent.
        num_points: Number of valid data points utilized in regression.
        empirical_classification: Human-readable empirical growth rate.
        asymptotic_distinction: Clarification differentiating empirical vs asymptotic bounds.
        valid: Whether the fit succeeded with sufficient statistical support.
        warning: Optional warning message if fit had edge conditions.
    """

    exponent: float
    constant: float
    r_squared: float
    standard_error: float
    num_points: int
    empirical_classification: str
    asymptotic_distinction: str
    valid: bool
    warning: Optional[str] = None


class ComplexityEstimator:
    """Estimates empirical scaling exponent k using log-log ordinary least squares."""

    @staticmethod
    def fit_log_log(
        n_values: Sequence[float | int],
        metric_values: Sequence[float | int],
    ) -> ComplexityFit:
        """Estimate scaling exponent k in metric ≈ c * n^k via log(metric) = log(c) + k * log(n).

        Safeguards:
            1. Zero-valued and negative measurements: Filtered out to avoid math domain errors.
            2. Insufficient data points: Returns invalid fit if < 2 distinct positive pairs.
            3. Degenerate n values: Detects zero variance in log(n) to avoid division by zero.

        Args:
            n_values: Independent variable values (e.g., text lengths n).
            metric_values: Dependent variable values (e.g., runtime T(n) or comparisons C(n)).

        Returns:
            ComplexityFit object with empirical exponent and diagnostic metadata.
        """
        if len(n_values) != len(metric_values):
            raise ValueError("n_values and metric_values must have identical lengths")

        # Safeguard 1 & 3: Filter non-positive values
        filtered_points: List[Tuple[float, float]] = [
            (float(n), float(m))
            for n, m in zip(n_values, metric_values)
            if n is not None and m is not None and n > 0 and m > 0
        ]

        # Safeguard 2: Insufficient data points
        if len(filtered_points) < 2:
            return ComplexityFit(
                exponent=0.0,
                constant=0.0,
                r_squared=0.0,
                standard_error=0.0,
                num_points=len(filtered_points),
                empirical_classification="Insufficient Data",
                asymptotic_distinction=(
                    "Empirical scaling cannot be evaluated with fewer than 2 positive points. "
                    "Theoretical asymptotic complexity remains unchanged."
                ),
                valid=False,
                warning="At least 2 positive data pairs required for log-log regression",
            )

        log_x = [math.log(p[0]) for p in filtered_points]
        log_y = [math.log(p[1]) for p in filtered_points]
        k_pts = len(log_x)

        mean_lx = sum(log_x) / k_pts
        mean_ly = sum(log_y) / k_pts

        var_lx = sum((lx - mean_lx) ** 2 for lx in log_x)

        # Safeguard 3: Degenerate x values (all n values identical)
        if var_lx < 1e-12:
            return ComplexityFit(
                exponent=0.0,
                constant=math.exp(mean_ly),
                r_squared=0.0,
                standard_error=0.0,
                num_points=k_pts,
                empirical_classification="Degenerate Domain (n is constant)",
                asymptotic_distinction=(
                    "Input text length n showed zero variance. "
                    "Empirical scaling requires varying input lengths."
                ),
                valid=False,
                warning="Zero variance in input lengths; cannot estimate growth rate",
            )

        cov_xy = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(log_x, log_y))
        exponent = cov_xy / var_lx
        log_c = mean_ly - exponent * mean_lx
        constant = math.exp(log_c)

        # Residuals and R^2 goodness of fit
        predicted_ly = [log_c + exponent * lx for lx in log_x]
        ss_tot = sum((ly - mean_ly) ** 2 for ly in log_y)
        ss_res = sum((ly - py) ** 2 for ly, py in zip(log_y, predicted_ly))

        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-12 else 1.0
        r_squared = max(0.0, min(1.0, r_squared))

        # Standard error of regression slope
        if k_pts > 2:
            s_res = math.sqrt(ss_res / (k_pts - 2))
            std_error = s_res / math.sqrt(var_lx)
        else:
            std_error = 0.0

        # Empirical classification
        if exponent < 0.2:
            classification = f"Empirical Sublinear / Constant (k ≈ {exponent:.2f})"
        elif 0.8 <= exponent <= 1.2:
            classification = f"Empirical Linear (k ≈ {exponent:.2f})"
        elif 1.2 < exponent <= 1.4:
            classification = f"Empirical Superlinear (k ≈ {exponent:.2f})"
        elif 1.8 <= exponent <= 2.2:
            classification = f"Empirical Quadratic (k ≈ {exponent:.2f})"
        else:
            classification = f"Empirical Polynomial (k ≈ {exponent:.2f})"

        distinction = (
            f"Observed empirical scaling exponent k={exponent:.3f} over finite range "
            f"n in [{int(filtered_points[0][0])}, {int(filtered_points[-1][0])}]. "
            "Note: Empirical scaling measures concrete wall-clock or step growth over a practical domain "
            "subject to hardware cache, interpreter overhead, and constant factors. "
            "It must not be conflated with asymptotic Big-O bounds as n -> infinity."
        )

        return ComplexityFit(
            exponent=exponent,
            constant=constant,
            r_squared=r_squared,
            standard_error=std_error,
            num_points=k_pts,
            empirical_classification=classification,
            asymptotic_distinction=distinction,
            valid=True,
        )

    @classmethod
    def fit_power_law(
        cls, x_values: Sequence[float | int], y_values: Sequence[float | int]
    ) -> ComplexityFit:
        """Alias for fit_log_log maintaining backward compatibility."""
        return cls.fit_log_log(x_values, y_values)
