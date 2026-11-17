"""Visualization utilities for algorithm comparison charts."""

from typing import List, Optional
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless environments
import matplotlib.pyplot as plt

from benchmarking.metrics import AggregatedMetric


def plot_comparisons_vs_text_length(
    metrics: List[AggregatedMetric],
    pattern_length: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot total comparisons vs text length across algorithms.

    Args:
        metrics: List of aggregated benchmark metrics.
        pattern_length: Optional pattern length filter.
        save_path: Optional file path to persist figure.
    """
    filtered = metrics
    if pattern_length is not None:
        filtered = [m for m in metrics if m.pattern_length == pattern_length]

    fig, ax = plt.subplots(figsize=(8, 5))

    algorithms = sorted(list(set(m.algorithm_name for m in filtered)))
    styles = {
        "Naive": {"color": "#e74c3c", "marker": "o", "linestyle": "--"},
        "KMP": {"color": "#3498db", "marker": "s", "linestyle": "-"},
        "Boyer-Moore": {"color": "#2ecc71", "marker": "^", "linestyle": "-."},
    }

    for alg in algorithms:
        alg_metrics = sorted([m for m in filtered if m.algorithm_name == alg], key=lambda x: x.text_length)
        x = [m.text_length for m in alg_metrics]
        y = [m.mean_total_comparisons for m in alg_metrics]
        style = styles.get(alg, {"color": "black", "marker": "x", "linestyle": "-"})
        ax.plot(x, y, label=alg, **style)

    ax.set_title(
        f"Total Character Comparisons vs Text Length"
        + (f" (Pattern Length m={pattern_length})" if pattern_length else "")
    )
    ax.set_xlabel("Text Length (n)")
    ax.set_ylabel("Mean Total Comparisons")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_execution_time_vs_text_length(
    metrics: List[AggregatedMetric],
    pattern_length: Optional[int] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot execution time (seconds) vs text length across algorithms."""
    filtered = metrics
    if pattern_length is not None:
        filtered = [m for m in metrics if m.pattern_length == pattern_length]

    fig, ax = plt.subplots(figsize=(8, 5))

    algorithms = sorted(list(set(m.algorithm_name for m in filtered)))
    styles = {
        "Naive": {"color": "#e74c3c", "marker": "o", "linestyle": "--"},
        "KMP": {"color": "#3498db", "marker": "s", "linestyle": "-"},
        "Boyer-Moore": {"color": "#2ecc71", "marker": "^", "linestyle": "-."},
    }

    for alg in algorithms:
        alg_metrics = sorted([m for m in filtered if m.algorithm_name == alg], key=lambda x: x.text_length)
        x = [m.text_length for m in alg_metrics]
        y = [m.mean_time_seconds * 1000.0 for m in alg_metrics]  # Convert to ms
        style = styles.get(alg, {"color": "black", "marker": "x", "linestyle": "-"})
        ax.plot(x, y, label=alg, **style)

    ax.set_title("Execution Time (ms) vs Text Length")
    ax.set_xlabel("Text Length (n)")
    ax.set_ylabel("Mean Execution Time (ms)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig


def plot_algorithm_comparison_bar(
    metrics: List[AggregatedMetric],
    text_length: int,
    pattern_length: int,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Generate a side-by-side grouped bar chart comparing Preprocessing vs Search comparisons."""
    target_metrics = [
        m for m in metrics
        if m.text_length == text_length and m.pattern_length == pattern_length
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [m.algorithm_name for m in target_metrics]
    prep = [m.mean_prep_comparisons for m in target_metrics]
    search = [m.mean_search_comparisons for m in target_metrics]

    x = range(len(names))
    width = 0.35

    ax.bar([i - width / 2 for i in x], prep, width, label="Preprocessing Comparisons", color="#f39c12")
    ax.bar([i + width / 2 for i in x], search, width, label="Search Comparisons", color="#2980b9")

    ax.set_title(f"Comparison Breakdown (Text Length n={text_length}, Pattern Length m={pattern_length})")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names)
    ax.set_ylabel("Character Comparisons")
    ax.legend()
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)

    return fig
