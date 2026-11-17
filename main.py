"""MatchLab: Empirical Pattern-Matching Algorithm Analyzer.

Command-line entry point providing:
1. Quick algorithmic demonstration across standard, repetitive, and adversarial inputs.
2. Benchmark sweep execution with comparison metrics.
3. Empirical complexity scaling analysis.
"""

import argparse
import sys
from pathlib import Path

# Ensure package directory is in sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import BenchmarkScenario, ExperimentConfig
from analytics.complexity_estimator import ComplexityEstimator
from generators.input_generator import (
    generate_worst_case_naive,
    generate_best_case_boyer_moore,
    generate_overlapping_case,
    generate_dna_sequence,
)


def run_demonstration() -> None:
    """Run an informative demonstration comparing Naïve, KMP, and Boyer-Moore."""
    matchers = [
        NaiveStringMatcher(),
        KMPMatcher(),
        BoyerMooreMatcher(),
    ]

    scenarios = [
        (
            "Natural DNA Sequence",
            "GCATATCGCATATAGCATATCGGCATATCGCATATCGCATATA",
            "GCATATC",
        ),
        (
            "Dense Overlapping Matches ('A'*30, 'A'*4)",
            generate_overlapping_case(30, 4)[0],
            generate_overlapping_case(30, 4)[1],
        ),
        (
            "Naïve Adversarial Worst-Case ('A'*50, 'A'*4 + 'B')",
            generate_worst_case_naive(50, 5)[0],
            generate_worst_case_naive(50, 5)[1],
        ),
        (
            "Boyer-Moore Best-Case Sublinear ('A'*50, 'B'*5)",
            generate_best_case_boyer_moore(50, 5)[0],
            generate_best_case_boyer_moore(50, 5)[1],
        ),
    ]

    print("\n" + "=" * 80)
    print("MatchLab: Empirical Algorithm Comparison Demo".center(80))
    print("=" * 80)

    for title, text, pattern in scenarios:
        print(f"\nScenario: {title}")
        print(f"  Text (n={len(text)}):    {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"  Pattern (m={len(pattern)}): {pattern}")
        print("-" * 80)
        print(
            f"{'Algorithm':<15} | {'Matches':<8} | {'Prep Comps':<10} | {'Search Comps':<12} | "
            f"{'Total Comps':<11} | {'Time (ms)':<9}"
        )
        print("-" * 80)

        for matcher in matchers:
            res = matcher.search(text, pattern)
            time_ms = res.execution_time_seconds * 1000.0
            print(
                f"{res.algorithm_name:<15} | {len(res.matches):<8} | "
                f"{res.preprocessing_comparisons:<10} | {res.search_comparisons:<12} | "
                f"{res.total_comparisons:<11} | {time_ms:<9.3f}"
            )
        print("-" * 80)


def run_benchmark_cli(scenario_name: str = "random_ascii") -> None:
    """Run an empirical benchmark sweep from the command line."""
    print(f"\nExecuting Benchmark Sweep on Scenario: {scenario_name}")
    config = ExperimentConfig(
        scenario=BenchmarkScenario(scenario_name),
        text_lengths=[500, 1000, 2000, 4000],
        pattern_lengths=[10],
        trials_per_config=3,
        warmup_runs=1,
    )
    runner = BenchmarkRunner()
    results = runner.run_experiment(config)

    print("-" * 85)
    print(
        f"{'Algorithm':<14} | {'n':<6} | {'m':<4} | {'Mean Prep':<10} | "
        f"{'Mean Search':<11} | {'Mean Total':<10} | {'Throughput (c/s)':<16}"
    )
    print("-" * 85)

    for r in results:
        print(
            f"{r.algorithm_name:<14} | {r.text_length:<6} | {r.pattern_length:<4} | "
            f"{r.mean_prep_comparisons:<10.1f} | {r.mean_search_comparisons:<11.1f} | "
            f"{r.mean_total_comparisons:<10.1f} | {r.mean_throughput_chars_per_sec:<16,.0f}"
        )
    print("-" * 85)

    # Complexity estimation analysis
    print("\nEmpirical Complexity Scaling Analysis (Comparisons vs Text Length n):")
    for alg in config.algorithms:
        alg_res = [r for r in results if r.algorithm_name == alg]
        if len(alg_res) >= 2:
            xs = [float(r.text_length) for r in alg_res]
            ys = [float(r.mean_search_comparisons) for r in alg_res]
            fit = ComplexityEstimator.fit_power_law(xs, ys)
            print(
                f"  {alg:<12}: Exponent b={fit.exponent:.2f} (R²={fit.r_squared:.4f}) -> {fit.classified_complexity}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MatchLab: Empirical Pattern-Matching Algorithm Analyzer"
    )
    parser.add_argument(
        "--demo", action="store_true", help="Run algorithm comparison demo"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default=None,
        help="Run benchmark sweep for scenario (random_ascii, random_dna, repetitive, worst_case_naive, best_case_bm)",
    )

    args = parser.parse_args()

    if args.benchmark:
        run_benchmark_cli(args.benchmark)
    else:
        run_demonstration()


if __name__ == "__main__":
    main()
