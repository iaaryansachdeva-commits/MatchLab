"""Benchmark runner for executing empirical comparisons and sweeps."""

import statistics as py_stats
import time
from typing import Dict, List, Optional, Tuple

from algorithms.base import BaseMatcher
from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from benchmarking.experiment_config import BenchmarkScenario, ExperimentConfig
from benchmarking.metrics import AggregatedMetric, TrialMetric
from generators.input_generator import (
    generate_best_case_boyer_moore,
    generate_binary_sequence,
    generate_dna_sequence,
    generate_overlapping_case,
    generate_random_string,
    generate_repetitive_string,
    generate_worst_case_naive,
    Alphabet,
)


class BenchmarkRunner:
    """Orchestrates empirical benchmark runs across algorithms and configurations."""

    AVAILABLE_MATCHERS: Dict[str, BaseMatcher] = {
        "Naive": NaiveStringMatcher(),
        "KMP": KMPMatcher(),
        "Boyer-Moore": BoyerMooreMatcher(),
    }

    def __init__(self, matchers: Optional[Dict[str, BaseMatcher]] = None) -> None:
        self.matchers = matchers or self.AVAILABLE_MATCHERS

    def generate_input_pair(
        self,
        scenario: BenchmarkScenario,
        text_length: int,
        pattern_length: int,
        seed: Optional[int] = None,
    ) -> Tuple[str, str]:
        """Generate text and pattern according to scenario specification."""
        if scenario == BenchmarkScenario.RANDOM_ASCII:
            text = generate_random_string(text_length, Alphabet.LOWERCASE, seed)
            # Pick pattern from text or random
            pattern = generate_random_string(pattern_length, Alphabet.LOWERCASE, (seed + 1) if seed else None)
            return text, pattern

        elif scenario == BenchmarkScenario.RANDOM_DNA:
            text = generate_dna_sequence(text_length, seed)
            pattern = generate_dna_sequence(pattern_length, (seed + 1) if seed else None)
            return text, pattern

        elif scenario == BenchmarkScenario.RANDOM_BINARY:
            text = generate_binary_sequence(text_length, seed)
            pattern = generate_binary_sequence(pattern_length, (seed + 1) if seed else None)
            return text, pattern

        elif scenario == BenchmarkScenario.REPETITIVE:
            text = generate_repetitive_string("A", text_length)
            pattern = generate_repetitive_string("A", pattern_length)
            return text, pattern

        elif scenario == BenchmarkScenario.OVERLAPPING:
            return generate_overlapping_case(text_length, pattern_length, char="A")

        elif scenario == BenchmarkScenario.WORST_CASE_NAIVE:
            return generate_worst_case_naive(text_length, pattern_length)

        elif scenario == BenchmarkScenario.BEST_CASE_BOYER_MOORE:
            return generate_best_case_boyer_moore(text_length, pattern_length)

        else:
            raise ValueError(f"Unsupported scenario: {scenario}")

    def run_single_trial(
        self,
        matcher: BaseMatcher,
        text: str,
        pattern: str,
        scenario_name: str,
        trial_index: int,
    ) -> TrialMetric:
        """Run a single timed trial with comparison metrics."""
        t_start = time.perf_counter_ns()
        result = matcher.search(text, pattern)
        t_end = time.perf_counter_ns()
        elapsed_seconds = (t_end - t_start) / 1_000_000_000.0

        return TrialMetric(
            algorithm_name=matcher.name,
            text_length=len(text),
            pattern_length=len(pattern),
            preprocessing_comparisons=result.preprocessing_comparisons,
            search_comparisons=result.search_comparisons,
            total_comparisons=result.total_comparisons,
            execution_time_seconds=elapsed_seconds,
            match_count=len(result.matches),
            scenario=scenario_name,
            trial_index=trial_index,
        )

    def aggregate_trials(
        self,
        trials: List[TrialMetric],
        algorithm_name: str,
        scenario: str,
        text_length: int,
        pattern_length: int,
    ) -> AggregatedMetric:
        """Calculate statistical summaries over multiple trials."""
        k = len(trials)
        if k == 0:
            raise ValueError("Cannot aggregate empty trials list")

        prep_comps = [t.preprocessing_comparisons for t in trials]
        search_comps = [t.search_comparisons for t in trials]
        total_comps = [t.total_comparisons for t in trials]
        times = [t.execution_time_seconds for t in trials]

        mean_prep = py_stats.mean(prep_comps)
        mean_search = py_stats.mean(search_comps)
        mean_total = py_stats.mean(total_comps)
        std_total = py_stats.stdev(total_comps) if k > 1 else 0.0
        median_total = py_stats.median(total_comps)

        mean_time = py_stats.mean(times)
        std_time = py_stats.stdev(times) if k > 1 else 0.0
        median_time = py_stats.median(times)
        min_time = min(times)
        max_time = max(times)

        throughput = text_length / mean_time if mean_time > 0 else 0.0

        return AggregatedMetric(
            algorithm_name=algorithm_name,
            scenario=scenario,
            text_length=text_length,
            pattern_length=pattern_length,
            trial_count=k,
            mean_prep_comparisons=mean_prep,
            mean_search_comparisons=mean_search,
            mean_total_comparisons=mean_total,
            std_total_comparisons=std_total,
            median_total_comparisons=median_total,
            mean_time_seconds=mean_time,
            std_time_seconds=std_time,
            median_time_seconds=median_time,
            min_time_seconds=min_time,
            max_time_seconds=max_time,
            mean_throughput_chars_per_sec=throughput,
        )

    def run_experiment(self, config: ExperimentConfig) -> List[AggregatedMetric]:
        """Execute complete parameter sweep according to ExperimentConfig."""
        results: List[AggregatedMetric] = []

        for alg_name in config.algorithms:
            if alg_name not in self.matchers:
                continue
            matcher = self.matchers[alg_name]

            for m in config.pattern_lengths:
                for n in config.text_lengths:
                    if m > n:
                        continue

                    # Warmup runs
                    for w in range(config.warmup_runs):
                        text, pattern = self.generate_input_pair(
                            config.scenario, n, m, config.seed
                        )
                        matcher.search(text, pattern)

                    # Measured trials
                    trials: List[TrialMetric] = []
                    for trial_idx in range(config.trials_per_config):
                        seed_val = (
                            (config.seed + trial_idx)
                            if config.seed is not None
                            else None
                        )
                        text, pattern = self.generate_input_pair(
                            config.scenario, n, m, seed_val
                        )
                        trial_metric = self.run_single_trial(
                            matcher, text, pattern, str(config.scenario), trial_idx
                        )
                        trials.append(trial_metric)

                    agg = self.aggregate_trials(
                        trials,
                        matcher.name,
                        str(config.scenario),
                        n,
                        m,
                    )
                    results.append(agg)

        return results
