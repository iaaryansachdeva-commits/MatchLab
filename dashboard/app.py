"""Interactive Streamlit Dashboard for MatchLab.

Provides visual and empirical exploration of pattern matching algorithms:
- Single execution playground with match highlighting.
- Detailed step breakdown (LPS table, Bad Character table, Good Suffix table).
- Empirical benchmark sweep runner with comparison charts.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path when running via streamlit
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    import streamlit as st
except ImportError:
    st = None  # Graceful fallback if streamlit is imported in non-GUI environment

from algorithms.naive import NaiveStringMatcher
from algorithms.kmp import KMPMatcher
from algorithms.boyer_moore import BoyerMooreMatcher
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.experiment_config import BenchmarkScenario, ExperimentConfig
from generators.input_generator import (
    Alphabet,
    generate_random_string,
    generate_dna_sequence,
    generate_repetitive_string,
    generate_worst_case_naive,
    generate_best_case_boyer_moore,
)


def run_dashboard() -> None:
    """Entry point for Streamlit web interface."""
    if st is None:
        print("Streamlit is not installed. Run: pip install streamlit")
        return

    st.set_page_config(
        page_title="MatchLab - Pattern Matching Analyzer",
        page_icon="🔍",
        layout="wide",
    )

    st.title("🔍 MatchLab: Empirical Pattern-Matching Analyzer")
    st.markdown(
        "A rigorous empirical engineering platform comparing **Naïve**, **KMP**, and **Boyer-Moore** string matching algorithms."
    )

    tab_playground, tab_benchmarks, tab_theory = st.tabs(
        ["🎮 Interactive Playground", "📊 Empirical Benchmarks", "📖 Algorithmic Theory"]
    )

    # Tab 1: Playground
    with tab_playground:
        st.subheader("Interactive Pattern Search")
        col_gen, col_inputs = st.columns([1, 2])

        with col_gen:
            st.markdown("##### Input Presets")
            preset = st.selectbox(
                "Choose Preset",
                [
                    "Custom Input",
                    "DNA Sequence",
                    "Adversarial (Naïve Worst-Case)",
                    "Best-Case (Boyer-Moore)",
                    "Repetitive Overlapping",
                ],
            )
            preset_text = "GCATATCGCATATAGCATATCGGCATATC"
            preset_pat = "GCATATC"

            if preset == "DNA Sequence":
                preset_text = generate_dna_sequence(60, seed=12)
                preset_pat = preset_text[15:25]
            elif preset == "Adversarial (Naïve Worst-Case)":
                preset_text, preset_pat = generate_worst_case_naive(50, 6)
            elif preset == "Best-Case (Boyer-Moore)":
                preset_text, preset_pat = generate_best_case_boyer_moore(50, 6)
            elif preset == "Repetitive Overlapping":
                preset_text = "AAAAAAA"
                preset_pat = "AAA"

        with col_inputs:
            text_input = st.text_area("Corpus Text (n)", value=preset_text, height=100)
            pat_input = st.text_input("Search Pattern (m)", value=preset_pat)

        selected_algs = st.multiselect(
            "Select Algorithms to Compare",
            ["Naive", "KMP", "Boyer-Moore"],
            default=["Naive", "KMP", "Boyer-Moore"],
        )

        if st.button("Run Search", type="primary"):
            matchers = {
                "Naive": NaiveStringMatcher(),
                "KMP": KMPMatcher(),
                "Boyer-Moore": BoyerMooreMatcher(),
            }

            cols = st.columns(len(selected_algs)) if selected_algs else []
            for idx, alg_name in enumerate(selected_algs):
                matcher = matchers[alg_name]
                res = matcher.search(text_input, pat_input)

                with cols[idx]:
                    st.markdown(f"### {alg_name}")
                    st.metric("Total Matches", len(res.matches))
                    st.metric("Total Comparisons", res.total_comparisons)
                    st.write(
                        f"**Preprocessing:** {res.preprocessing_comparisons} comps | "
                        f"**Search:** {res.search_comparisons} comps"
                    )
                    st.write(f"**Execution Time:** {res.execution_time_seconds * 1000:.3f} ms")
                    st.write(f"**Match Indices:** `{res.matches}`")

                    if alg_name == "KMP" and "lps_table" in res.metadata:
                        st.write("**LPS Array (π):**")
                        st.json(res.metadata["lps_table"])
                    elif alg_name == "Boyer-Moore" and "good_suffix_table" in res.metadata:
                        st.write(f"**Full Match Shift:** {res.metadata['full_match_shift']}")
                        st.write("**Good Suffix Table:**")
                        st.json(res.metadata["good_suffix_table"])

    # Tab 2: Benchmarks
    with tab_benchmarks:
        st.subheader("Empirical Scaling Benchmarks")
        st.write("Run empirical sweeps across input lengths and measure character comparisons.")

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            scenario_choice = st.selectbox(
                "Benchmark Scenario",
                [s.value for s in BenchmarkScenario],
                index=0,
            )
        with col_b2:
            max_text_len = st.slider("Max Text Length (n)", 500, 5000, 2000, 500)
        with col_b3:
            pattern_len = st.slider("Pattern Length (m)", 5, 50, 10, 5)

        if st.button("Execute Empirical Benchmark"):
            config = ExperimentConfig(
                scenario=BenchmarkScenario(scenario_choice),
                text_lengths=[max_text_len // 4, max_text_len // 2, (3 * max_text_len) // 4, max_text_len],
                pattern_lengths=[pattern_len],
                trials_per_config=3,
                warmup_runs=1,
            )
            runner = BenchmarkRunner()
            with st.spinner("Running benchmark sweep..."):
                results = runner.run_experiment(config)

            st.success("Benchmark completed!")
            st.dataframe(
                [
                    {
                        "Algorithm": r.algorithm_name,
                        "Text Length (n)": r.text_length,
                        "Pattern Length (m)": r.pattern_length,
                        "Mean Total Comparisons": round(r.mean_total_comparisons, 1),
                        "Mean Search Comparisons": round(r.mean_search_comparisons, 1),
                        "Mean Prep Comparisons": round(r.mean_prep_comparisons, 1),
                        "Mean Time (ms)": round(r.mean_time_seconds * 1000, 4),
                        "Throughput (chars/sec)": f"{r.mean_throughput_chars_per_sec:,.0f}",
                    }
                    for r in results
                ]
            )

    # Tab 3: Algorithmic Theory
    with tab_theory:
        st.subheader("Theoretical Complexity Comparison")
        st.table(
            [
                {
                    "Algorithm": "Naïve (Brute-Force)",
                    "Best Time": "O(n)",
                    "Worst Time": "O((n - m + 1) * m)",
                    "Space": "O(1)",
                    "Key Technique": "Sliding window; resets text pointer on mismatch.",
                },
                {
                    "Algorithm": "Knuth-Morris-Pratt (KMP)",
                    "Best Time": "O(n)",
                    "Worst Time": "O(n)",
                    "Space": "O(m)",
                    "Key Technique": "LPS table (π); shifts pattern without re-evaluating text characters.",
                },
                {
                    "Algorithm": "Boyer-Moore",
                    "Best Time": "O(n / m)",
                    "Worst Time": "O(n * m)",
                    "Space": "O(m + |Σ|)",
                    "Key Technique": "Right-to-left scan; Bad Character + Strong Good Suffix heuristics.",
                },
            ]
        )


if __name__ == "__main__":
    run_dashboard()
