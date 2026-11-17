# 🔍 MatchLab: Empirical Pattern-Matching Algorithm Analyzer

MatchLab is an empirical algorithm engineering and performance analysis platform designed to experimentally analyze, benchmark, and compare classic exact string matching algorithms:
1. **Naïve String Matching** (Brute-Force)
2. **Knuth-Morris-Pratt (KMP)**
3. **Boyer-Moore** (Bad Character + Strong Good Suffix Heuristics)

MatchLab emphasizes algorithmic correctness, clean modular architecture, rigorous comparison tracking, reproducible benchmarking, and empirical complexity validation.

---

## 📐 Algorithmic Overview & Complexity

| Algorithm | Preprocessing Time | Preprocessing Space | Best-Case Search Time | Worst-Case Search Time | Key Invariant & Strategy |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Naïve** | $O(1)$ | $O(1)$ | $O(n)$ | $O((n - m + 1) \cdot m)$ | Character-by-character window sliding. Resets text pointer on mismatch. |
| **KMP** | $O(m)$ | $O(m)$ | $O(n)$ | $O(n)$ | $\pi$ / LPS array tracking longest proper prefix which is also a suffix. Never regresses text pointer. |
| **Boyer-Moore** | $O(m + \|\Sigma\|)$ | $O(m + \|\Sigma\|)$ | $O(n / m)$ | $O(n \cdot m)$ | Right-to-left scanning combining Bad Character rule with Strong Good Suffix rule ($N$-array / Z-algorithm). Sublinear on typical inputs. |

*Notation: $n = \text{len}(\text{text})$, $m = \text{len}(\text{pattern})$, $\Sigma = \text{pattern alphabet}$.*

---

## 🏛️ Architecture & Project Structure

MatchLab adheres to a clean separation of concerns:

```
MatchLab/
├── algorithms/                  # Core algorithmic implementations
│   ├── base.py                  # BaseMatcher ABC and MatchResult dataclass contract
│   ├── naive.py                 # NaiveStringMatcher implementation
│   ├── kmp.py                   # KMPMatcher with LPS table construction
│   ├── boyer_moore.py           # BoyerMooreMatcher (Bad Character + Strong Good Suffix)
│   └── __init__.py
├── generators/                  # Empirical input & benchmark scenario generators
│   ├── input_generator.py       # Random, repetitive, periodic, adversarial, & embedded datasets
│   └── __init__.py
├── benchmarking/                # Experiment execution engine
│   ├── experiment_config.py     # Experiment configurations & predefined scenarios
│   ├── metrics.py               # TrialMetric & AggregatedMetric data structures
│   ├── benchmark_runner.py      # High-precision timer runner & statistical aggregator
│   └── __init__.py
├── analytics/                   # Data science & empirical analysis utilities
│   ├── statistics.py            # Summary stats, percentiles, confidence intervals
│   ├── complexity_estimator.py  # Power-law regression for empirical complexity orders
│   └── __init__.py
├── visualization/               # Publication-grade plotting routines (Matplotlib Agg)
│   ├── plots.py                 # Comparison vs n, execution time, and breakdown charts
│   └── __init__.py
├── dashboard/                   # Interactive Streamlit application
│   └── app.py                   # Interactive pattern search, step inspectors, & visual sweeps
├── tests/                       # Pytest verification suite
│   ├── test_naive.py            # Naïve algorithm unit and edge case tests
│   ├── test_kmp.py              # KMP and LPS table tests
│   ├── test_boyer_moore.py      # Boyer-Moore, bad char, and good suffix tests
│   ├── test_generators.py       # Input generator validation tests
│   └── test_correctness.py      # Cross-algorithm property testing against Python ground truth
├── experiments/                 # Target directory for automated experiment artifacts
├── results/                     # Target directory for exported figures and data
├── main.py                      # CLI entry point for demos and benchmark sweeps
├── requirements.txt             # Project dependencies
└── pytest.ini                  # Pytest configuration
```

---

## 🚀 Quick Start

### 1. Installation
Install project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run Verification Test Suite
Run the comprehensive Pytest suite (includes 79 unit, edge case, and randomized cross-validation tests):
```bash
pytest -v
```

### 3. Run Algorithmic Demonstration CLI
Run a formatted demonstration comparing Naïve, KMP, and Boyer-Moore across natural, repetitive, and adversarial inputs:
```bash
python MatchLab/main.py --demo
```

### 4. Run Empirical Benchmark Sweep
Execute an empirical scaling sweep across input text sizes with empirical complexity curve fitting:
```bash
python MatchLab/main.py --benchmark random_ascii
```
Available benchmark scenarios:
- `random_ascii`
- `random_dna`
- `random_binary`
- `repetitive`
- `overlapping`
- `worst_case_naive`
- `best_case_boyer_moore`

### 5. Launch Interactive Dashboard
Explore pattern matching interactively in the Streamlit web interface:
```bash
streamlit run MatchLab/dashboard/app.py
```

---

## 🔬 Core Algorithm Contract (`BaseMatcher`)

All algorithms implement `BaseMatcher` and expose a unified interface:

```python
from algorithms.kmp import KMPMatcher

matcher = KMPMatcher()
result = matcher.search(text="ABABDABACDABABCABAB", pattern="ABABCABAB")

print("Match indices:", result.matches)
print("Preprocessing comparisons:", result.preprocessing_comparisons)
print("Search comparisons:", result.search_comparisons)
print("Total comparisons:", result.total_comparisons)
print("Execution time:", result.execution_time_seconds)
```

Every implementation rigorously supports:
- **Empty patterns**: Returns every starting position `[0, 1, ..., len(text)]` with zero comparisons.
- **Patterns longer than text**: Returns `[]` without error.
- **Overlapping matches**: Accurately tracks all valid starting positions without skipping (e.g. `"AAAA"` with `"AA"` returns `[0, 1, 2]`).
- **No-match cases**: Returns `[]`.
- **Character comparison accounting**: Preprocessing comparisons (LPS computation, Z-algorithm suffix borders) and search comparisons are counted independently.
