# MatchLab: Scientific Pattern-Matching Performance Analyzer

MatchLab is an algorithm engineering and empirical performance analysis system designed to evaluate, benchmark, and compare three canonical exact string-matching algorithms:
1. **Naïve (Brute-Force) String Matching**
2. **Knuth-Morris-Pratt (KMP)**
3. **Boyer-Moore** (Bad Character and Strong Good Suffix Heuristics)

MatchLab combines algorithm correctness with experimental methodology: high-precision nanosecond timing (`time.perf_counter_ns`), independent character comparison tracking, non-polluted execution boundaries, reproducible dataset generation, tabular Pandas processing, and empirical log-log complexity scaling regression.

---

## Architecture & Project Structure

The platform enforces strict separation of concerns across its architectural layers:

```
MatchLab/
├── algorithms/                  # Algorithmic engine & execution contracts
│   ├── base.py                  # BaseMatcher ABC & MatchResult contract
│   ├── naive.py                 # NaiveStringMatcher (brute-force sliding window)
│   ├── kmp.py                   # KMPMatcher (LPS computation & isolated search)
│   ├── boyer_moore.py           # BoyerMooreMatcher (Bad Char + Good Suffix via Z-algorithm)
│   └── __init__.py
├── generators/                  # Reproducible input & distribution generators
│   ├── input_generator.py       # 11 input distributions, dynamic alphabet pools, and seed management
│   └── __init__.py
├── benchmarking/                # Experimental benchmarking framework
│   ├── experiment_config.py     # ExperimentConfig schema with default 30 repetitions
│   ├── metrics.py               # RawMeasurement & AggregatedMetric models with Pandas conversion
│   ├── experiment_manager.py    # Experiment persistence, tabular processing, and CSV/JSON export
│   ├── benchmark_runner.py      # Centralized runner: warmups, nanosecond timing, shared inputs
│   └── __init__.py
├── analytics/                   # Mathematical & empirical analysis
│   ├── statistics.py            # Summary statistics (mean, median, stdev, min, max, IQR, 95% CI)
│   ├── complexity_estimator.py  # Empirical log-log regression: log(T(n)) = log(c) + k * log(n)
│   └── __init__.py
├── visualization/               # Presentation & charting layer
│   ├── plots.py                 # Interactive Altair charts & publication-grade Matplotlib figures
│   └── __init__.py
├── dashboard/                   # Scientific Streamlit analysis interface
│   └── app.py                   # Multi-panel analytical workspace (Linear/Vercel design system)
├── tests/                       # Pytest verification suite (109 tests)
│   ├── test_naive.py            # Naïve unit, edge case, and comparison tests
│   ├── test_kmp.py              # KMP and LPS table tests
│   ├── test_boyer_moore.py      # Boyer-Moore heuristics and sublinear tests
│   ├── test_generators.py       # Generator parameter validation tests
│   ├── test_correctness.py      # Cross-algorithm property testing against Python ground truth
│   ├── test_benchmark_engine.py # Benchmarking framework and reproducibility tests
│   └── test_visualization.py    # Chart builders and figure rendering tests
├── main.py                      # CLI entry point for demos, benchmarks, and data exports
├── requirements.txt             # Project dependencies
└── pytest.ini                  # Pytest configuration
```

---

## Algorithm Descriptions & Theoretical Bounds

| Algorithm | Preprocessing Time | Preprocessing Space | Best-Case Search Time | Worst-Case Search Time | Core Invariant & Shift Logic |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Naïve** | $O(1)$ | $O(1)$ | $O(n)$ | $O((n - m + 1) \cdot m)$ | Slides pattern character-by-character. Resets text pointer on mismatch. |
| **KMP** | $O(m)$ | $O(m)$ | $O(n)$ | $O(n)$ | Precomputes $\pi$ / LPS array (longest proper prefix which is also a suffix). Advances without regressing text index $i$. |
| **Boyer-Moore** | $O(m + \|\Sigma\|)$ | $O(m + \|\Sigma\|)$ | $O(n / m)$ | $O(n \cdot m)$ | Right-to-left scan combining Bad Character rule with Strong Good Suffix rule ($N$-array via Z-algorithm). Skips up to $m$ characters per mismatch. |

*Notation: $n = \text{len}(\text{text})$, $m = \text{len}(\text{pattern})$, $\Sigma = \text{pattern alphabet}$.*

---

## Experimental Methodology

To achieve reproducible and unpolluted performance measurements, MatchLab adheres to the following principles:

1. **Isolation of Measurement Windows**: Input generation, Pandas conversions, plotting, and filesystem serialization are executed strictly outside measured blocks.
2. **Nanosecond Timer Resolution**: Timing is measured using `time.perf_counter_ns()`, converting to seconds or milliseconds only during post-run aggregation.
3. **Phase Separation**: Preprocessing time/comparisons and search time/comparisons are recorded independently:
   $$\text{total\_time} = \text{preprocessing\_time} + \text{search\_time}$$
4. **Controlled Input Sharing**: For any evaluation point $(distribution, n, m, |\Sigma|)$, the identical corpus and pattern pair is passed to all evaluated algorithms.
5. **Cache & Bytecode Warm-up**: Multiple unmeasured warm-up iterations precede timed runs to prime CPU instruction caches, memory prefetchers, and Python bytecode caches.
6. **Central Limit Theorem Compliance**: Default 30 independent repetitions are collected per parameter configuration point to enable statistically meaningful estimation of mean, median, standard deviation, and confidence intervals.

---

## Installation & Setup

### Requirements
- Python 3.10+
- Dependencies listed in `requirements.txt`:
  - `pytest>=7.0.0`
  - `pandas>=2.0.0`
  - `altair>=5.0.0`
  - `matplotlib>=3.5.0`
  - `streamlit>=1.20.0`

### Installation
```bash
pip install -r requirements.txt
```

---

## Execution Instructions

### 1. Run Verification Test Suite
Execute the complete test suite (109 passing tests):
```bash
pytest -v
```

### 2. Algorithmic Comparison Demo
Run a terminal demonstration comparing Naïve, KMP, and Boyer-Moore across natural DNA, repetitive, and adversarial inputs:
```bash
python MatchLab/main.py --demo
```

### 3. Command-Line Benchmark Sweep
Execute a systematic empirical benchmark sweep with Pandas tabular output and automatic CSV/JSON serialization:
```bash
python MatchLab/main.py --benchmark random --repetitions 30 --export-dir results
```
Available input distributions:
- `random`: Uniform pseudo-random text and pattern.
- `random_with_match`: Random text with pattern guaranteed present.
- `repetitive`: Homogeneous single-character sequences.
- `periodic`: Periodic repeating strings.
- `adversarial_prefix`: Repetitive prefix with terminal mismatch (triggers $O(n \cdot m)$ in Naïve).
- `best_case`: Disjoint alphabet characters (Boyer-Moore sublinear skips).
- `absent`: Pattern guaranteed absent from text.
- `match_at_start`, `match_at_middle`, `match_at_end`: Controlled single-match placements.
- `overlapping`: Dense overlapping occurrences.

---

## Interactive Dashboard (Streamlit)

Launch the scientific Streamlit analytical workspace:
```bash
streamlit run MatchLab/dashboard/app.py
```

### Design Philosophy
The dashboard adheres to a minimal, monochromatic, data-dense scientific design system inspired by developer tools like Linear and Vercel Analytics:
- Zero decorative animations, gradients, glassmorphism, or emojis.
- High visual hierarchy prioritizing readable charts and concise metrics.
- High-contrast, color-blind accessible algorithm palette:
  - **Naïve**: Crimson (`#dc2626`)
  - **KMP**: Cobalt (`#2563eb`)
  - **Boyer-Moore**: Emerald (`#059669`)

### Workspace Sections
1. **Overview**: Key performance indicators (fastest algorithm in context, median search runtime, total comparisons, preprocessing overhead ratio, and execution provenance).
2. **Benchmark Lab**: Parametric configuration panel controlling algorithm selection, corpus lengths ($n$), pattern lengths ($m$), alphabet sizes ($|\Sigma|$), generative distributions, and repetition counts.
3. **Algorithm Comparison**: Interactive Altair analytical graphs:
   - Search runtime vs. corpus length ($n$).
   - Character comparisons vs. corpus length ($n$).
   - Preprocessing vs. search comparison breakdown.
   - Runtime sensitivity across input distributions.
   - Runtime scaling vs. pattern length ($m$).
4. **Complexity Analysis**: Empirical log-log ordinary least squares regression estimating scaling exponent $k$, goodness of fit $R^2$, regression line visualization, and theoretical asymptotic comparison.
5. **Results Explorer**: Tabular browser for raw individual repetitions and aggregated metrics with multi-column filtering and direct CSV/JSON export downloads.

---

## Interpretation of Empirical Complexity Results

MatchLab fits empirical data using the power-law relation:
$$T(n) \approx c \cdot n^k \iff \log(T(n)) = \log(c) + k \cdot \log(n)$$

Ordinary least squares regression over $(\log(n), \log(T(n)))$ or $(\log(n), \log(C(n)))$ yields the empirical scaling exponent $k$:
- **$k \approx 1.00$ ($R^2 \ge 0.99$)**: Indicates empirical linear scaling with text length $n$.
- **$k < 1.00$**: Indicates empirical sublinear behavior (frequently observed in Boyer-Moore over large alphabets where shifts exceed 1).
- **$k \approx 2.00$**: Indicates quadratic scaling (observed in Naïve matching under adversarial repetitive-prefix distributions).

---

## Limitations of Runtime Benchmarking

When interpreting empirical runtime results, engineers must account for practical hardware and software realities:

1. **Hardware Caching Effects**: Real CPUs utilize multi-tiered cache hierarchies (L1, L2, L3). Corpora smaller than the L1 cache execute with negligible memory latency, whereas large texts cause cache misses and memory bus saturation that can distort empirical slope estimates.
2. **CPython Interpreter Overhead**: In high-level interpreted languages, loop overhead and variable lookups can overshadow low-level algorithmic differences on small inputs ($n < 200$).
3. **Finite Domain vs. Asymptotic Limits**: Empirical measurements evaluate a discrete interval $[n_{\min}, n_{\max}]$. An algorithm with optimal Big-$O$ bounds may show higher absolute runtime on small inputs due to constant factor overheads (e.g. table initialization).
4. **Branch Prediction & Memory Alignment**: Repetitive inputs benefit heavily from CPU branch predictor caching, causing search loops to execute faster per comparison than on high-entropy random inputs.
