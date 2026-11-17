"""Synthetic and empirical input generators for string matching experiments.

Provides configurable, reproducible dataset generation including:
- Random text and patterns with configurable alphabet sizes.
- Repetitive and periodic sequences.
- Best-case inputs for sublinear matching.
- Adversarial repetitive-prefix inputs.
- Patterns guaranteed absent from text.
- Controlled match positioning (start, middle, end, overlapping).
"""

from dataclasses import dataclass
from enum import Enum
import random
import string
from typing import List, Optional, Tuple


class Alphabet:
    """Predefined alphabet constants for algorithmic benchmarks."""

    BINARY = "01"
    DNA = "ACGT"
    LOWERCASE = string.ascii_lowercase
    UPPERCASE = string.ascii_uppercase
    ALPHANUMERIC = string.ascii_letters + string.digits
    PRINTABLE = string.printable.strip()


class InputDistribution(str, Enum):
    """Categorical input distributions for empirical evaluation."""

    RANDOM = "random"
    RANDOM_WITH_MATCH = "random_with_match"
    REPETITIVE = "repetitive"
    PERIODIC = "periodic"
    BEST_CASE = "best_case"
    ADVERSARIAL_PREFIX = "adversarial_prefix"
    ABSENT = "absent"
    MATCH_AT_START = "match_at_start"
    MATCH_AT_MIDDLE = "match_at_middle"
    MATCH_AT_END = "match_at_end"
    OVERLAPPING = "overlapping"


def get_alphabet_by_size(size: int) -> str:
    """Deterministically construct an alphabet containing exactly `size` unique characters.

    Args:
        size: Desired alphabet cardinality (|Σ|). Must be >= 1.

    Returns:
        String of distinct characters.
    """
    if size <= 0:
        raise ValueError(f"Alphabet size must be positive, got {size}")

    if size == 2:
        return Alphabet.BINARY
    if size == 4:
        return Alphabet.DNA
    if size <= 26:
        return string.ascii_lowercase[:size]
    if size <= 52:
        return (string.ascii_lowercase + string.ascii_uppercase)[:size]
    if size <= 62:
        return (string.ascii_letters + string.digits)[:size]

    # For larger alphabets, build deterministic unicode character pool
    chars = [chr(c) for c in range(33, 127) if chr(c) not in string.whitespace]
    if size <= len(chars):
        return "".join(chars[:size])

    # Extend with Latin-1 supplement if needed
    for code in range(161, 255):
        if len(chars) >= size:
            break
        chars.append(chr(code))

    if len(chars) < size:
        raise ValueError(f"Requested alphabet size {size} exceeds supported pool ({len(chars)})")

    return "".join(chars[:size])


@dataclass(frozen=True)
class GeneratedInput:
    """Container holding a reproducible input pair and its generative provenance."""

    text: str
    pattern: str
    distribution: InputDistribution
    text_length: int
    pattern_length: int
    alphabet_size: int
    seed: Optional[int]


def generate_random_string(
    length: int,
    alphabet: str = Alphabet.LOWERCASE,
    seed: Optional[int] = None,
) -> str:
    """Generate a uniform random string of specified length and alphabet."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if not alphabet:
        raise ValueError("alphabet cannot be empty")

    rng = random.Random(seed)
    return "".join(rng.choice(alphabet) for _ in range(length))


def generate_dna_sequence(length: int, seed: Optional[int] = None) -> str:
    """Generate a random DNA nucleotide sequence (A, C, G, T)."""
    return generate_random_string(length, Alphabet.DNA, seed)


def generate_binary_sequence(length: int, seed: Optional[int] = None) -> str:
    """Generate a random binary string (0 and 1)."""
    return generate_random_string(length, Alphabet.BINARY, seed)


def generate_repetitive_string(char: str, length: int) -> str:
    """Generate a string consisting of a single character repeated `length` times."""
    if len(char) != 1:
        raise ValueError("char must be a single character")
    if length < 0:
        raise ValueError("length must be non-negative")
    return char * length


def generate_periodic_string(period: str, total_length: int) -> str:
    """Generate a periodic string repeating `period` up to `total_length`."""
    if not period:
        raise ValueError("period cannot be empty")
    if total_length < 0:
        raise ValueError("total_length must be non-negative")
    repeats = (total_length // len(period)) + 1
    return (period * repeats)[:total_length]


def generate_worst_case_naive(
    text_length: int, pattern_length: int
) -> Tuple[str, str]:
    """Generate adversarial input triggering O(n * m) comparisons in Naïve search.

    Pattern: 'A' * (m - 1) + 'B'
    Text:    'A' * n
    Forces Naïve to compare all m characters at each shift before failing on the last.
    """
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    pattern = ("A" * (pattern_length - 1)) + "B"
    text = "A" * text_length
    return text, pattern


def generate_best_case_boyer_moore(
    text_length: int, pattern_length: int
) -> Tuple[str, str]:
    """Generate best-case input for Boyer-Moore yielding O(n / m) search time.

    Pattern: 'B' * m
    Text:    'A' * n
    Causes immediate mismatch on text character 'A' (not in pattern), shifting by m.
    """
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    pattern = "B" * pattern_length
    text = "A" * text_length
    return text, pattern


def generate_overlapping_case(
    text_length: int, pattern_length: int, char: str = "A"
) -> Tuple[str, str]:
    """Generate text and pattern consisting of identical characters."""
    if pattern_length <= 0 or text_length <= 0:
        raise ValueError("Lengths must be positive")
    if pattern_length > text_length:
        raise ValueError("pattern_length cannot exceed text_length")

    text = char * text_length
    pattern = char * pattern_length
    return text, pattern


def generate_embedded_matches(
    text_length: int,
    pattern: str,
    num_matches: int,
    seed: Optional[int] = None,
    alphabet: str = Alphabet.LOWERCASE,
) -> Tuple[str, str, List[int]]:
    """Generate random text and embed `pattern` at non-overlapping positions."""
    m = len(pattern)
    if m == 0:
        raise ValueError("pattern cannot be empty")
    if num_matches * m > text_length:
        raise ValueError("num_matches * pattern_length exceeds text_length")

    rng = random.Random(seed)
    available_slots = text_length - (num_matches * m)
    gaps = [rng.randint(0, available_slots) for _ in range(num_matches + 1)]
    scale = available_slots / max(1, sum(gaps))
    gap_lengths = [int(g * scale) for g in gaps]
    diff = available_slots - sum(gap_lengths)
    gap_lengths[0] += diff

    text_parts = []
    inserted_indices = []
    current_idx = 0

    for i in range(num_matches):
        bg = "".join(rng.choice(alphabet) for _ in range(gap_lengths[i]))
        text_parts.append(bg)
        current_idx += len(bg)
        inserted_indices.append(current_idx)
        text_parts.append(pattern)
        current_idx += m

    bg_tail = "".join(rng.choice(alphabet) for _ in range(gap_lengths[-1]))
    text_parts.append(bg_tail)
    text = "".join(text_parts)

    return text, pattern, inserted_indices


def generate_benchmark_input(
    distribution: InputDistribution,
    text_length: int,
    pattern_length: int,
    alphabet_size: int = 26,
    seed: Optional[int] = None,
) -> GeneratedInput:
    """Centralized, reproducible input generator supporting all benchmark distributions.

    Args:
        distribution: Desired input distribution type.
        text_length: Number of characters in the corpus text (n).
        pattern_length: Number of characters in the search pattern (m).
        alphabet_size: Alphabet cardinality (|Σ|).
        seed: Deterministic random seed for reproducibility.

    Returns:
        GeneratedInput holding text, pattern, and metadata.
    """
    if text_length < 0 or pattern_length < 0:
        raise ValueError("Lengths must be non-negative")

    rng = random.Random(seed)
    alphabet = get_alphabet_by_size(alphabet_size)

    if distribution == InputDistribution.RANDOM:
        text = "".join(rng.choice(alphabet) for _ in range(text_length))
        pat_seed = rng.randint(0, 1_000_000)
        rng_pat = random.Random(pat_seed)
        pattern = "".join(rng_pat.choice(alphabet) for _ in range(pattern_length))

    elif distribution == InputDistribution.RANDOM_WITH_MATCH:
        text = "".join(rng.choice(alphabet) for _ in range(text_length))
        if pattern_length == 0:
            pattern = ""
        elif pattern_length <= text_length:
            start = rng.randint(0, text_length - pattern_length)
            pattern = text[start : start + pattern_length]
        else:
            pattern = "".join(rng.choice(alphabet) for _ in range(pattern_length))

    elif distribution == InputDistribution.REPETITIVE:
        char = alphabet[0]
        text = char * text_length
        pattern = char * pattern_length

    elif distribution == InputDistribution.PERIODIC:
        if alphabet_size >= 2:
            period = alphabet[0] + alphabet[1]
        else:
            period = alphabet[0]
        text = generate_periodic_string(period, text_length)
        pattern = generate_periodic_string(period, pattern_length)

    elif distribution == InputDistribution.BEST_CASE:
        # Boyer-Moore best case: pattern chars do not appear in text
        if alphabet_size >= 2:
            text_char = alphabet[0]
            pat_char = alphabet[1]
        else:
            text_char = "A"
            pat_char = "B"
        text = text_char * text_length
        pattern = pat_char * pattern_length

    elif distribution == InputDistribution.ADVERSARIAL_PREFIX:
        # Naïve worst case: pattern matches prefix of A's, then fails on last char
        if alphabet_size >= 2:
            a_char = alphabet[0]
            b_char = alphabet[1]
        else:
            a_char = "A"
            b_char = "B"
        text = a_char * text_length
        if pattern_length <= 1:
            pattern = b_char * pattern_length
        else:
            pattern = (a_char * (pattern_length - 1)) + b_char

    elif distribution == InputDistribution.ABSENT:
        # Pattern characters guaranteed disjoint from text
        if alphabet_size >= 2:
            half = max(1, alphabet_size // 2)
            text_alpha = alphabet[:half]
            pat_alpha = alphabet[half:]
            text = "".join(rng.choice(text_alpha) for _ in range(text_length))
            pattern = "".join(rng.choice(pat_alpha) for _ in range(pattern_length))
        else:
            text = "A" * text_length
            pattern = "B" * pattern_length

    elif distribution == InputDistribution.MATCH_AT_START:
        # Place pattern at index 0; rest has disjoint characters to prevent other matches
        if alphabet_size >= 2:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            tail = "".join(rng.choice(text_alpha) for _ in range(max(0, text_length - pattern_length)))
            text = (pattern + tail)[:text_length]
        else:
            pattern = "A" * pattern_length
            text = "A" * text_length

    elif distribution == InputDistribution.MATCH_AT_MIDDLE:
        # Place pattern at index n // 2; remainder disjoint
        if alphabet_size >= 2 and text_length >= pattern_length:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            mid_start = (text_length - pattern_length) // 2
            prefix = "".join(rng.choice(text_alpha) for _ in range(mid_start))
            suffix_len = text_length - mid_start - pattern_length
            suffix = "".join(rng.choice(text_alpha) for _ in range(max(0, suffix_len)))
            text = prefix + pattern + suffix
        else:
            text = "A" * text_length
            pattern = "A" * pattern_length

    elif distribution == InputDistribution.MATCH_AT_END:
        # Place pattern at index n - m; prefix disjoint
        if alphabet_size >= 2 and text_length >= pattern_length:
            text_alpha = alphabet[1:]
            pat_char = alphabet[0]
            pattern = pat_char * pattern_length
            prefix_len = text_length - pattern_length
            prefix = "".join(rng.choice(text_alpha) for _ in range(prefix_len))
            text = prefix + pattern
        else:
            text = "A" * text_length
            pattern = "A" * pattern_length

    elif distribution == InputDistribution.OVERLAPPING:
        char = alphabet[0]
        text = char * text_length
        pattern = char * pattern_length

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return GeneratedInput(
        text=text,
        pattern=pattern,
        distribution=distribution,
        text_length=len(text),
        pattern_length=len(pattern),
        alphabet_size=alphabet_size,
        seed=seed,
    )
