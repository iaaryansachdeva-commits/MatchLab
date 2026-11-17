"""Synthetic and empirical input generators for string matching experiments.

Provides configurable, reproducible dataset generation including:
- Uniform random text over customizable alphabets (Binary, DNA, English, ASCII).
- Repetitive and periodic sequences.
- Overlapping match benchmarks.
- Adversarial/worst-case patterns for Naïve, KMP, and Boyer-Moore.
"""

import random
import string
from typing import Optional, Tuple


class Alphabet:
    """Predefined alphabet constants for algorithmic benchmarks."""

    BINARY = "01"
    DNA = "ACGT"
    LOWERCASE = string.ascii_lowercase
    UPPERCASE = string.ascii_uppercase
    ALPHANUMERIC = string.ascii_letters + string.digits
    PRINTABLE = string.printable.strip()


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
    """Generate text and pattern consisting of identical characters.

    Produces dense overlapping occurrences at every index from 0 to n - m.
    """
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
) -> Tuple[str, str, list[int]]:
    """Generate random text and embed `pattern` at non-overlapping or chosen random positions.

    Returns:
        (text, pattern, inserted_indices)
    """
    m = len(pattern)
    if m == 0:
        raise ValueError("pattern cannot be empty")
    if num_matches * m > text_length:
        raise ValueError("num_matches * pattern_length exceeds text_length")

    # Generate random background text
    # Exclude characters in pattern to avoid accidental matches if desired,
    # or use normal alphabet
    rng = random.Random(seed)
    # Choose positions that do not collide
    available_slots = text_length - (num_matches * m)
    gaps = [rng.randint(0, available_slots) for _ in range(num_matches + 1)]
    scale = available_slots / max(1, sum(gaps))
    gap_lengths = [int(g * scale) for g in gaps]
    # Adjust difference
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
