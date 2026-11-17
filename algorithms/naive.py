"""Naïve (Brute-Force) String Matching Algorithm.

This module implements the straightforward character-by-character sliding-window
pattern matching algorithm with strict character comparison tracking.
"""

import time
from typing import List

from algorithms.base import BaseMatcher, MatchResult


class NaiveStringMatcher(BaseMatcher):
    """Naïve (Brute-Force) String Matching Algorithm.

    Complexity Characteristics:
        - Preprocessing Time: O(1) (No preprocessing performed).
        - Preprocessing Comparisons: 0.
        - Space Complexity: O(1) auxiliary space.
        - Best-Case Search Time: O(n) comparisons when the first character
          of the pattern mismatches on every slide (e.g., text='AAAAA', pattern='BA').
        - Worst-Case Search Time: O((n - m + 1) * m) comparisons when nearly all
          characters match before a terminal mismatch occurs (e.g., text='AAAAA', pattern='AAAAB').
        - Overlapping Matches: Supported by advancing shift by 1 after each match.
    """

    @property
    def name(self) -> str:
        return "Naive"

    def search(self, text: str, pattern: str) -> MatchResult:
        """Search for all occurrences of pattern in text using naïve comparison.

        Args:
            text: Corpus string.
            pattern: Target pattern to locate.

        Returns:
            MatchResult containing starting indices, comparisons, and timing.
        """
        self._validate_inputs(text, pattern)
        start_time = time.perf_counter()

        n = len(text)
        m = len(pattern)

        # Handle empty pattern
        if m == 0:
            return self._handle_empty_pattern(text, start_time)

        # Handle pattern longer than text
        if m > n:
            return self._handle_pattern_longer_than_text(text, pattern, start_time, 0)

        matches: List[int] = []
        search_comparisons = 0

        # Slide pattern window across text
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                search_comparisons += 1
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                matches.append(i)

        elapsed = time.perf_counter() - start_time

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=matches,
            preprocessing_comparisons=0,
            search_comparisons=search_comparisons,
            total_comparisons=search_comparisons,
            execution_time_seconds=elapsed,
            metadata={
                "shifts_evaluated": n - m + 1,
            },
        )


def naive_search(text: str, pattern: str) -> MatchResult:
    """Convenience functional interface for Naïve string matching."""
    return NaiveStringMatcher().search(text, pattern)
