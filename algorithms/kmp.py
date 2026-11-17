"""Knuth-Morris-Pratt (KMP) String Matching Algorithm.

This module implements the KMP algorithm with separate tracking of:
1. Preprocessing comparisons (LPS / π table construction).
2. Searching comparisons against the corpus text.
"""

import time
from typing import List, Tuple

from algorithms.base import BaseMatcher, MatchResult


class KMPMatcher(BaseMatcher):
    """Knuth-Morris-Pratt (KMP) String Matching Algorithm.

    Complexity Characteristics:
        - Preprocessing Time: O(m) where m is pattern length.
        - Preprocessing Comparisons: <= 2 * m character comparisons.
        - Space Complexity: O(m) auxiliary space for the LPS (Longest Proper Prefix
          which is also a Suffix) array.
        - Search Time: O(n) where n is text length, making worst-case search linear
          independent of alphabet size or repetition.
        - Search Comparisons: <= 2 * n character comparisons.
        - Total Time: O(n + m).
        - Overlapping Matches: Supported by transitioning to j = lps[j - 1] after a match.
    """

    @property
    def name(self) -> str:
        return "KMP"

    def compute_lps_table(self, pattern: str) -> Tuple[List[int], int]:
        """Compute the Longest Proper Prefix which is also a Suffix (LPS) array.

        Args:
            pattern: The pattern string.

        Returns:
            A tuple of (lps_array, preprocessing_comparisons).
        """
        m = len(pattern)
        if m == 0:
            return [], 0

        lps = [0] * m
        length = 0  # Length of the previous longest prefix suffix
        i = 1
        preprocessing_comparisons = 0

        while i < m:
            preprocessing_comparisons += 1
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            elif length != 0:
                # Fall back to shorter prefix suffix without character comparison
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

        return lps, preprocessing_comparisons

    def preprocess(self, pattern: str) -> Tuple[List[int], int]:
        """Precompute the LPS table and count preprocessing comparisons."""
        return self.compute_lps_table(pattern)

    def search_with_preprocessed(
        self, text: str, pattern: str, preprocessed: Tuple[List[int], int]
    ) -> MatchResult:
        """Execute KMP search using a precomputed LPS table."""
        self._validate_inputs(text, pattern)
        start_time = time.perf_counter()

        n = len(text)
        m = len(pattern)
        lps, prep_comparisons = preprocessed

        if m == 0:
            return self._handle_empty_pattern(text, start_time)

        if m > n:
            return self._handle_pattern_longer_than_text(
                text, pattern, start_time, prep_comparisons
            )

        matches: List[int] = []
        search_comparisons = 0
        i = 0  # Text index
        j = 0  # Pattern index

        while i < n:
            search_comparisons += 1
            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == m:
                    matches.append(i - j)
                    j = lps[j - 1]
            elif j != 0:
                j = lps[j - 1]
            else:
                i += 1

        elapsed = time.perf_counter() - start_time

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=matches,
            preprocessing_comparisons=prep_comparisons,
            search_comparisons=search_comparisons,
            total_comparisons=prep_comparisons + search_comparisons,
            execution_time_seconds=elapsed,
            preprocessing_time_seconds=0.0,
            search_time_seconds=elapsed,
            metadata={
                "lps_table": lps,
            },
        )

    def search(self, text: str, pattern: str) -> MatchResult:
        """Search for all occurrences of pattern in text using KMP.

        Args:
            text: Corpus string.
            pattern: Target pattern.

        Returns:
            MatchResult containing starting indices, comparisons, and timing.
        """
        self._validate_inputs(text, pattern)
        t_prep_start = time.perf_counter()
        preprocessed = self.preprocess(pattern)
        prep_elapsed = time.perf_counter() - t_prep_start

        t_search_start = time.perf_counter()
        result = self.search_with_preprocessed(text, pattern, preprocessed)
        search_elapsed = time.perf_counter() - t_search_start

        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=result.matches,
            preprocessing_comparisons=result.preprocessing_comparisons,
            search_comparisons=result.search_comparisons,
            total_comparisons=result.total_comparisons,
            execution_time_seconds=prep_elapsed + search_elapsed,
            preprocessing_time_seconds=prep_elapsed,
            search_time_seconds=search_elapsed,
            metadata=result.metadata,
        )


def kmp_search(text: str, pattern: str) -> MatchResult:
    """Convenience functional interface for KMP string matching."""
    return KMPMatcher().search(text, pattern)
