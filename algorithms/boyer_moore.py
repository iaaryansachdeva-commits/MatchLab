"""Boyer-Moore String Matching Algorithm.

This module implements the full classical Boyer-Moore string matching algorithm
featuring:
1. Extended Bad Character Heuristic (rightmost occurrence lookup).
2. Strong Good Suffix Heuristic (Gusfield's N-array / fundamental Z-algorithm preprocessing).
3. Right-to-left character comparison during search.
4. Independent tracking of preprocessing and searching character comparisons.
"""

import time
from typing import Dict, List, Tuple

from algorithms.base import BaseMatcher, MatchResult


class BoyerMooreMatcher(BaseMatcher):
    """Boyer-Moore String Matching Algorithm.

    Algorithmic Variant:
        Classical Boyer-Moore utilizing:
        - Bad Character Rule: Precomputed dictionary mapping each alphabet character
          to its rightmost occurrence in the pattern. On mismatch at pattern[j] with
          text character c, the suggested shift is max(1, j - bad_char.get(c, -1)).
        - Strong Good Suffix Rule: Precomputed using Gusfield's N-array (derived from
          the fundamental Z-algorithm on the reversed pattern string). Distinguishes:
            * Case 1: Rightmost occurrence of matched suffix preceded by a character
              distinct from pattern[j].
            * Case 2: Longest proper prefix of pattern that matches a suffix of the
              matched suffix.
        - Right-to-Left Scanning: At each shift s, pattern characters are compared
          against text from index (m - 1) down to 0.
        - Overlapping Matches: Handled upon full match by shifting by
          (m - longest_proper_border), ensuring complete recall of overlapping occurrences.

    Complexity Characteristics:
        - Preprocessing Time: O(m + |Σ|) where m is pattern length and Σ is pattern alphabet.
        - Preprocessing Space: O(m + |Σ|) for Bad Character table and Good Suffix table.
        - Preprocessing Comparisons: Derived strictly from character comparisons during
          the fundamental Z-algorithm execution on the reversed pattern.
        - Search Time:
            * Best Case: O(n / m) sublinear time when text characters do not occur
              in pattern, advancing by m on each iteration.
            * Average Case: O(n / m) on large natural alphabets.
            * Worst Case: O(n * m) without Galil/Apostolico match memory on highly
              periodic strings; O(n + m) when pattern does not appear in text.
    """

    @property
    def name(self) -> str:
        return "Boyer-Moore"

    def compute_bad_character_table(self, pattern: str) -> Dict[str, int]:
        """Compute the Bad Character table (rightmost occurrence mapping).

        Args:
            pattern: The pattern string.

        Returns:
            Dictionary mapping character -> rightmost 0-based index in pattern.
        """
        table: Dict[str, int] = {}
        for index, char in enumerate(pattern):
            table[char] = index
        return table

    def compute_z_array(self, string: str) -> Tuple[List[int], int]:
        """Compute the Z-array for a string while tracking character comparisons.

        Z[i] is the length of the longest substring starting from string[i] that
        is also a prefix of string.

        Args:
            string: Target string.

        Returns:
            Tuple of (z_array, comparison_count).
        """
        n = len(string)
        if n == 0:
            return [], 0

        z = [0] * n
        z[0] = n
        left = 0
        right = 0
        comparisons = 0

        for i in range(1, n):
            if i <= right:
                k = i - left
                remaining = right - i + 1
                if z[k] < remaining:
                    z[i] = z[k]
                    continue
                else:
                    # Need to compare beyond right boundary
                    match_len = remaining
            else:
                match_len = 0

            while (i + match_len < n) and (match_len < n):
                comparisons += 1
                if string[i + match_len] == string[match_len]:
                    match_len += 1
                else:
                    break

            z[i] = match_len
            if i + match_len - 1 > right:
                left = i
                right = i + match_len - 1

        return z, comparisons

    def compute_good_suffix_table(
        self, pattern: str
    ) -> Tuple[List[int], int, int]:
        """Compute the Good Suffix shift table and full match shift.

        Uses Gusfield's N-array derived from the reversed pattern's Z-array.

        Args:
            pattern: The pattern string.

        Returns:
            Tuple of (good_suffix_table, full_match_shift, preprocessing_comparisons).
        """
        m = len(pattern)
        if m == 0:
            return [], 1, 0

        # Reversed pattern for suffix preprocessing
        reversed_pattern = pattern[::-1]
        z_array, prep_comparisons = self.compute_z_array(reversed_pattern)

        # N[j] is the length of the longest suffix of pattern[0..j] that is also a suffix of pattern.
        # Index j in pattern corresponds to index (m - 1 - j) in reversed_pattern.
        n_array = [0] * m
        for j in range(m - 1):
            n_array[j] = z_array[m - 1 - j]
        n_array[m - 1] = m

        # Step 1: Precompute longest proper prefix that is also a suffix (borders)
        # Prefix of length L is a suffix of pattern iff N[L - 1] == L.
        max_prefix_len = [0] * (m + 1)
        longest_proper_border = 0

        for l in range(1, m):
            if n_array[l - 1] == l:
                max_prefix_len[l] = l
                longest_proper_border = l

        # Propagate longest prefix match backward for smaller suffix lengths
        for length in range(1, m + 1):
            if max_prefix_len[length] == 0:
                max_prefix_len[length] = max_prefix_len[length - 1]

        # Step 2: Initialize good suffix shifts with Case 2 (Prefix/Border)
        # For mismatch at index j, matched suffix length is k = m - 1 - j.
        # If k == 0 (mismatch at last char), no suffix matched, fallback shift is 1.
        good_suffix = [1] * m
        for j in range(m - 1):
            k = m - 1 - j
            prefix_len = max_prefix_len[k]
            good_suffix[j] = m - prefix_len

        # Step 3: Apply Case 1 (Strong Good Suffix)
        # Find occurrences of the matched suffix preceded by a character distinct from pattern[j].
        # For j in 0..m-2, if N[j] > 0, substring of length k = N[j] ending at j matches pattern suffix.
        # By maximality of N[j], pattern[j - k] != pattern[m - 1 - k].
        for j in range(m - 1):
            k = n_array[j]
            if k > 0 and k < m:
                mismatch_pos = m - 1 - k
                # Overwriting earlier j with larger j gives rightmost occurrence (minimal shift)
                good_suffix[mismatch_pos] = (m - 1) - j

        # Full match shift: when all m characters matched, shift by (m - longest_proper_border)
        full_match_shift = m - longest_proper_border
        if full_match_shift <= 0:
            full_match_shift = 1

        return good_suffix, full_match_shift, prep_comparisons

    def search(self, text: str, pattern: str) -> MatchResult:
        """Search for all occurrences of pattern in text using Boyer-Moore.

        Args:
            text: Corpus string.
            pattern: Target pattern.

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

        # Preprocessing phase
        bad_char = self.compute_bad_character_table(pattern)
        good_suffix, full_match_shift, prep_comparisons = self.compute_good_suffix_table(pattern)

        # Handle pattern longer than text
        if m > n:
            return self._handle_pattern_longer_than_text(
                text, pattern, start_time, prep_comparisons
            )

        matches: List[int] = []
        search_comparisons = 0
        s = 0  # Shift of the pattern with respect to text

        # Search phase: slide pattern over text
        while s <= n - m:
            j = m - 1

            # Right-to-left scan
            while j >= 0:
                search_comparisons += 1
                if pattern[j] != text[s + j]:
                    break
                j -= 1

            if j < 0:
                # Full match found at shift s
                matches.append(s)
                s += full_match_shift
            else:
                # Mismatch at pattern[j] against text[s + j]
                mismatched_char = text[s + j]
                last_char_pos = bad_char.get(mismatched_char, -1)
                bc_shift = max(1, j - last_char_pos)
                gs_shift = good_suffix[j]
                s += max(bc_shift, gs_shift)

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
            metadata={
                "bad_char_table_size": len(bad_char),
                "good_suffix_table": good_suffix,
                "full_match_shift": full_match_shift,
            },
        )


def boyer_moore_search(text: str, pattern: str) -> MatchResult:
    """Convenience functional interface for Boyer-Moore string matching."""
    return BoyerMooreMatcher().search(text, pattern)
