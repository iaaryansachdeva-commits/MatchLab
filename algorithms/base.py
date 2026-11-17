"""Base definitions, contracts, and data structures for string matching algorithms.

This module defines the standard `MatchResult` data class and the `BaseMatcher`
abstract base class that all matching algorithms implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any, Dict, List


@dataclass(frozen=True)
class MatchResult:
    """Immutable result structure produced by string pattern matching algorithms.

    Attributes:
        algorithm_name: Identifier name of the algorithm (e.g. 'Naive', 'KMP', 'Boyer-Moore').
        pattern: The target pattern queried.
        text: The source text searched.
        matches: List of 0-based starting indices where the pattern occurs in the text.
        preprocessing_comparisons: Number of character comparisons made during pattern preprocessing.
        search_comparisons: Number of character comparisons made during the search phase.
        total_comparisons: Sum of preprocessing and search comparisons.
        execution_time_seconds: Elapsed search time in seconds (high precision).
        metadata: Optional dictionary with algorithm-specific statistics (e.g. LPS array, shifts).
    """

    algorithm_name: str
    pattern: str
    text: str
    matches: List[int]
    preprocessing_comparisons: int
    search_comparisons: int
    total_comparisons: int
    execution_time_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.total_comparisons != (self.preprocessing_comparisons + self.search_comparisons):
            # Enforce total consistency if not aligned
            object.__setattr__(
                self,
                "total_comparisons",
                self.preprocessing_comparisons + self.search_comparisons,
            )


class BaseMatcher(ABC):
    """Abstract base class establishing the contract for all string matching algorithms.

    Each algorithm implementation must:
    1. Be stateless across distinct search calls (avoiding global mutable state).
    2. Track character comparisons during preprocessing and searching separately.
    3. Return all matching starting indices (including overlapping occurrences).
    4. Support empty patterns, empty texts, and patterns longer than text.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm identifier."""
        pass

    @abstractmethod
    def search(self, text: str, pattern: str) -> MatchResult:
        """Find all starting indices of `pattern` in `text`.

        Args:
            text: The corpus string to search within.
            pattern: The pattern substring to look for.

        Returns:
            A `MatchResult` object containing match indices, comparison counts,
            timing, and algorithm metadata.
        """
        pass

    def _validate_inputs(self, text: str, pattern: str) -> None:
        """Validate input types."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}")
        if not isinstance(pattern, str):
            raise TypeError(f"pattern must be a string, got {type(pattern).__name__}")

    def _handle_empty_pattern(self, text: str, start_time: float) -> MatchResult:
        """Standardized handling for empty patterns (epsilon string).

        In standard string semantics, an empty pattern matches at every position
        from index 0 to len(text) inclusive, with 0 comparisons.
        """
        elapsed = time.perf_counter() - start_time
        return MatchResult(
            algorithm_name=self.name,
            pattern="",
            text=text,
            matches=list(range(len(text) + 1)),
            preprocessing_comparisons=0,
            search_comparisons=0,
            total_comparisons=0,
            execution_time_seconds=elapsed,
            metadata={"note": "empty_pattern"},
        )

    def _handle_pattern_longer_than_text(
        self, text: str, pattern: str, start_time: float, preprocessing_comparisons: int = 0
    ) -> MatchResult:
        """Standardized handling when pattern length exceeds text length."""
        elapsed = time.perf_counter() - start_time
        return MatchResult(
            algorithm_name=self.name,
            pattern=pattern,
            text=text,
            matches=[],
            preprocessing_comparisons=preprocessing_comparisons,
            search_comparisons=0,
            total_comparisons=preprocessing_comparisons,
            execution_time_seconds=elapsed,
            metadata={"note": "pattern_longer_than_text"},
        )
