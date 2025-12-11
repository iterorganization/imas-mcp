"""Fuzzy matching for IMAS path suggestions.

Provides typo correction and path suggestions for invalid or not-found paths.
"""

import logging
from difflib import get_close_matches

logger = logging.getLogger(__name__)


class PathFuzzyMatcher:
    """Provides fuzzy matching suggestions for invalid paths.

    Uses difflib.get_close_matches for efficient similarity matching.
    Builds an internal index for faster IDS-scoped lookups.
    """

    def __init__(self, valid_ids_names: list[str], valid_paths: list[str]):
        """Initialize the fuzzy matcher with valid IDS names and paths.

        Args:
            valid_ids_names: List of valid IDS names (e.g., ["equilibrium", "magnetics"]).
            valid_paths: List of valid full paths (e.g., ["equilibrium/time_slice/..."]).
        """
        self.valid_ids_names = valid_ids_names
        self.valid_paths = valid_paths
        self._path_by_ids: dict[str, list[str]] = {}
        self._build_path_index()

    def _build_path_index(self) -> None:
        """Build index of paths by IDS name for faster lookup."""
        for path in self.valid_paths:
            ids_name = path.split("/")[0] if "/" in path else path
            if ids_name not in self._path_by_ids:
                self._path_by_ids[ids_name] = []
            self._path_by_ids[ids_name].append(path)

    def suggest_ids(self, invalid_ids: str, max_suggestions: int = 3) -> list[str]:
        """Suggest valid IDS names for a typo.

        Args:
            invalid_ids: The invalid IDS name to find matches for.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of similar valid IDS names, sorted by similarity.
        """
        if not invalid_ids:
            return []

        # Use case-insensitive matching
        matches = get_close_matches(
            invalid_ids.lower(),
            [ids.lower() for ids in self.valid_ids_names],
            n=max_suggestions,
            cutoff=0.6,
        )

        # Return original case versions
        ids_lower_map = {ids.lower(): ids for ids in self.valid_ids_names}
        return [ids_lower_map[m] for m in matches if m in ids_lower_map]

    def suggest_paths(self, invalid_path: str, max_suggestions: int = 3) -> list[str]:
        """Suggest valid paths for a typo or partial path.

        Args:
            invalid_path: The invalid path to find matches for.
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of similar valid paths, sorted by similarity.
        """
        if not invalid_path:
            return []

        # Extract IDS name from path
        parts = invalid_path.split("/")
        ids_name = parts[0]

        # First check if IDS name is valid
        if ids_name not in self.valid_ids_names:
            # Suggest similar IDS names and try to construct corrected path
            similar_ids = self.suggest_ids(ids_name, max_suggestions=1)
            if similar_ids:
                corrected_ids = similar_ids[0]
                if corrected_ids in self._path_by_ids:
                    # Try to find paths with the corrected IDS
                    corrected_path = "/".join([corrected_ids] + parts[1:])
                    candidates = self._path_by_ids[corrected_ids]
                    return get_close_matches(
                        corrected_path, candidates, n=max_suggestions, cutoff=0.5
                    )
            return []

        # IDS is valid, search within that IDS for better performance
        candidates = self._path_by_ids.get(ids_name, self.valid_paths)
        return get_close_matches(
            invalid_path, candidates, n=max_suggestions, cutoff=0.5
        )

    def get_suggestion_for_path(self, path: str) -> str | None:
        """Get a single best suggestion for a path, formatted as hint.

        Args:
            path: The invalid path to get a suggestion for.

        Returns:
            Formatted suggestion string like "Did you mean: X?", or None.
        """
        suggestions = self.suggest_paths(path, max_suggestions=1)
        if suggestions:
            return f"Did you mean: {suggestions[0]}?"

        # Try IDS-level suggestion
        ids_name = path.split("/")[0] if "/" in path else path
        ids_suggestions = self.suggest_ids(ids_name, max_suggestions=1)
        if ids_suggestions:
            return f"Unknown IDS '{ids_name}'. Did you mean: {ids_suggestions[0]}?"

        return None


# Singleton instance - initialized lazily
_matcher: PathFuzzyMatcher | None = None


def get_fuzzy_matcher(
    valid_ids_names: list[str], valid_paths: list[str]
) -> PathFuzzyMatcher:
    """Get or create the fuzzy matcher singleton.

    Args:
        valid_ids_names: List of valid IDS names.
        valid_paths: List of valid full paths.

    Returns:
        PathFuzzyMatcher instance.
    """
    global _matcher
    if _matcher is None:
        _matcher = PathFuzzyMatcher(valid_ids_names, valid_paths)
    return _matcher


def reset_fuzzy_matcher() -> None:
    """Reset the fuzzy matcher singleton (useful for testing)."""
    global _matcher
    _matcher = None


def suggest_correction(
    path: str, valid_ids_names: list[str], valid_paths: list[str]
) -> str | None:
    """Convenience function to get a suggestion for a path.

    Args:
        path: The invalid path to get a suggestion for.
        valid_ids_names: List of valid IDS names.
        valid_paths: List of valid full paths.

    Returns:
        Formatted suggestion string, or None if no suggestion.
    """
    matcher = get_fuzzy_matcher(valid_ids_names, valid_paths)
    return matcher.get_suggestion_for_path(path)
