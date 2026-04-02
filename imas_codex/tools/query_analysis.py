"""Query analysis for IMAS DD search.

Classifies user queries and extracts structural hints to improve search
routing. Detects path-like queries, physics abbreviations, and accessor
terminal patterns.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Accessor terminals — child nodes that are metadata/data containers
# rather than physics concepts. Excluded from search indexes but must
# be surfaced when explicitly queried.
ACCESSOR_TERMINALS: frozenset[str] = frozenset(
    {
        "data",
        "value",
        "time",
        "r",
        "z",
        "phi",
        "coefficients",
        "label",
        "grid_index",
        "measured",
        "reconstructed",
        "parallel",
        "toroidal",
        "perpendicular",
    }
)

# Physics abbreviations commonly used in fusion research
PHYSICS_ABBREVIATIONS: dict[str, list[str]] = {
    "ip": ["plasma current", "ip"],
    "te": ["electron temperature", "te"],
    "ti": ["ion temperature", "ti"],
    "ne": ["electron density", "ne"],
    "ni": ["ion density", "ni"],
    "bt": ["toroidal magnetic field", "b_field_tor", "bt", "b0"],
    "bp": ["poloidal magnetic field", "b_field_pol", "bp"],
    "q": ["safety factor", "q"],
    "psi": ["poloidal flux", "psi"],
    "beta": ["plasma beta", "beta_pol", "beta_tor", "beta_normal", "beta"],
    "li": ["internal inductance", "li"],
    "wmhd": ["stored energy", "w_mhd", "wmhd"],
    "zeff": ["effective charge", "z_eff", "zeff"],
    "vloop": ["loop voltage", "v_loop", "vloop"],
    "bpol": ["poloidal magnetic field", "b_field_pol", "bpol"],
    "btor": ["toroidal magnetic field", "b_field_tor", "btor"],
    "te0": ["central electron temperature", "te", "te0"],
    "ti0": ["central ion temperature", "ti", "ti0"],
    "ne0": ["central electron density", "ne", "ne0"],
}


@dataclass
class QueryIntent:
    """Classified query with extracted hints."""

    query_type: str  # "path_exact", "path_partial", "concept", "hybrid"
    original_query: str
    accessor_hint: str | None = None  # e.g., "data", "value"
    expanded_terms: list[str] = field(default_factory=list)
    path_segments: list[str] = field(default_factory=list)
    stripped_query: str | None = None  # Query with accessor terminal removed
    ids_hint: str | None = None  # Detected IDS name from path
    is_abbreviation: bool = False


class QueryAnalyzer:
    """Analyzes search queries and extracts structural hints."""

    def analyze(self, query: str) -> QueryIntent:
        """Classify query and extract accessor/path/abbreviation hints."""
        query = query.strip()
        if not query:
            return QueryIntent(query_type="concept", original_query=query)

        # Check if query looks like a path (contains "/")
        if "/" in query and " " not in query:
            return self._analyze_path_query(query)

        # Check for physics abbreviations (single short word)
        words = query.lower().split()
        if len(words) == 1 and words[0] in PHYSICS_ABBREVIATIONS:
            return self._analyze_abbreviation(query, words[0])

        # Check for hybrid: concept + IDS hint (e.g., "electron temperature core_profiles")
        if len(words) >= 2:
            # Check if any word is a known abbreviation
            for w in words:
                if w in PHYSICS_ABBREVIATIONS:
                    return self._analyze_hybrid(query, w)

        # Pure concept query
        return QueryIntent(
            query_type="concept",
            original_query=query,
            expanded_terms=[query],
        )

    def _analyze_path_query(self, query: str) -> QueryIntent:
        """Analyze a path-like query (contains '/')."""
        segments = query.split("/")

        # Check if the last segment is an accessor terminal
        accessor_hint = None
        stripped = None
        if segments[-1].lower() in ACCESSOR_TERMINALS:
            accessor_hint = segments[-1].lower()
            stripped = "/".join(segments[:-1])

        # Detect IDS hint (first segment if it looks like an IDS name)
        ids_hint = segments[0] if not segments[0].startswith("_") else None

        # Determine if this is an exact path (has IDS prefix) or partial
        if len(segments) >= 2 and not any(c in query for c in " *?"):
            # Likely an exact or near-exact path
            query_type = "path_exact"
        else:
            query_type = "path_partial"

        return QueryIntent(
            query_type=query_type,
            original_query=query,
            accessor_hint=accessor_hint,
            path_segments=segments,
            stripped_query=stripped,
            ids_hint=ids_hint,
        )

    def _analyze_abbreviation(self, query: str, abbrev: str) -> QueryIntent:
        """Analyze a single-word abbreviation query."""
        expansions = PHYSICS_ABBREVIATIONS[abbrev]
        return QueryIntent(
            query_type="concept",
            original_query=query,
            expanded_terms=expansions,
            is_abbreviation=True,
        )

    def _analyze_hybrid(self, query: str, abbrev_word: str) -> QueryIntent:
        """Analyze a multi-word query containing an abbreviation."""
        expansions = PHYSICS_ABBREVIATIONS[abbrev_word]
        other_words = [w for w in query.lower().split() if w != abbrev_word]

        # Combine expanded terms with other query words
        all_terms = expansions + other_words + [query]

        return QueryIntent(
            query_type="hybrid",
            original_query=query,
            expanded_terms=all_terms,
            is_abbreviation=True,
        )


def strip_accessor_suffix(path: str) -> str:
    """Strip trailing accessor terminal from a path.

    e.g., "magnetics/flux_loop/flux/data" → "magnetics/flux_loop/flux"
    """
    segments = path.split("/")
    if segments and segments[-1].lower() in ACCESSOR_TERMINALS:
        return "/".join(segments[:-1])
    return path
