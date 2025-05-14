# filepath: c:\Users\mcintos\Code\imas-mcp-server\imas_mcp_server\mcp_imas.py
# Standard library imports
import functools
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, ClassVar, Dict, Pattern

# Third-party imports
from fastmcp import FastMCP
import nest_asyncio
from pydantic import Field

# Local imports
from imas_mcp_server.path_index_cache import PathIndexCache

# apply nest_asyncio to allow nested event loops
# This is necessary for Jupyter notebooks and some other environments
# that don't support nested event loops by default.
nest_asyncio.apply()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
mcp = FastMCP("IMAS")


@dataclass
class DataDictionaryServer:
    """An IMAS MCP server class to serve IMAS Data Dictionary context."""

    mcp: FastMCP  # MCP server instance
    version: str | None = None  # IMAS version to use
    xml_path: Path | None = (
        None  # Path to the XML file    # Cache for compiled regex patterns to improve performance
    )
    _compiled_patterns: Dict[tuple[str, int], Pattern] = field(default_factory=dict)

    tools: ClassVar[list[str]] = [
        "list_ids",
        "find_paths_by_pattern",
        "get_path_documentation",
    ]

    def __post_init__(self):
        """Initialize MCP servers."""
        for tool in self.tools:
            self.mcp.add_tool(getattr(self, tool))

        # Initialize the patterns cache
        self._compiled_patterns = {}

    @functools.cached_property
    def path_index(self):
        """Return a DD path index instance."""
        return PathIndexCache(version=self.version, xml_path=self.xml_path).path_index

    def list_ids(self) -> set[str]:
        """Return a set of all the IDSs defined by the IMAS Data Dictionary."""
        return self.path_index.ids

    def _get_pattern_cache(self, pattern: str, flags: int = 0) -> Pattern:
        """Get a cached compiled regex pattern or compile and cache a new one."""
        key = (pattern, flags)
        if key not in self._compiled_patterns:
            self._compiled_patterns[key] = re.compile(pattern, flags)
        return self._compiled_patterns[key]

    @staticmethod
    def _extract_literal_substring(pattern: str) -> str:
        """Extract a literal substring from a regex pattern for pre-filtering."""
        # Simple heuristic to find a significant literal part in the regex
        # This won't handle all cases perfectly but helps in common scenarios
        parts = re.split(r"[.*+?()|\[\]{}^$]", pattern)
        parts = [p for p in parts if p and len(p) > 2]
        if parts:
            return max(parts, key=len)
        return ""

    def find_paths_by_pattern(
        self,
        pattern: Annotated[
            str,
            Field(
                description="A regular expression pattern to match against IDS paths",
                examples=[
                    ".*diagnostic.*",  # Find all paths containing 'diagnostic'
                    "equilibrium",  # Simple search for a specific IDS
                    "time_slice\\[\\]",  # Find array elements with escaped brackets
                    ".*profiles_1d/.*",  # Find all paths with 'profiles_1d' component
                    "equilibrium/time_slice.*/q_safety_factor",  # Specific property with wildcard time slice
                    "pulse/.*",  # All paths under the pulse IDS
                    "^core_profiles/.*electron.*",  # Paths starting with core_profiles and containing 'electron'
                    ".*/psi$",  # Match 'psi' as the final segment of any path
                ],
            ),
        ],
        use_regex: Annotated[
            bool,
            Field(
                default=True,
                description="Whether to interpret the pattern as regex or simple string",
            ),
        ] = True,
        case_sensitive: Annotated[
            bool,
            Field(
                default=True,
                description="Whether the search should be case-sensitive",
            ),
        ] = True,
    ) -> list[str]:
        """Find all IDS paths that match a given regex pattern using precomputed paths.

        Parameters:
        -----------
        pattern : str
            A regular expression pattern to match against IDS paths
        use_regex : bool, optional
            Whether to interpret the pattern as regex or simple string (default: True)
        case_sensitive : bool, optional
            Whether the search should be case-sensitive (default: True)

        Returns:
        --------
        list[str]
            A list of IDS paths that match the given pattern
        """

        try:
            # Handle non-regex search (faster)
            if not use_regex:
                if not case_sensitive:
                    pattern = pattern.lower()
                    matching_paths = [
                        p for p in self.path_index.paths if pattern in p.lower()
                    ]
                else:
                    matching_paths = [p for p in self.path_index.paths if pattern in p]
            else:
                # Handle regex search with optimizations
                try:
                    # Use cached regex pattern compilation for better performance
                    flags = 0 if case_sensitive else re.IGNORECASE
                    compiled_pattern = self._get_pattern_cache(pattern, flags)

                    # Attempt to optimize search by looking for common substrings first
                    common_substr = self._extract_literal_substring(pattern)
                    if common_substr and len(common_substr) > 2:
                        # Pre-filter paths using the literal substring for speed
                        candidate_paths = [
                            p for p in self.path_index.paths if common_substr in p
                        ]
                        matching_paths = [
                            p for p in candidate_paths if compiled_pattern.search(p)
                        ]
                    else:
                        # Fall back to full regex search on all paths
                        matching_paths = [
                            p
                            for p in self.path_index.paths
                            if compiled_pattern.search(p)
                        ]
                except re.error as e:
                    return [f"Error in regex pattern: {str(e)}"]

            return sorted(matching_paths)

        except Exception as e:
            return [f"Error searching paths: {str(e)}"]

    def get_path_documentation(
        self,
        path: Annotated[
            str,
            Field(
                description="The IDS path for which to retrieve documentation",
                examples=[
                    "/equilibrium/time_slice[]/profiles_1d/psi",  # Full path with array notation
                    "equilibrium",  # Root-level IDS
                    "core_profiles/electrons/density",  # Specific property path
                    "core_profiles/time_slice/ion/",  # Path ending with slash
                    "edge_profiles/ggd/*/electron_density",  # Path with wildcard character
                    "wall/description/limiter/unit/*/outline/",  # Complex nested path with wildcards
                ],
            ),
        ],
        search_method: Annotated[
            str,
            Field(
                default="smart",
                description="Method to use for finding the path documentation",
                examples=["exact", "prefix", "segments", "keywords", "regex", "smart"],
            ),
        ] = "smart",
        max_results: Annotated[
            int,
            Field(
                default=5,
                description="Maximum number of results to return for non-exact matches",
            ),
        ] = 5,
    ) -> str:
        """Return documentation for a given IDS path.

        Parameters:
        -----------
        path : str
            The IDS path for which to retrieve documentation
        search_method : str, optional
            Method to use for finding paths:
            - 'exact': Require exact match
            - 'prefix': Return paths that start with or are prefixes of the given path
            - 'segments': Match based on path segments/components
            - 'keywords': Match based on keywords extracted from the path
            - 'regex': Use regular expression pattern matching for flexible searches
            - 'smart': Try multiple methods automatically in succession (default)
        max_results : int, optional
            Maximum number of results to return for non-exact matches (default: 5)

        Returns:
        --------
        str
            Documentation string for the specified path, or error message if not found
        """
        try:
            # Clean up the path (remove leading/trailing whitespace and slashes)
            clean_path = path.strip().strip("/")

            # For exact match, simply look up in the docs dictionary
            if search_method == "exact":
                if clean_path in self.path_index.docs:
                    return self.path_index.docs[clean_path]
                return f"No documentation found for exact path: {path}"

            # First check if the path exists directly regardless of method
            if clean_path in self.path_index.docs:
                return self.path_index.docs[clean_path]

            matching_paths = set()  # Smart search tries multiple methods in succession
            if search_method == "smart":
                methods = ["prefix", "segments", "keywords", "regex"]
                for method in methods:
                    result = self.get_path_documentation(path, method, max_results)
                    if not result.startswith("No documentation found"):
                        return f"Smart search found results using {method} method:\n{result.split(':', 1)[1]}"
                return f"No documentation found for path: {path} using smart search"

            elif search_method == "regex":
                # Create a regex pattern to match against paths
                # Escape special regex characters in the path to treat it as literal
                # unless the user provided actual regex expressions
                if any(c in path for c in ".*+?()[]{}|^$"):
                    # Path likely contains regex already, use as is
                    pattern = path
                else:
                    # Treat as a partial match (contains search)
                    pattern = f".*{re.escape(clean_path)}.*"

                # Call find_paths_by_pattern with the created pattern
                # This leverages all the optimization and caching already present
                found_paths = self.find_paths_by_pattern(
                    pattern=pattern, use_regex=True, case_sensitive=True
                )

                # Convert to set for consistency with other methods
                matching_paths.update(found_paths)

            elif search_method == "prefix":
                # First use the prefixes index for paths this is a prefix of (more efficient)
                if hasattr(self.path_index, "prefixes"):
                    # Find all prefixes of the path that are in the index
                    for i in range(1, len(clean_path) + 1):
                        prefix = clean_path[:i]
                        if prefix in self.path_index.prefixes:
                            matching_paths.update(self.path_index.prefixes[prefix])
                            # If we have enough results, we can stop early
                            if (
                                len(matching_paths) >= max_results * 2
                            ):  # Get 2x to allow for sorting later
                                break

                # Then find paths that are prefixes of the query path
                # This is for cases where the user has entered a longer path but we only have docs
                # for a parent path
                prefix_matches = [
                    p for p in self.path_index.paths if p.startswith(clean_path)
                ]
                matching_paths.update(prefix_matches)

                # If still not enough, look for paths that this is a prefix of
                if len(matching_paths) < max_results:
                    parent_matches = [
                        p for p in self.path_index.paths if clean_path.startswith(p)
                    ]
                    matching_paths.update(parent_matches)

            elif search_method == "segments":
                # Split the path into segments and find paths that contain these segments
                segments = clean_path.split("/")

                # Use the segments index if available
                if hasattr(self.path_index, "segments"):
                    # Start with the more specific segments (longer ones)
                    # This improves relevance by prioritizing more specific matches
                    sorted_segments = sorted(segments, key=len, reverse=True)

                    # Use a weighted approach - paths that match more segments are more relevant
                    segment_matches = {}

                    for segment in sorted_segments:
                        if segment and segment in self.path_index.segments:
                            for p in self.path_index.segments[segment]:
                                if p in segment_matches:
                                    segment_matches[p] += (
                                        1 + len(segment) * 0.1
                                    )  # Weight by segment length
                                else:
                                    segment_matches[p] = 1 + len(segment) * 0.1

                    # Sort by the number of matching segments
                    matching_paths.update(
                        [
                            p
                            for p, _ in sorted(
                                segment_matches.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[: max_results * 2]
                        ]
                    )
                else:
                    # Fall back to a simpler approach if no segments index
                    for segment in segments:
                        if segment:  # Skip empty segments
                            segment_matches = [
                                p
                                for p in self.path_index.paths
                                if segment in p.split("/")
                            ]
                            matching_paths.update(segment_matches)

            elif search_method == "keywords":
                # Extract keywords and find paths that contain these keywords
                keywords = set()
                for part in clean_path.split("/"):
                    # Add smaller parts of each segment as keywords (min 3 chars)
                    for i in range(len(part)):
                        for j in range(i + 3, len(part) + 1):
                            keywords.add(part[i:j].lower())

                # Use the keywords index if available
                if hasattr(self.path_index, "keywords"):
                    # Prioritize longer keywords (more specific)
                    sorted_keywords = sorted(keywords, key=len, reverse=True)

                    # Use a weighted approach for keyword matching
                    keyword_matches = {}

                    for keyword in sorted_keywords:
                        if keyword in self.path_index.keywords:
                            for p in self.path_index.keywords[keyword]:
                                if p in keyword_matches:
                                    keyword_matches[p] += (
                                        1 + len(keyword) * 0.05
                                    )  # Weight by keyword length
                                else:
                                    keyword_matches[p] = 1 + len(keyword) * 0.05

                    # Sort by the number of matching keywords
                    matching_paths.update(
                        [
                            p
                            for p, _ in sorted(
                                keyword_matches.items(),
                                key=lambda x: x[1],
                                reverse=True,
                            )[: max_results * 2]
                        ]
                    )
                else:
                    # Fall back to a simpler approach if no keywords index
                    # Sort keywords by length to prioritize longer (more specific) keywords
                    sorted_keywords = sorted(keywords, key=len, reverse=True)
                    for keyword in sorted_keywords[
                        :10
                    ]:  # Limit to top 10 keywords for performance
                        if len(keyword) > 2:  # Only consider keywords with length > 2
                            keyword_matches = [
                                p for p in self.path_index.paths if keyword in p.lower()
                            ]
                            matching_paths.update(keyword_matches)
                            if len(matching_paths) >= max_results * 2:
                                break

            # Limit results and prepare response
            matching_paths = list(matching_paths)
            if not matching_paths:
                return f"No documentation found for path: {path} using {search_method} search method"

            # Sort by relevance - for now using length as a proxy for relevance
            # Shorter paths for prefix matches, longer paths for keyword/segment matches
            if search_method == "prefix":
                matching_paths.sort(key=len)
            else:
                # For keywords and segments, use path similarity for sorting
                # Using Levenshtein distance would be ideal, but we'll use a simpler approach
                matching_paths.sort(key=lambda p: abs(len(p) - len(clean_path)))

            # Limit the number of results
            matching_paths = matching_paths[:max_results]

            # Format the result
            if len(matching_paths) == 1:
                match = matching_paths[0]
                return (
                    f"Documentation for match '{match}':\n{self.path_index.docs[match]}"
                )
            else:
                result = f"Found {len(matching_paths)} matches using {search_method} search method:\n"
                for i, match in enumerate(matching_paths, 1):
                    result += f"\n{i}. {match}\n"
                    doc = self.path_index.docs[match].strip()
                    result += f"   {doc[:200]}"
                    if len(doc) > 200:
                        result += "..."
                        result += "\n   (truncated for brevity)"
                return result

        except Exception as e:
            return f"Error retrieving documentation: {str(e)}"


def run_server() -> None:
    """Run the MCP server."""
    logger.info("Starting MCP server...")
    mcp = FastMCP("IMAS")
    try:
        DataDictionaryServer(mcp)
        mcp.run()
    except KeyboardInterrupt:
        logger.info("Stopping MCP server...")


if __name__ == "__main__":
    run_server()
