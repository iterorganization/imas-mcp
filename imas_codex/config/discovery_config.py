"""
Discovery configuration loader.

Loads and merges pattern configuration from YAML files in config/patterns/.
Provides a unified interface for scanner, scorer, and frontier modules.

Configuration structure:
    config/patterns/
    ├── exclude.yaml          # Exclusion patterns (loaded first)
    ├── file_types.yaml       # File extension categories
    └── scoring/
        ├── base.yaml         # Dimension weights and thresholds
        ├── data_systems.yaml # IMAS, MDSplus, HDF5, etc.
        └── physics.yaml      # Physics domain patterns

Facility-specific exclusions are merged from *_private.yaml files.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExclusionConfig:
    """Configuration for path exclusions.

    Two-tier exclusion system:
    1. Deterministic exclusions (this config) - fast, no SSH/LLM cost
    2. LLM scoring (scorer.py) - nuanced decisions after scanning

    Facility-specific exclusions can be merged via merge_facility_excludes().
    """

    directories: list[str] = field(default_factory=list)
    """Directory basenames to exclude (e.g., '.git', '__pycache__')"""

    patterns: list[str] = field(default_factory=list)
    """Fnmatch patterns for directory names (e.g., 'anaconda*')"""

    path_prefixes: list[str] = field(default_factory=list)
    """Absolute path prefixes to never scan (e.g., '/proc', '/tmp')"""

    archive_extensions: list[str] = field(default_factory=list)
    """File extensions indicating archive directories to skip"""

    dotfile_exclude: bool = True
    """Whether to exclude dotfiles by default"""

    dotfile_allow: list[str] = field(default_factory=list)
    """Specific dotfiles to allow (e.g., '.github')"""

    scratch_names: list[str] = field(default_factory=list)
    """Directory names indicating scratch/temp space (e.g., 'scratch', 'tmp')"""

    scratch_path_patterns: list[str] = field(default_factory=list)
    """Full path patterns for scratch directories (e.g., '*/scratch/*')"""

    max_depth: int = 15
    """Maximum scan depth"""

    large_dir_threshold: int = 10000
    """Directory entry count that triggers size_skipped"""

    scan_timeout: int = 30
    """Maximum seconds for a single directory scan"""

    def merge_facility_excludes(self, facility_excludes: dict) -> ExclusionConfig:
        """Merge facility-specific exclusions from *_private.yaml.

        Args:
            facility_excludes: Dict with optional keys:
                - directories: list[str]
                - patterns: list[str]
                - path_prefixes: list[str]

        Returns:
            New ExclusionConfig with merged values
        """
        if not facility_excludes:
            return self

        return ExclusionConfig(
            directories=self.directories + facility_excludes.get("directories", []),
            patterns=self.patterns + facility_excludes.get("patterns", []),
            path_prefixes=self.path_prefixes
            + facility_excludes.get("path_prefixes", []),
            archive_extensions=self.archive_extensions,
            dotfile_exclude=self.dotfile_exclude,
            dotfile_allow=self.dotfile_allow,
            scratch_names=self.scratch_names,
            scratch_path_patterns=self.scratch_path_patterns,
            max_depth=self.max_depth,
            large_dir_threshold=self.large_dir_threshold,
            scan_timeout=self.scan_timeout,
        )

    def is_scratch_path(self, path: str) -> bool:
        """Check if path is a scratch/temporary directory.

        Scratch directories are excluded from seeding and scanning.
        """
        basename = path.rstrip("/").split("/")[-1]

        # Check scratch names
        if basename.lower() in [n.lower() for n in self.scratch_names]:
            return True

        # Check scratch path patterns
        for pattern in self.scratch_path_patterns:
            if fnmatch.fnmatch(path, pattern):
                return True

        return False

    def should_exclude(self, path: str) -> tuple[bool, str | None]:
        """Check if a path should be excluded.

        Args:
            path: Absolute path to check

        Returns:
            (should_exclude, reason) tuple
        """
        basename = path.rstrip("/").split("/")[-1]

        # Check path prefixes first (absolute exclusions)
        for prefix in self.path_prefixes:
            if path.startswith(prefix):
                return True, f"path_prefix:{prefix}"

        # Check scratch patterns (before other checks)
        if self.is_scratch_path(path):
            return True, f"scratch:{basename}"

        # Check explicit directory exclusions
        if basename in self.directories:
            return True, f"directory:{basename}"

        # Check fnmatch patterns
        for pattern in self.patterns:
            if fnmatch.fnmatch(basename, pattern):
                return True, f"pattern:{pattern}"

        # Check archive extensions (directories ending with .tar.gz, .zip, etc.)
        for ext in self.archive_extensions:
            if basename.endswith(ext):
                return True, f"archive:{ext}"

        # Check dotfiles
        if basename.startswith("."):
            if self.dotfile_exclude:
                if basename not in self.dotfile_allow:
                    return True, "dotfile"

        return False, None

    def filter_paths(self, paths: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
        """Filter a list of paths, returning included and excluded.

        Args:
            paths: List of absolute paths

        Returns:
            (included_paths, excluded_with_reasons) tuple
        """
        included = []
        excluded = []

        for path in paths:
            should_exclude, reason = self.should_exclude(path)
            if should_exclude:
                excluded.append((path, reason))
            else:
                included.append(path)

        return included, excluded


@dataclass
class FileTypeConfig:
    """Configuration for file type categorization."""

    code_extensions: set[str] = field(default_factory=set)
    """All code file extensions"""

    data_extensions: set[str] = field(default_factory=set)
    """All data file extensions"""

    config_extensions: set[str] = field(default_factory=set)
    """All config file extensions"""

    doc_extensions: set[str] = field(default_factory=set)
    """All documentation file extensions"""

    tree_sitter_languages: set[str] = field(default_factory=set)
    """Extensions with tree-sitter support"""

    quality_patterns: dict[str, list[str]] = field(default_factory=dict)
    """Quality indicator name -> patterns"""

    quality_boosts: dict[str, float] = field(default_factory=dict)
    """Quality indicator name -> score boost"""


@dataclass
class PatternDef:
    """A single regex pattern for detection."""

    pattern: str
    weight: float
    description: str
    compiled: re.Pattern | None = None

    def __post_init__(self):
        try:
            self.compiled = re.compile(self.pattern, re.IGNORECASE)
        except re.error as e:
            logger.warning(f"Invalid regex pattern '{self.pattern}': {e}")


@dataclass
class DataSystemConfig:
    """Configuration for a data system (IMAS, MDSplus, etc.)."""

    name: str
    description: str
    patterns: list[PatternDef]
    path_patterns: list[str]

    def matches_path(self, path: str) -> bool:
        """Check if path matches any path patterns."""
        basename = path.lower()
        for pattern in self.path_patterns:
            if fnmatch.fnmatch(basename, pattern.lower()):
                return True
        return False


@dataclass
class PhysicsDomainConfig:
    """Configuration for a physics domain."""

    name: str
    description: str
    patterns: list[PatternDef]
    path_patterns: list[str]


@dataclass
class ScoringConfig:
    """Configuration for scoring."""

    dimension_weights: dict[str, float] = field(default_factory=dict)
    """Dimension name -> weight for combined scoring"""

    thresholds: dict[str, float] = field(default_factory=dict)
    """Threshold name -> value (expand, low_value, high_value)"""

    purpose_multipliers: dict[str, float] = field(default_factory=dict)
    """Purpose name -> score multiplier"""

    suppressed_purposes: set[str] = field(default_factory=set)
    """Purposes that should not be expanded"""

    evidence_boosts: dict[str, float] = field(default_factory=dict)
    """Evidence type -> score boost"""

    data_systems: dict[str, DataSystemConfig] = field(default_factory=dict)
    """Data system configurations"""

    physics_domains: dict[str, PhysicsDomainConfig] = field(default_factory=dict)
    """Physics domain configurations"""

    def get_all_patterns(self) -> list[str]:
        """Get all regex patterns for quick rg detection."""
        patterns = []
        for ds in self.data_systems.values():
            for p in ds.patterns:
                patterns.append(p.pattern)
        for pd in self.physics_domains.values():
            for p in pd.patterns:
                patterns.append(p.pattern)
        return patterns

    def get_rg_pattern(self) -> str:
        """Get combined pattern for ripgrep -e flag."""
        patterns = self.get_all_patterns()
        # Combine with | for single rg call
        # Limit to most important patterns to avoid command line limits
        important = patterns[:20]  # Top 20 patterns
        return "|".join(important)


@dataclass
class DiscoveryConfig:
    """Complete discovery configuration."""

    exclusions: ExclusionConfig
    file_types: FileTypeConfig
    scoring: ScoringConfig

    @classmethod
    def load(cls, patterns_dir: Path | None = None) -> DiscoveryConfig:
        """Load configuration from YAML files.

        Args:
            patterns_dir: Override patterns directory (default: config/patterns/)

        Returns:
            Loaded DiscoveryConfig
        """
        if patterns_dir is None:
            patterns_dir = Path(__file__).parent / "patterns"

        exclusions = cls._load_exclusions(patterns_dir / "exclude.yaml")
        file_types = cls._load_file_types(patterns_dir / "file_types.yaml")
        scoring = cls._load_scoring(patterns_dir / "scoring")

        return cls(
            exclusions=exclusions,
            file_types=file_types,
            scoring=scoring,
        )

    @classmethod
    def _load_exclusions(cls, path: Path) -> ExclusionConfig:
        """Load exclusion configuration."""
        if not path.exists():
            logger.warning(f"Exclusion config not found: {path}")
            return ExclusionConfig()

        with path.open() as f:
            data = yaml.safe_load(f) or {}

        dotfile_config = data.get("dotfile_patterns", {})
        scratch_config = data.get("scratch_patterns", {})

        return ExclusionConfig(
            directories=data.get("directories", []),
            patterns=data.get("patterns", []),
            path_prefixes=data.get("path_prefixes", []),
            archive_extensions=data.get("archive_extensions", []),
            dotfile_exclude=dotfile_config.get("exclude", True),
            dotfile_allow=dotfile_config.get("allow", []),
            scratch_names=scratch_config.get("names", []),
            scratch_path_patterns=scratch_config.get("path_patterns", []),
            max_depth=data.get("max_depth", 15),
            large_dir_threshold=data.get("large_dir_threshold", 10000),
            scan_timeout=data.get("scan_timeout", 30),
        )

    @classmethod
    def _load_file_types(cls, path: Path) -> FileTypeConfig:
        """Load file type configuration."""
        if not path.exists():
            logger.warning(f"File types config not found: {path}")
            return FileTypeConfig()

        with path.open() as f:
            data = yaml.safe_load(f) or {}

        config = FileTypeConfig()

        # Process code extensions
        for lang_config in data.get("code", {}).values():
            extensions = lang_config.get("extensions", [])
            config.code_extensions.update(extensions)
            if lang_config.get("tree_sitter", False):
                config.tree_sitter_languages.update(extensions)

        # Process data extensions
        for cat_config in data.get("data", {}).values():
            config.data_extensions.update(cat_config.get("extensions", []))

        # Process config extensions
        for cat_config in data.get("config", {}).values():
            config.config_extensions.update(cat_config.get("extensions", []))

        # Process doc extensions
        for cat_config in data.get("docs", {}).values():
            config.doc_extensions.update(cat_config.get("extensions", []))

        # Process quality indicators
        for name, indicator in data.get("quality_indicators", {}).items():
            config.quality_patterns[name] = indicator.get("patterns", [])
            config.quality_boosts[name] = indicator.get("boost", 0.0)

        return config

    @classmethod
    def _load_scoring(cls, scoring_dir: Path) -> ScoringConfig:
        """Load scoring configuration from directory."""
        config = ScoringConfig()

        # Load base configuration
        base_path = scoring_dir / "base.yaml"
        if base_path.exists():
            with base_path.open() as f:
                base_data = yaml.safe_load(f) or {}

            # Dimension weights
            for dim, dim_config in base_data.get("dimensions", {}).items():
                config.dimension_weights[dim] = dim_config.get("weight", 1.0)

            # Thresholds
            config.thresholds = base_data.get("thresholds", {})

            # Purpose multipliers
            config.purpose_multipliers = base_data.get("purpose_multipliers", {})

            # Suppressed purposes
            config.suppressed_purposes = set(base_data.get("suppressed_purposes", []))

            # Evidence boosts
            config.evidence_boosts = base_data.get("evidence_boosts", {})

        # Load data systems
        ds_path = scoring_dir / "data_systems.yaml"
        if ds_path.exists():
            with ds_path.open() as f:
                ds_data = yaml.safe_load(f) or {}

            for name, ds_config in ds_data.items():
                if name == "version":
                    continue
                config.data_systems[name] = DataSystemConfig(
                    name=name,
                    description=ds_config.get("description", ""),
                    patterns=[
                        PatternDef(
                            pattern=p.get("pattern", ""),
                            weight=p.get("weight", 0.5),
                            description=p.get("description", ""),
                        )
                        for p in ds_config.get("patterns", [])
                    ],
                    path_patterns=ds_config.get("path_patterns", []),
                )

        # Load physics domains
        physics_path = scoring_dir / "physics.yaml"
        if physics_path.exists():
            with physics_path.open() as f:
                physics_data = yaml.safe_load(f) or {}

            for name, pd_config in physics_data.items():
                if name == "version":
                    continue
                config.physics_domains[name] = PhysicsDomainConfig(
                    name=name,
                    description=pd_config.get("description", ""),
                    patterns=[
                        PatternDef(
                            pattern=p.get("pattern", ""),
                            weight=p.get("weight", 0.5),
                            description=p.get("description", ""),
                        )
                        for p in pd_config.get("patterns", [])
                    ],
                    path_patterns=pd_config.get("path_patterns", []),
                )

        return config


@lru_cache(maxsize=1)
def get_discovery_config() -> DiscoveryConfig:
    """Get cached discovery configuration.

    Returns:
        DiscoveryConfig instance (cached)
    """
    return DiscoveryConfig.load()


def get_exclusion_config_for_facility(facility: str) -> ExclusionConfig:
    """Get exclusion config with facility-specific patterns merged.

    Loads base exclusions from patterns/exclude.yaml, then merges
    any facility-specific exclusions from the facility's private config.

    Args:
        facility: Facility identifier (e.g., 'iter', 'epfl', 'jet')

    Returns:
        ExclusionConfig with merged facility-specific excludes
    """
    base_config = get_discovery_config().exclusions

    try:
        from imas_codex.discovery.facility import get_facility_infrastructure

        infra = get_facility_infrastructure(facility)
        facility_excludes = infra.get("excludes", {})

        if facility_excludes:
            logger.debug(
                f"Merging facility excludes for {facility}: "
                f"{len(facility_excludes.get('directories', []))} dirs, "
                f"{len(facility_excludes.get('path_prefixes', []))} prefixes"
            )
            return base_config.merge_facility_excludes(facility_excludes)

    except Exception as e:
        logger.debug(f"Could not load facility excludes for {facility}: {e}")

    return base_config


def clear_config_cache() -> None:
    """Clear the configuration cache (for testing/reloading)."""
    get_discovery_config.cache_clear()
