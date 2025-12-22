"""
Configuration loading and merging for the Discovery Engine.

Loads core.yaml (universal settings) and facility-specific YAML files,
merging them into a unified configuration for exploration tasks.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def get_config_dir() -> Path:
    """Get the config directory path."""
    return Path(__file__).parent.parent / "config"


def get_facilities_dir() -> Path:
    """Get the facilities config directory."""
    return get_config_dir() / "facilities"


class DiscoverySettings(BaseModel):
    """Settings for the discovery/investigation process."""

    model: str = "anthropic/claude-opus-4.5"
    max_iterations: int = 5
    timeout_seconds: int = 60
    max_output_bytes: int = 102400


class ExcludeSettings(BaseModel):
    """Directory and pattern exclusions."""

    directories: list[str] = Field(default_factory=list)
    patterns: list[str] = Field(default_factory=list)


class ToolCategories(BaseModel):
    """Tools to probe for by category."""

    search: list[str] = Field(default_factory=list)
    tree: list[str] = Field(default_factory=list)
    code_analysis: list[str] = Field(default_factory=list)
    python: list[str] = Field(default_factory=list)
    data: list[str] = Field(default_factory=list)
    system: list[str] = Field(default_factory=list)
    version_control: list[str] = Field(default_factory=list)

    def all_tools(self) -> list[str]:
        """Get flat list of all tools."""
        tools = []
        for category in [
            self.search,
            self.tree,
            self.code_analysis,
            self.python,
            self.data,
            self.system,
            self.version_control,
        ]:
            tools.extend(category)
        return tools


class SafetySettings(BaseModel):
    """Safety constraints for script generation."""

    forbidden_commands: list[str] = Field(default_factory=list)
    max_script_lines: int = 200
    forbidden_patterns: list[str] = Field(default_factory=list)
    allowed_redirections: list[str] = Field(default_factory=list)


class CoreConfig(BaseModel):
    """Core configuration loaded from core.yaml."""

    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    excludes: ExcludeSettings = Field(default_factory=ExcludeSettings)
    tools: ToolCategories = Field(default_factory=ToolCategories)
    safety: SafetySettings = Field(default_factory=SafetySettings)


class MDSplusConfig(BaseModel):
    """MDSplus configuration for a facility."""

    server: str | None = None
    trees: list[str] = Field(default_factory=list)


class KnownSystems(BaseModel):
    """Known data systems at a facility."""

    mdsplus: MDSplusConfig = Field(default_factory=MDSplusConfig)
    diagnostics: list[str] = Field(default_factory=list)
    codes: list[str] = Field(default_factory=list)


class PathConfig(BaseModel):
    """Path configuration for a facility."""

    data: list[str] = Field(default_factory=list)
    code: list[str] = Field(default_factory=list)
    docs: list[str] = Field(default_factory=list)


class FacilityConfig(BaseModel):
    """Configuration for a specific facility."""

    facility: str
    ssh_host: str
    description: str = ""
    hostnames: list[str] = Field(default_factory=list)
    """Hostnames that identify this facility (for local execution detection)."""
    paths: PathConfig = Field(default_factory=PathConfig)
    excludes: ExcludeSettings = Field(default_factory=ExcludeSettings)
    known_systems: KnownSystems = Field(default_factory=KnownSystems)
    exploration_hints: list[str] = Field(default_factory=list)


class MergedConfig(BaseModel):
    """Merged configuration combining core and facility settings."""

    # Core settings
    discovery: DiscoverySettings
    tools: ToolCategories
    safety: SafetySettings

    # Facility settings
    facility: str
    ssh_host: str
    description: str
    hostnames: list[str] = Field(default_factory=list)
    paths: PathConfig
    known_systems: KnownSystems
    exploration_hints: list[str]

    # Merged excludes (core + facility)
    excludes: ExcludeSettings

    def to_context(self) -> dict[str, Any]:
        """Convert to a dict suitable for prompt template context."""
        return {
            "facility": self.facility,
            "ssh_host": self.ssh_host,
            "description": self.description,
            "paths": self.paths.model_dump(),
            "known_systems": self.known_systems.model_dump(),
            "exploration_hints": self.exploration_hints,
            "excludes": self.excludes.model_dump(),
            "available_tools": self.tools.all_tools(),
            "safety": {
                "forbidden_commands": self.safety.forbidden_commands,
                "max_script_lines": self.safety.max_script_lines,
            },
        }


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_core_config() -> CoreConfig:
    """Load the core configuration (cached)."""
    core_path = get_config_dir() / "core.yaml"
    data = load_yaml(core_path)
    return CoreConfig.model_validate(data)


def load_facility_config(facility: str) -> FacilityConfig:
    """Load a facility-specific configuration."""
    facility_path = get_facilities_dir() / f"{facility}.yaml"
    if not facility_path.exists():
        raise ValueError(f"Unknown facility: {facility}. No config at {facility_path}")
    data = load_yaml(facility_path)
    return FacilityConfig.model_validate(data)


def list_facilities() -> list[str]:
    """List all available facility configurations."""
    facilities_dir = get_facilities_dir()
    if not facilities_dir.exists():
        return []
    return [p.stem for p in facilities_dir.glob("*.yaml")]


def merge_excludes(core: ExcludeSettings, facility: ExcludeSettings) -> ExcludeSettings:
    """Merge core and facility exclusion settings."""
    # Combine lists, removing duplicates while preserving order
    dirs = list(dict.fromkeys(core.directories + facility.directories))
    patterns = list(dict.fromkeys(core.patterns + facility.patterns))
    return ExcludeSettings(directories=dirs, patterns=patterns)


def get_config(facility: str) -> MergedConfig:
    """
    Load and merge configuration for a facility.

    Args:
        facility: Facility identifier (e.g., 'epfl', 'jet')

    Returns:
        MergedConfig with core + facility settings combined
    """
    core = load_core_config()
    fac = load_facility_config(facility)

    return MergedConfig(
        # Core settings
        discovery=core.discovery,
        tools=core.tools,
        safety=core.safety,
        # Facility settings
        facility=fac.facility,
        ssh_host=fac.ssh_host,
        description=fac.description,
        hostnames=fac.hostnames,
        paths=fac.paths,
        known_systems=fac.known_systems,
        exploration_hints=fac.exploration_hints,
        # Merged
        excludes=merge_excludes(core.excludes, fac.excludes),
    )
