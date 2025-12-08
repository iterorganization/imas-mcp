"""Structure-specific models that don't depend on search functionality."""

from pydantic import BaseModel, Field


class HierarchyMetrics(BaseModel):
    """Metrics about IDS hierarchy structure."""

    total_nodes: int = 0
    leaf_nodes: int = 0
    max_depth: int = 0
    branching_factor: float = 0.0
    complexity_score: float = 0.0


class DomainDistribution(BaseModel):
    """Distribution of physics domains within an IDS."""

    domain: str
    node_count: int = 0
    percentage: float = 0.0
    key_paths: list[str] = Field(default_factory=list)


class NavigationHints(BaseModel):
    """Hints for navigating the IDS structure."""

    entry_points: list[str] = Field(default_factory=list)
    common_patterns: list[str] = Field(default_factory=list)
    drill_down_suggestions: list[str] = Field(default_factory=list)


class StructureAnalysis(BaseModel):
    """Enhanced structure analysis for IDS."""

    hierarchy_metrics: HierarchyMetrics = Field(default_factory=HierarchyMetrics)
    domain_distribution: list[DomainDistribution] = Field(default_factory=list)
    navigation_hints: NavigationHints = Field(default_factory=NavigationHints)
    complexity_summary: str = ""
    organization_pattern: str = ""
