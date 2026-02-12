# Phase 1: Schema & Core Infrastructure

**Goal**: Define LinkML schema updates and core data structures for discovery pipeline  
**Scope**: ~2500 lines (dataclass specs, Cypher queries, test specs, CLI integration)  
**Testable Outputs**: Schema validation, graph operations, CLI integration  
**Duration**: 3-5 days (1-2 dev)

## Deliverables

### 1. Schema Definitions (LinkML)

Update `imas_codex/schemas/facility.yaml` with FacilityPath enhancements:

```yaml
# additions to FacilityPath class

attributes:
  status:
    description: Discovery lifecycle status (pending, scanned, scored, skipped)
    range: DiscoveryStatus
    required: true
    recommended: true
  
  expand_to:
    description: Depth to expand children to (set by score, consumed by scan)
    range: integer
  
  # DirStats - collected during scan
  file_type_counts:
    description: JSON map of extension -> count
    structured_pattern: '{"py": 42, "f90": 12}'
  total_files:
    range: integer
  total_dirs:
    range: integer
  total_size_bytes:
    range: integer
  size_skipped:
    description: True if size calc was skipped (large directory)
    range: boolean
  has_readme:
    range: boolean
  has_makefile:
    range: boolean
  has_git:
    range: boolean
  patterns_detected:
    description: IMAS/physics patterns from rg scan
    multivalued: true
    examples: ["put_slice", "equilibrium", "stellarator"]
  
  # Scoring - set during score phase
  score:
    description: Combined interest (0.0-1.0)
    range: float
  score_code:
    description: Code interest dimension
    range: float
  score_data:
    description: Data interest dimension
    range: float
  score_imas:
    description: IMAS relevance dimension
    range: float
  description:
    description: One-sentence LLM summary
  path_purpose:
    description: Classified purpose
    range: PathPurpose
  evidence:
    description: JSON evidence collected by LLM
    structured_pattern: '{"code_indicators": [...], "quality_indicators": [...]}'
  
  # Metadata
  scanned_at:
    description: When directory was scanned
    range: datetime
  scored_at:
    description: When LLM scored this path
    range: datetime
  skip_reason:
    description: Why path was skipped

# New enums
enums:
  DiscoveryStatus:
    permissible_values:
      pending:
        description: Awaiting scan
      scanned:
        description: Directory enumerated
      scored:
        description: LLM evaluated
      skipped:
        description: Excluded or error
  
  PathPurpose:
    permissible_values:
      physics_code:
      data_files:
      documentation:
      configuration:
      build_artifacts:
      test_files:
      user_home:
      system:
      unknown:
```

### 2. Python Dataclasses

Create `imas_codex/discovery/models.py` with domain models:

```python
"""Domain models for discovery pipeline."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

class DiscoveryStatus(str, Enum):
    """Discovery lifecycle status."""
    PENDING = "pending"
    SCANNED = "scanned"
    SCORED = "scored"
    SKIPPED = "skipped"

class PathPurpose(str, Enum):
    """Classified directory purpose."""
    PHYSICS_CODE = "physics_code"
    DATA_FILES = "data_files"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    BUILD_ARTIFACTS = "build_artifacts"
    TEST_FILES = "test_files"
    USER_HOME = "user_home"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class DirStats:
    """Statistics collected during directory scan."""
    file_type_counts: dict[str, int] = field(default_factory=dict)
    total_files: int = 0
    total_dirs: int = 0
    total_size_bytes: int | None = None
    size_skipped: bool = False
    has_readme: bool = False
    has_makefile: bool = False
    has_git: bool = False
    patterns_detected: list[str] = field(default_factory=list)
    
    @property
    def largest_extension(self) -> str | None:
        """Most common file type."""
        if not self.file_type_counts:
            return None
        return max(self.file_type_counts.items(), key=lambda x: x[1])[0]
    
    def to_dict(self) -> dict:
        """Serialize for graph storage."""
        return {
            "file_type_counts": self.file_type_counts,
            "total_files": self.total_files,
            "total_dirs": self.total_dirs,
            "total_size_bytes": self.total_size_bytes,
            "size_skipped": self.size_skipped,
            "has_readme": self.has_readme,
            "has_makefile": self.has_makefile,
            "has_git": self.has_git,
            "patterns_detected": self.patterns_detected,
        }

@dataclass
class DirectoryEvidence:
    """Evidence collected by LLM for grounded scoring."""
    code_indicators: list[str] = field(default_factory=list)
    data_indicators: list[str] = field(default_factory=list)
    imas_indicators: list[str] = field(default_factory=list)
    physics_indicators: list[str] = field(default_factory=list)
    quality_indicators: list[str] = field(default_factory=list)

@dataclass
class ScoredDirectory:
    """Result of LLM scoring for a directory."""
    path: str
    path_purpose: PathPurpose
    description: str
    evidence: DirectoryEvidence
    score_code: float
    score_data: float
    score_imas: float
    should_expand: bool
    expansion_reason: str | None = None
    skip_reason: str | None = None
    
    @property
    def score(self) -> float:
        """Combined score as mean of dimensions."""
        return (self.score_code + self.score_data + self.score_imas) / 3.0
    
    def to_dict(self) -> dict:
        """Serialize for graph storage."""
        return {
            "path": self.path,
            "path_purpose": self.path_purpose.value,
            "description": self.description,
            "evidence": {
                "code_indicators": self.evidence.code_indicators,
                "data_indicators": self.evidence.data_indicators,
                "imas_indicators": self.evidence.imas_indicators,
                "physics_indicators": self.evidence.physics_indicators,
                "quality_indicators": self.evidence.quality_indicators,
            },
            "score_code": self.score_code,
            "score_data": self.score_data,
            "score_imas": self.score_imas,
            "score": self.score,
            "should_expand": self.should_expand,
            "expansion_reason": self.expansion_reason,
            "skip_reason": self.skip_reason,
        }

@dataclass
class DiscoveryScanConfig:
    """Configuration for scan phase."""
    facility: str
    dry_run: bool = False
    limit: int | None = None  # Max paths to scan this run
    max_sessions: int = 4  # Concurrent SSH sessions
    timeout: int = 30  # Per-command timeout
    max_files_for_size: int = 10_000  # Don't compute size if >10k files
    
    def validate(self):
        """Validate configuration."""
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be ≥ 1")
        if self.timeout < 5:
            raise ValueError("timeout must be ≥ 5 seconds")
        if self.max_files_for_size < 100:
            raise ValueError("max_files_for_size must be ≥ 100")

@dataclass
class DiscoveryScoreConfig:
    """Configuration for score phase."""
    facility: str
    dry_run: bool = False
    batch_size: int = 25  # Paths per LLM call
    budget: float | None = None  # Max spend in USD
    focus: str | None = None  # Natural language focus query
    threshold: float = 0.7  # Min score to expand
    model: str = "claude-sonnet-4-5"
    
    def validate(self):
        """Validate configuration."""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0.0, 1.0]")
        if self.batch_size < 1:
            raise ValueError("batch_size must be ≥ 1")

@dataclass
class DiscoveryStats:
    """Summary statistics for discovery state."""
    facility: str
    total_paths: int = 0
    pending: int = 0
    scanned: int = 0
    scored: int = 0
    skipped: int = 0
    accumulated_cost: float = 0.0
    
    @property
    def frontier_size(self) -> int:
        """Paths awaiting scan."""
        return self.pending
    
    @property
    def completion_fraction(self) -> float:
        """Fraction scored."""
        if self.total_paths == 0:
            return 0.0
        return self.scored / self.total_paths
    
    def to_dict(self) -> dict:
        """Serialize for display."""
        return {
            "facility": self.facility,
            "total_paths": self.total_paths,
            "pending": self.pending,
            "scanned": self.scanned,
            "scored": self.scored,
            "skipped": self.skipped,
            "accumulated_cost": self.accumulated_cost,
            "frontier_size": self.frontier_size,
            "completion_fraction": self.completion_fraction,
        }
```

### 3. Graph Operations

Create `imas_codex/discovery/graph_ops.py`:

```python
"""Graph operations for discovery pipeline."""

from datetime import datetime
from imas_codex.graph.client import GraphClient
from imas_codex.discovery.models import (
    DiscoveryStatus,
    DirStats,
    DiscoveryStats,
    ScoredDirectory,
)

class DiscoveryGraphOps:
    """Graph operations for discovery pipeline."""
    
    def __init__(self, client: GraphClient):
        self.client = client
    
    def seed_facility(self, facility: str) -> list[str]:
        """Create initial pending nodes for facility root paths.
        
        Args:
            facility: Facility ID (e.g., "tcv", "iter")
        
        Returns:
            List of seeded paths
        
        Cypher:
            MATCH (f:Facility {id: $facility})
            UNWIND $root_paths AS root_path
            MERGE (p:FacilityPath {
              id: f.id + ':' + root_path,
              path: root_path,
              depth: 0,
              status: 'pending'
            })
            SET p.created_at = datetime()
            RETURN p.path
        """
        # Get root paths from facility config
        # Then create nodes
        pass
    
    def get_frontier(self, facility: str, limit: int | None = None) -> list[str]:
        """Get pending paths for scanning.
        
        Query graph for:
        1. status = 'pending'
        2. OR (status = 'scanned' AND expand_to > depth)
        
        Returns list of paths to scan.
        """
        pass
    
    def persist_scan_result(
        self,
        facility: str,
        path: str,
        dir_stats: DirStats,
        children: list[str],
        skipped: bool = False,
        skip_reason: str | None = None,
    ):
        """Persist scan result to graph.
        
        Updates:
        - Parent FacilityPath: status=scanned, DirStats fields, scanned_at
        - Creates child FacilityPath nodes: status=pending
        
        Cypher (pseudo):
            MATCH (p:FacilityPath {id: ...})
            SET p.status = 'scanned'
            SET p.total_files = $total_files
            ... (other DirStats fields)
            SET p.scanned_at = datetime()
            WITH p
            UNWIND $children AS child_path
            CREATE (p)-[:CONTAINS]->(child:FacilityPath {
              id: parent_id + ':' + child_path,
              path: child_path,
              depth: p.depth + 1,
              status: 'pending'
            })
        """
        pass
    
    def get_scored_paths(self, facility: str, limit: int | None = None) -> list[dict]:
        """Get scanned but unscored paths for LLM evaluation.
        
        Query:
            MATCH (p:FacilityPath)
            WHERE p.status = 'scanned' AND p.score IS NULL
            RETURN p {.*} LIMIT $limit
        """
        pass
    
    def persist_score_results(
        self,
        facility: str,
        scored_dirs: list[ScoredDirectory],
    ):
        """Persist LLM scores to graph.
        
        For each scored directory:
        - Set score, score_code, score_data, score_imas
        - Set path_purpose, description, evidence
        - Set status = 'scored'
        - If should_expand: set expand_to = depth + 1
        - Set scored_at = now()
        
        Cypher (pseudo):
            UNWIND $scored_dirs AS sd
            MATCH (p:FacilityPath {id: sd.id})
            SET p.status = 'scored'
            SET p.score = sd.score
            SET p.score_code = sd.score_code
            SET p.score_data = sd.score_data
            SET p.score_imas = sd.score_imas
            SET p.path_purpose = sd.path_purpose
            SET p.description = sd.description
            SET p.evidence = sd.evidence
            SET p.scored_at = datetime()
            WITH p, sd
            WHERE sd.should_expand
            SET p.expand_to = p.depth + 1
        """
        pass
    
    def get_stats(self, facility: str) -> DiscoveryStats:
        """Get discovery statistics for facility.
        
        Query:
            MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
            WITH p.status AS status, count(*) AS cnt
            RETURN {
              total: sum(cnt),
              pending: sum(IF status='pending' THEN cnt ELSE 0),
              scanned: sum(IF status='scanned' THEN cnt ELSE 0),
              scored: sum(IF status='scored' THEN cnt ELSE 0),
              skipped: sum(IF status='skipped' THEN cnt ELSE 0)
            }
        """
        pass
```

### 4. Unit Tests

Create `tests/discovery/test_models.py`:

```python
"""Unit tests for discovery models."""

import pytest
from imas_codex.discovery.models import (
    DirStats,
    DirectoryEvidence,
    PathPurpose,
    ScoredDirectory,
    DiscoveryScanConfig,
    DiscoveryScoreConfig,
    DiscoveryStats,
)

def test_dir_stats_serialization():
    """Test DirStats -> dict conversion."""
    stats = DirStats(
        file_type_counts={"py": 42, "f90": 12},
        total_files=54,
        total_dirs=3,
        has_readme=True,
        has_git=True,
    )
    data = stats.to_dict()
    
    assert data["total_files"] == 54
    assert data["file_type_counts"]["py"] == 42
    assert data["has_readme"] is True
    assert data["size_skipped"] is False

def test_scored_directory_combined_score():
    """Test score calculation from dimensions."""
    evidence = DirectoryEvidence(code_indicators=["py", "f90"])
    scored = ScoredDirectory(
        path="/home/codes/liuqe",
        path_purpose=PathPurpose.PHYSICS_CODE,
        description="LIUQE equilibrium code",
        evidence=evidence,
        score_code=0.9,
        score_data=0.7,
        score_imas=0.8,
        should_expand=True,
    )
    
    assert scored.score == pytest.approx(0.8)  # (0.9 + 0.7 + 0.8) / 3

def test_scan_config_validation():
    """Test ScanConfig validation."""
    with pytest.raises(ValueError, match="max_sessions"):
        DiscoveryScanConfig(facility="tcv", max_sessions=0).validate()
    
    with pytest.raises(ValueError, match="timeout"):
        DiscoveryScanConfig(facility="tcv", timeout=2).validate()

def test_score_config_validation():
    """Test ScoreConfig validation."""
    with pytest.raises(ValueError, match="threshold"):
        DiscoveryScoreConfig(facility="tcv", threshold=1.5).validate()
    
    # Valid config should not raise
    DiscoveryScoreConfig(facility="tcv", threshold=0.7).validate()

def test_discovery_stats_properties():
    """Test DiscoveryStats computed properties."""
    stats = DiscoveryStats(
        facility="tcv",
        total_paths=100,
        pending=10,
        scanned=30,
        scored=60,
    )
    
    assert stats.frontier_size == 10
    assert stats.completion_fraction == pytest.approx(0.6)  # 60/100
```

Create `tests/discovery/test_graph_ops.py`:

```python
"""Unit tests for graph operations."""

import pytest
from unittest.mock import Mock, patch
from imas_codex.discovery.graph_ops import DiscoveryGraphOps
from imas_codex.discovery.models import DirStats

@pytest.fixture
def mock_client():
    """Mock GraphClient."""
    return Mock()

@pytest.fixture
def graph_ops(mock_client):
    """DiscoveryGraphOps instance with mock client."""
    return DiscoveryGraphOps(mock_client)

def test_persist_scan_result(graph_ops, mock_client):
    """Test persisting scan results to graph."""
    dir_stats = DirStats(
        file_type_counts={"py": 5},
        total_files=5,
        total_dirs=2,
    )
    
    # This should call client.query() with proper Cypher
    graph_ops.persist_scan_result(
        facility="tcv",
        path="/home/codes",
        dir_stats=dir_stats,
        children=["/home/codes/subdir1", "/home/codes/subdir2"],
    )
    
    # Verify client.query was called
    mock_client.query.assert_called_once()
    cypher = mock_client.query.call_args[0][0]
    assert "MATCH" in cypher
    assert "CREATE" in cypher

def test_get_frontier(graph_ops, mock_client):
    """Test querying frontier for scanning."""
    mock_client.query.return_value = [
        {"path": "/home/codes/dir1"},
        {"path": "/home/codes/dir2"},
    ]
    
    paths = graph_ops.get_frontier(facility="tcv", limit=10)
    
    assert len(paths) == 2
    mock_client.query.assert_called_once()
```

### 5. CLI Integration

Update `imas_codex/cli.py` with scan/score/discover commands (using Click):

```python
import click
from imas_codex.discovery.models import DiscoveryScanConfig, DiscoveryScoreConfig

@click.group()
def discover():
    """Discovery pipeline commands."""
    pass

@discover.command()
@click.argument("facility")
@click.option("--dry-run", is_flag=True)
@click.option("--limit", type=int, default=None)
@click.option("--max-sessions", type=int, default=4)
@click.option("--timeout", type=int, default=30)
def scan(facility, dry_run, limit, max_sessions, timeout):
    """Scan directories on a facility."""
    config = DiscoveryScanConfig(
        facility=facility,
        dry_run=dry_run,
        limit=limit,
        max_sessions=max_sessions,
        timeout=timeout,
    )
    config.validate()
    # Implementation in Phase 2
    click.echo(f"Would scan {facility} (dry_run={dry_run})")

@discover.command()
@click.argument("facility")
@click.option("--dry-run", is_flag=True)
@click.option("--batch-size", type=int, default=25)
@click.option("--budget", type=float, default=None)
@click.option("--focus", type=str, default=None)
@click.option("--threshold", type=float, default=0.7)
@click.option("--model", type=str, default="claude-sonnet-4-5")
def score(facility, dry_run, batch_size, budget, focus, threshold, model):
    """Score directories on a facility."""
    config = DiscoveryScoreConfig(
        facility=facility,
        dry_run=dry_run,
        batch_size=batch_size,
        budget=budget,
        focus=focus,
        threshold=threshold,
        model=model,
    )
    config.validate()
    # Implementation in Phase 3
    click.echo(f"Would score {facility} (budget={budget})")

@discover.command()
@click.argument("facility")
@click.option("--budget", type=float, required=True)
@click.option("--max-cycles", type=int, default=10)
@click.option("--focus", type=str, default=None)
def run(facility, budget, max_cycles, focus):
    """Automated scan+score cycle."""
    click.echo(f"Discover {facility} (budget=${budget})")

@discover.command()
@click.argument("facility")
def status(facility):
    """Show discovery status for facility."""
    click.echo(f"Status for {facility}")
```

## Success Criteria

- ✅ LinkML schema updates pass validation
- ✅ Dataclasses instantiate without errors
- ✅ Graph operations accept proper Cypher queries
- ✅ CLI commands parse arguments and validate config
- ✅ Unit tests achieve >85% coverage
- ✅ No import cycles or dependency issues

## Testing Checklist

```bash
# Type checking
mypy imas_codex/discovery/

# Unit tests
pytest tests/discovery/test_models.py -v
pytest tests/discovery/test_graph_ops.py -v

# CLI parsing
imas-codex discover scan --help
imas-codex discover score --help

# Schema validation (Phase 2)
# uv run imas-codex build schemas
```

## Dependencies

- `pydantic` (validation) - already available
- `rich` (progress) - already available
- Neo4j driver - already available
- Click - already available

## Continuation Points

- **Phase 2**: Implement ScanPhase with actual SSH execution
- **Phase 3**: Implement ScorePhase with actual LLM calls
- **Phase 4**: Implement DiscoverCommand orchestration loop
