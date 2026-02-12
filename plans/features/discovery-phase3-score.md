# Phase 3: Score Pipeline

**Goal**: Implement LLM-based semantic scoring with grounded evidence and frontier expansion  
**Scope**: ~2500 lines (scorer implementation, batch processing, evidence collection, frontier logic)  
**Testable Outputs**: DirectoryScorer class, grounded scoring function, frontier expansion logic, batching  
**Duration**: 3-5 days (1-2 dev)

## Deliverables

### 1. Scorer Implementation

Create `imas_codex/discovery/scorer.py`:

```python
"""LLM-based directory scoring with grounded evidence."""

import json
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic

from imas_codex.discovery.models import (
    DirectoryEvidence,
    PathPurpose,
    ScoredDirectory,
    DirStats,
)


@dataclass
class ScoredBatch:
    """Result of scoring a batch of directories."""
    scored_dirs: list[ScoredDirectory]
    total_cost: float
    model: str
    tokens_used: int


class DirectoryScorer:
    """Score directories using Claude LLM with grounded evidence.
    
    Implements:
    1. Batch prompt construction from DirStats
    2. LLM evidence collection
    3. Deterministic grounded scoring from evidence
    4. Frontier expansion logic
    
    Args:
        model: Model name (default: claude-sonnet-4-5)
        client: Anthropic client (optional, creates new one if not provided)
    
    Example:
        scorer = DirectoryScorer()
        batch = scorer.score_batch(
            directories=[...],
            focus="equilibrium codes",
            threshold=0.7,
        )
    """
    
    # Dimension weights for grounded scoring
    WEIGHTS = {
        "code": 1.0,
        "data": 0.8,
        "imas": 1.2,
    }
    
    def __init__(self, model: str = "claude-sonnet-4-5", client: Optional[Anthropic] = None):
        self.model = model
        self.client = client or Anthropic()
    
    def score_batch(
        self,
        directories: list[dict],  # From graph: {path, dir_stats, parent_path, siblings}
        focus: Optional[str] = None,
        threshold: float = 0.7,
    ) -> ScoredBatch:
        """Score a batch of directories.
        
        Args:
            directories: List of directory info dicts with:
              - path: str
              - dir_stats: dict (serialized DirStats)
              - parent_path: Optional[str]
              - siblings: List[str]
            focus: Natural language focus query (e.g., "equilibrium codes")
            threshold: Min score to expand (0.0-1.0)
        
        Returns:
            ScoredBatch with results and cost
        """
        # Build batched prompt
        prompt = self._build_prompt(directories, focus)
        
        # Call Claude with structured output
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=self._system_prompt(focus),
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )
        
        # Parse response
        result_text = response.content[0].text
        scored_dirs = self._parse_response(result_text, directories, threshold)
        
        # Calculate cost (rough estimate)
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens
        
        # Claude Sonnet 4.5 pricing: $3/$15 per 1M tokens
        cost = (input_tokens * 3 + output_tokens * 15) / 1_000_000
        
        return ScoredBatch(
            scored_dirs=scored_dirs,
            total_cost=cost,
            model=self.model,
            tokens_used=total_tokens,
        )
    
    def _system_prompt(self, focus: Optional[str] = None) -> str:
        """Build system prompt."""
        base = """You are analyzing directories at a fusion research facility to determine their value for IMAS code discovery.

For each directory, collect evidence about its contents and purpose, then provide structured assessment.

## Evidence Categories

1. **path_purpose**: Classify as one of:
   - physics_code: Simulation or analysis code
   - data_files: Scientific data storage
   - documentation: Documentation, wikis
   - configuration: Config files
   - build_artifacts: Compiled outputs
   - test_files: Test suites
   - user_home: Personal directories
   - system: OS or infrastructure
   - unknown: Cannot determine

2. **evidence**: Specific observations:
   - code_indicators: Programming files present (list extensions)
   - data_indicators: Data files present
   - imas_indicators: IMAS-related patterns
   - physics_indicators: Physics domain patterns
   - quality_indicators: Project signals (readme, makefile, git)

3. **should_expand**: Whether to explore children (true/false)

## Scoring

Provide three independent scores (0.0-1.0):
- **score_code**: Value for code discovery (based on code_indicators + quality)
- **score_data**: Value for data discovery (based on data_indicators + size)
- **score_imas**: IMAS relevance (based on imas_indicators + patterns)

Expansion Criteria:
- Expand if score > 0.7 AND likely to contain valuable children
- Don't expand: user_home, system, build_artifacts, test_files, single files

Examples of high-value (expand=true):
- /home/codes/liuqe: equilibrium code with put_slice patterns
- /work/data/experimental: large physics data directory
- /projects/physics: multi-project directory with mixed code

Examples of low-value (expand=false):
- /home/user: personal directory (even if has code)
- /tmp: temporary files
- /usr/lib: system libraries
"""
        
        if focus:
            base += f"\n\n## Special Focus\n\nPrioritize paths related to: {focus}\nBoost scores for relevant matches."
        
        return base
    
    def _build_prompt(
        self,
        directories: list[dict],
        focus: Optional[str] = None,
    ) -> str:
        """Build user prompt with all directories to score."""
        lines = ["Score these directories:\n"]
        
        for i, d in enumerate(directories, 1):
            lines.append(f"\n## Directory {i}: {d['path']}")
            
            # Add context
            if d.get("parent_path"):
                lines.append(f"Parent: {d['parent_path']}")
            
            if d.get("siblings"):
                lines.append(f"Siblings: {', '.join(d['siblings'][:5])}")
            
            # Add DirStats
            stats = d.get("dir_stats", {})
            lines.append(f"\nFile types: {stats.get('file_type_counts', {})}")
            lines.append(f"Total files: {stats.get('total_files', 0)}")
            lines.append(f"Total dirs: {stats.get('total_dirs', 0)}")
            lines.append(f"Size: {stats.get('total_size_bytes', 'unknown')} bytes")
            
            if stats.get("has_readme"):
                lines.append("Has README")
            if stats.get("has_makefile"):
                lines.append("Has Makefile")
            if stats.get("has_git"):
                lines.append("Has .git")
            
            patterns = stats.get("patterns_detected", [])
            if patterns:
                lines.append(f"Patterns: {', '.join(patterns)}")
        
        lines.append("\n\nReturn JSON array with results for each directory (in order).")
        
        return "\n".join(lines)
    
    def _parse_response(
        self,
        response_text: str,
        directories: list[dict],
        threshold: float,
    ) -> list[ScoredDirectory]:
        """Parse Claude's response into ScoredDirectory objects.
        
        Claude returns JSON array with scored directory objects.
        """
        import json
        
        try:
            # Extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_str = response_text[json_start:json_end]
            
            results = json.loads(json_str)
        except (json.JSONDecodeError, ValueError):
            # Fallback: return empty scores for all
            return [
                ScoredDirectory(
                    path=d["path"],
                    path_purpose=PathPurpose.UNKNOWN,
                    description="Parse error",
                    evidence=DirectoryEvidence(),
                    score_code=0.0,
                    score_data=0.0,
                    score_imas=0.0,
                    should_expand=False,
                    skip_reason="LLM response parse failed",
                )
                for d in directories
            ]
        
        scored = []
        for i, result in enumerate(results[:len(directories)]):
            path = directories[i]["path"]
            
            # Extract and validate scores
            score_code = float(result.get("score_code", 0.0))
            score_data = float(result.get("score_data", 0.0))
            score_imas = float(result.get("score_imas", 0.0))
            
            # Clamp scores to [0, 1]
            score_code = max(0.0, min(1.0, score_code))
            score_data = max(0.0, min(1.0, score_data))
            score_imas = max(0.0, min(1.0, score_imas))
            
            # Parse purpose
            try:
                purpose = PathPurpose[result.get("path_purpose", "UNKNOWN").upper()]
            except KeyError:
                purpose = PathPurpose.UNKNOWN
            
            # Build evidence
            evidence = DirectoryEvidence(
                code_indicators=result.get("evidence", {}).get("code_indicators", []),
                data_indicators=result.get("evidence", {}).get("data_indicators", []),
                imas_indicators=result.get("evidence", {}).get("imas_indicators", []),
                physics_indicators=result.get("evidence", {}).get("physics_indicators", []),
                quality_indicators=result.get("evidence", {}).get("quality_indicators", []),
            )
            
            # Compute combined score via grounded function
            combined = self._grounded_score(
                score_code,
                score_data,
                score_imas,
                evidence,
                purpose,
            )
            
            # Check expansion criteria
            should_expand = (
                combined >= threshold
                and result.get("should_expand", False)
                and purpose not in [
                    PathPurpose.USER_HOME,
                    PathPurpose.SYSTEM,
                    PathPurpose.BUILD_ARTIFACTS,
                ]
            )
            
            scored_dir = ScoredDirectory(
                path=path,
                path_purpose=purpose,
                description=result.get("description", ""),
                evidence=evidence,
                score_code=score_code,
                score_data=score_data,
                score_imas=score_imas,
                should_expand=should_expand,
                expansion_reason=result.get("expansion_reason"),
                skip_reason=result.get("skip_reason"),
            )
            
            scored.append(scored_dir)
        
        return scored
    
    def _grounded_score(
        self,
        score_code: float,
        score_data: float,
        score_imas: float,
        evidence: DirectoryEvidence,
        purpose: PathPurpose,
    ) -> float:
        """Compute combined score from dimension scores with evidence adjustments.
        
        Grounded scoring:
        1. Start with dimension average
        2. Boost for quality indicators
        3. Suppress for low-quality signals
        4. Apply purpose-based priors
        
        Returns:
            Combined score (0.0-1.0)
        """
        # Start with weighted average
        base_score = (
            score_code * self.WEIGHTS["code"]
            + score_data * self.WEIGHTS["data"]
            + score_imas * self.WEIGHTS["imas"]
        ) / sum(self.WEIGHTS.values())
        
        # Adjust for quality signals
        quality_boost = 0.0
        if "readme" in evidence.quality_indicators:
            quality_boost += 0.05
        if "makefile" in evidence.quality_indicators:
            quality_boost += 0.05
        if "git" in evidence.quality_indicators:
            quality_boost += 0.05
        
        # Suppress certain purposes
        purpose_multiplier = 1.0
        if purpose in [PathPurpose.SYSTEM, PathPurpose.BUILD_ARTIFACTS]:
            purpose_multiplier = 0.3
        elif purpose == PathPurpose.TEST_FILES:
            purpose_multiplier = 0.6
        
        # Apply evidence of size/completeness
        if len(evidence.code_indicators) > 3:
            quality_boost += 0.05
        if len(evidence.imas_indicators) > 0:
            quality_boost += 0.10
        
        combined = (base_score + quality_boost) * purpose_multiplier
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, combined))


class ScorePhase:
    """Execute complete score phase for a facility.
    
    Orchestrates:
    1. Query graph for scanned but unscored paths
    2. Batch into groups for LLM
    3. Score each batch
    4. Persist results to graph
    5. Track cost and frontier expansion
    """
    
    def __init__(self, config, graph_client):
        self.config = config
        self.graph = graph_client
        self.scorer = DirectoryScorer(
            model=config.model,
        )
    
    def run(self) -> dict:
        """Execute score phase.
        
        Returns:
            Statistics dict with:
            {
              "scored": 123,
              "expanded": 45,
              "cost": 1.23,
              "total_time_ms": 15000,
              "throughput": 30,  # paths/minute
            }
        """
        import time
        start = time.monotonic()
        
        # 1. Get all scanned but unscored paths
        paths = self.graph.get_scored_paths(
            facility=self.config.facility,
            limit=None,  # Get all
        )
        
        scored = 0
        expanded = 0
        total_cost = 0.0
        
        # 2. Process in batches
        batch_size = self.config.batch_size
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i+batch_size]
            
            # Check budget
            if self.config.budget and total_cost >= self.config.budget:
                break
            
            # Score batch
            result = self.scorer.score_batch(
                directories=batch,
                focus=self.config.focus,
                threshold=self.config.threshold,
            )
            
            total_cost += result.total_cost
            
            # Persist results
            self.graph.persist_score_results(
                facility=self.config.facility,
                scored_dirs=result.scored_dirs,
            )
            
            scored += len(result.scored_dirs)
            expanded += sum(1 for d in result.scored_dirs if d.should_expand)
        
        duration_ms = int((time.monotonic() - start) * 1000)
        throughput = (scored * 60000) // max(duration_ms, 1000)
        
        return {
            "scored": scored,
            "expanded": expanded,
            "cost": total_cost,
            "total_time_ms": duration_ms,
            "throughput": throughput,  # paths/minute
        }
```

### 2. Frontier Expansion Logic

Create `imas_codex/discovery/frontier.py`:

```python
"""Frontier tracking and expansion logic."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FrontierState:
    """State of discovery frontier."""
    total_paths: int = 0
    pending: int = 0  # Ready for scan
    scanned: int = 0  # Awaiting score
    scored: int = 0  # Scored, may expand
    skipped: int = 0  # Dead-end or error
    expanded_this_cycle: int = 0


def get_discovery_stats(graph_client, facility: str) -> dict:
    """Get current discovery statistics from graph.
    
    Cypher:
        MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
        WITH p.status AS status, count(*) AS cnt
        RETURN {
          total: sum(cnt),
          pending: sum(IF(status='pending') THEN cnt ELSE 0),
          scanned: sum(IF(status='scanned') THEN cnt ELSE 0),
          scored: sum(IF(status='scored') THEN cnt ELSE 0),
          skipped: sum(IF(status='skipped') THEN cnt ELSE 0)
        }
    """
    pass


def has_frontier(graph_client, facility: str) -> bool:
    """Check if there are paths to expand.
    
    True if any scored paths have should_expand=true
    """
    pass


def get_expansion_depth(graph_client, facility: str) -> int:
    """Get current maximum depth explored.
    
    Cypher:
        MATCH (p:FacilityPath)-[:AT_FACILITY]->(f:Facility {id: $facility})
        RETURN max(p.depth)
    """
    pass


def should_continue_discovery(
    stats: dict,
    accumulated_cost: float,
    budget: Optional[float],
    max_cycles: int,
    current_cycle: int,
) -> tuple[bool, Optional[str]]:
    """Determine if discovery should continue.
    
    Returns:
        (should_continue, reason_if_stopping)
    """
    # Check budget
    if budget and accumulated_cost >= budget:
        return False, f"Budget exhausted (${accumulated_cost:.2f})"
    
    # Check cycles
    if current_cycle >= max_cycles:
        return False, f"Max cycles ({max_cycles}) reached"
    
    # Check frontier
    if stats["pending"] == 0 and stats["scanned"] == 0:
        return False, "No more frontier to explore"
    
    return True, None
```

### 3. Unit Tests

Create `tests/discovery/test_scorer.py`:

```python
"""Unit tests for directory scorer."""

import pytest
from unittest.mock import Mock, patch
from imas_codex.discovery.scorer import DirectoryScorer
from imas_codex.discovery.models import DirectoryEvidence, PathPurpose


@pytest.fixture
def mock_client():
    """Mock Anthropic client."""
    mock = Mock()
    
    # Mock response
    mock.messages.create.return_value = Mock(
        content=[Mock(text='''[
  {
    "path": "/home/codes",
    "path_purpose": "physics_code",
    "description": "Physics simulation code",
    "evidence": {
      "code_indicators": ["py", "f90"],
      "imas_indicators": ["put_slice"],
      "quality_indicators": ["readme", "git"]
    },
    "score_code": 0.9,
    "score_data": 0.2,
    "score_imas": 0.8,
    "should_expand": true,
    "expansion_reason": "High-value physics code"
  }
]''')],
        usage=Mock(input_tokens=500, output_tokens=200),
    )
    
    return mock


def test_grounded_score(mock_client):
    """Test grounded scoring function."""
    scorer = DirectoryScorer(client=mock_client)
    
    evidence = DirectoryEvidence(
        code_indicators=["py", "f90"],
        imas_indicators=["put_slice"],
        quality_indicators=["readme", "git"],
    )
    
    score = scorer._grounded_score(
        score_code=0.9,
        score_data=0.2,
        score_imas=0.8,
        evidence=evidence,
        purpose=PathPurpose.PHYSICS_CODE,
    )
    
    # Should have base score plus quality boost
    assert 0.7 < score < 1.0  # Rough bounds


def test_batch_scoring(mock_client):
    """Test batch scoring."""
    scorer = DirectoryScorer(client=mock_client)
    
    directories = [
        {
            "path": "/home/codes",
            "dir_stats": {
                "file_type_counts": {"py": 42, "f90": 12},
                "total_files": 54,
                "patterns_detected": ["put_slice"],
            },
        }
    ]
    
    batch = scorer.score_batch(directories=directories)
    
    assert len(batch.scored_dirs) == 1
    assert batch.scored_dirs[0].path == "/home/codes"
    assert batch.total_cost > 0
    assert batch.tokens_used > 0


def test_purpose_suppression(mock_client):
    """Test that system directories get lower scores."""
    scorer = DirectoryScorer(client=mock_client)
    
    evidence = DirectoryEvidence()
    
    # System directory should be suppressed
    system_score = scorer._grounded_score(
        score_code=0.9,
        score_data=0.9,
        score_imas=0.9,
        evidence=evidence,
        purpose=PathPurpose.SYSTEM,
    )
    
    # Physics code should not be suppressed
    code_score = scorer._grounded_score(
        score_code=0.9,
        score_data=0.9,
        score_imas=0.9,
        evidence=evidence,
        purpose=PathPurpose.PHYSICS_CODE,
    )
    
    assert system_score < code_score
```

Create `tests/discovery/test_frontier.py`:

```python
"""Tests for frontier tracking."""

import pytest
from imas_codex.discovery.frontier import should_continue_discovery


def test_should_continue_with_budget_exhausted():
    """Test stopping when budget exhausted."""
    stats = {"pending": 10, "scanned": 5}
    should_continue, reason = should_continue_discovery(
        stats=stats,
        accumulated_cost=100.0,
        budget=50.0,  # Budget exceeded
        max_cycles=10,
        current_cycle=1,
    )
    
    assert should_continue is False
    assert "Budget" in reason


def test_should_continue_with_empty_frontier():
    """Test stopping when no more paths."""
    stats = {"pending": 0, "scanned": 0}
    should_continue, reason = should_continue_discovery(
        stats=stats,
        accumulated_cost=10.0,
        budget=100.0,
        max_cycles=10,
        current_cycle=1,
    )
    
    assert should_continue is False
    assert "frontier" in reason.lower()


def test_should_continue_within_limits():
    """Test continuing when within limits."""
    stats = {"pending": 10, "scanned": 5}
    should_continue, reason = should_continue_discovery(
        stats=stats,
        accumulated_cost=10.0,
        budget=100.0,
        max_cycles=10,
        current_cycle=1,
    )
    
    assert should_continue is True
    assert reason is None
```

## Success Criteria

- ✅ DirectoryScorer correctly calls Claude API
- ✅ Grounded scoring combines dimensions with evidence weighting
- ✅ Frontier expansion logic respects constraints
- ✅ Batch processing respects budget limits
- ✅ JSON parsing handles various response formats
- ✅ All tests pass with >85% coverage

## Testing Checklist

```bash
# Unit tests with mocks
pytest tests/discovery/test_scorer.py -v
pytest tests/discovery/test_frontier.py -v

# Mock API tests
pytest tests/discovery/ -v -k "mock"

# Cost tracking
pytest tests/discovery/ -v -k "cost or budget"
```

## Pricing/Cost Model

**Claude Sonnet 4.5** (as of 2024):
- Input: $3 per 1M tokens
- Output: $15 per 1M tokens

**Cost per directory**:
- ~40-50 input tokens per directory (path + stats + context)
- ~200-300 output tokens per directory (evidence + scores)
- 25-directory batch ≈ 1000 input + 6000 output tokens
- ≈ 0.003 + 0.090 = **$0.093 per batch ≈ $0.004 per directory**

**Budget implications**:
- $10 budget: ~2,500 paths scored
- $100 budget: ~25,000 paths scored
- $1,000 budget: ~250,000 paths scored

## Continuation Points

- **Phase 4**: Implement discover command with cycle control
- **Phase 5**: Add cross-facility learning
- **Phase 6**: Comprehensive testing and validation
- **Cost optimization**: Batch larger sets, caching, cheaper models for pre-filtering
