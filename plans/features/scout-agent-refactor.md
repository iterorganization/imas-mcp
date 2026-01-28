# Scout Agent Refactor Plan

## Problem Analysis

After analyzing the `scout files` CLI and related code agents, I've identified several critical issues:

### 1. Context Window Accumulation (~907k → 963k tokens in 30 steps)

**Root Cause**: smolagents' `CodeAgent` accumulates ALL step history in `self.memory.steps`:

```python
# From smolagents.agents.MultiStepAgent.write_memory_to_messages():
def write_memory_to_messages(self, summary_mode: bool = False):
    messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
    for memory_step in self.memory.steps:  # <-- ALL steps included
        messages.extend(memory_step.to_messages(summary_mode=summary_mode))
    return messages
```

Each step includes:
- **model_output**: Full LLM reasoning (~2-10k tokens)
- **tool_calls**: Tool invocation details
- **observations**: Full tool output (up to `MAX_LENGTH_TRUNCATE_CONTENT=20000` chars each)

With 30 steps, each having ~30k tokens average = **~900k tokens**.

### 2. Findings Not Properly Persisted to Graph

Current tools persist data correctly but the **agent output/summary is lost**:

| Tool | Persistence | Issue |
|------|-------------|-------|
| `QueueFilesTool` | ✅ SourceFile nodes created | Works correctly |
| `AddNoteTool` | ✅ Notes saved to YAML | Only used for manual observations |
| `RunCommandTool` | ❌ Output lost after LLM processes | No capture of discoveries |

**Key Problem**: Interesting patterns discovered via `rg` or `fd` commands are processed by the LLM but not systematically captured. The LLM may find "5 files with IMAS patterns" but only queue 2, losing context about the other 3.

### 3. Agent Unaware of Approaching Limits

The agent has no visibility into:
- Current step count vs max_steps
- Remaining budget vs cost_limit_usd
- Whether it should prioritize summarization over exploration

**Current behavior**: Hits max_steps hard limit, cuts off mid-exploration with incomplete output.

## Proposed Architecture: Windowed Context with Checkpointing

### Core Changes

```
┌─────────────────────────────────────────────────────────────────┐
│                     Scout Session Manager                        │
├─────────────────────────────────────────────────────────────────┤
│  Window 1 (steps 1-10)  │  Window 2 (steps 11-20)  │  Window 3  │
│  └─ Summary → Graph     │  └─ Summary → Graph      │  └─ Final  │
├─────────────────────────────────────────────────────────────────┤
│              Session State (Neo4j + YAML)                        │
│  - Files discovered per window                                   │
│  - Patterns found                                                │
│  - Areas explored                                                │
│  - Areas remaining                                               │
└─────────────────────────────────────────────────────────────────┘
```

### 1. Windowed Execution with Session Checkpointing

**New class: `ScoutSession`**

```python
@dataclass
class ScoutSession:
    """Persistent session state for multi-window exploration."""
    
    facility: str
    session_id: str
    window_size: int = 10  # Steps per window
    max_windows: int = 3   # Total windows before forced summary
    
    # Persisted to graph/YAML after each window
    discoveries: list[Discovery] = field(default_factory=list)
    areas_explored: list[str] = field(default_factory=list)
    areas_remaining: list[str] = field(default_factory=list)
    patterns_found: dict[str, int] = field(default_factory=dict)
    
    # Current window metrics
    current_window: int = 0
    steps_in_window: int = 0
    
    def checkpoint(self) -> None:
        """Save session state to graph and YAML."""
        # Persist to Neo4j as ScoutSession node
        # Persist to facility infrastructure as current_session
        
    def should_summarize(self) -> bool:
        """Check if current window is complete."""
        return self.steps_in_window >= self.window_size
    
    def summarize_window(self, agent_output: str) -> str:
        """Generate concise window summary for next context."""
        # LLM call to compress findings
        # Return ~500 token summary
```

### 2. Context Summarization Between Windows

**New module: `imas_codex/agentic/summarizer.py`**

```python
async def summarize_exploration_window(
    memory_steps: list[MemoryStep],
    discoveries: list[Discovery],
    model: str = "gemini-3-flash-preview",
) -> WindowSummary:
    """Compress a window of exploration into structured summary.
    
    Returns:
        WindowSummary with:
        - files_found: List of paths with scores
        - patterns_found: Dict of pattern -> count
        - areas_explored: List of directories/patterns searched
        - areas_remaining: Suggested next areas
        - key_findings: 3-5 bullet points
    """
```

### 3. Agent Limit Awareness

**Inject context into each step via step_callbacks:**

```python
def create_limit_aware_callback(
    session: ScoutSession,
    monitor: AgentMonitor,
) -> Callable[[MemoryStep], None]:
    """Create callback that injects limit awareness."""
    
    def callback(step: MemoryStep) -> None:
        session.steps_in_window += 1
        
        # Check if approaching limits
        remaining_steps = session.window_size - session.steps_in_window
        remaining_budget = monitor.remaining_usd()
        
        if remaining_steps <= 2:
            # Inject warning into next prompt
            step.observations += (
                f"\n⚠️ LIMIT WARNING: {remaining_steps} steps remaining "
                f"in current window. Begin summarizing discoveries."
            )
        
        if remaining_budget and remaining_budget < 0.50:
            step.observations += (
                f"\n⚠️ BUDGET WARNING: ${remaining_budget:.2f} remaining. "
                f"Queue remaining files and prepare final summary."
            )
    
    return callback
```

### 4. Enhanced Graph Persistence

**New node type: `ScoutDiscovery`**

```yaml
# In imas_codex/schemas/facility.yaml
ScoutDiscovery:
  description: A discovery made during exploration
  attributes:
    id:
      description: Unique identifier
      range: string
      identifier: true
    session_id:
      description: Scout session that made this discovery
      range: string
    discovery_type:
      description: Type of discovery
      range: DiscoveryType  # file, pattern, path, convention
    path:
      description: File or directory path
      range: string
    pattern:
      description: Pattern that matched
      range: string
    interest_score:
      description: Computed interest score
      range: float
    context:
      description: Surrounding context from command output
      range: string
    window_number:
      description: Window in which discovery was made
      range: integer
```

**Tool enhancement: Capture discovery context**

```python
class QueueFilesTool(Tool):
    def forward(self, file_paths: list[str], interest_score: float = 0.7,
                context: str = "") -> str:
        """Queue files AND persist discovery context."""
        # Existing queue logic...
        
        # NEW: Create ScoutDiscovery nodes with context
        for path in file_paths:
            self._create_discovery_node(
                path=path,
                interest_score=interest_score,
                context=context[:500],  # Truncated context
                discovery_type="file",
            )
```

## Implementation Plan

### Phase 1: Session Management (Priority: High)

**Files to create:**
- `imas_codex/agentic/scout_session.py` - Session state management
- `imas_codex/schemas/scout.yaml` - ScoutSession, ScoutDiscovery schemas

**Files to modify:**
- `imas_codex/agentic/explore.py` - Integrate ScoutSession
- `imas_codex/agentic/tools.py` - Add context capture to tools
- `imas_codex/cli.py` - Add `scout resume`, `scout status` commands

**Tasks:**
1. [ ] Create `ScoutSession` dataclass with checkpoint methods
2. [ ] Add `ScoutSession` node to Neo4j schema
3. [ ] Modify `ExplorationAgent` to use windowed execution
4. [ ] Add session resume capability to CLI

### Phase 2: Context Summarization (Priority: High)

**Files to create:**
- `imas_codex/agentic/summarizer.py` - Window summarization

**Tasks:**
1. [ ] Create `summarize_exploration_window()` function
2. [ ] Create `WindowSummary` structured output model
3. [ ] Integrate summarization at window boundaries
4. [ ] Test context reduction (goal: <10k tokens per window carryover)

### Phase 3: Limit Awareness (Priority: Medium)

**Files to modify:**
- `imas_codex/agentic/monitor.py` - Add limit injection
- `imas_codex/agentic/explore.py` - Use limit-aware callbacks

**Tasks:**
1. [ ] Create `create_limit_aware_callback()` function
2. [ ] Add step/budget warnings to observations
3. [ ] Test graceful degradation near limits
4. [ ] Add "wrapping up" mode when limits approach

### Phase 4: Enhanced Persistence (Priority: Medium)

**Files to modify:**
- `imas_codex/agentic/tools.py` - Context capture
- `imas_codex/schemas/facility.yaml` - ScoutDiscovery schema

**Tasks:**
1. [ ] Add `ScoutDiscovery` node type
2. [ ] Modify `QueueFilesTool` to capture context
3. [ ] Add `CapturePatternTool` for non-file discoveries
4. [ ] Link discoveries to sessions and windows

### Phase 5: CLI Improvements (Priority: Low)

**Tasks:**
1. [ ] Add `scout resume <session-id>` command
2. [ ] Add `scout status <facility>` command
3. [ ] Add `scout list` command for sessions
4. [ ] Add `--window-size` and `--max-windows` options

## API Changes

### New CLI Commands

```bash
# Start exploration with windowed execution
imas-codex scout tcv files --window-size 10 --max-windows 5

# Resume interrupted session
imas-codex scout resume tcv --session-id <uuid>

# Check active/recent sessions
imas-codex scout status tcv
imas-codex scout list tcv

# Force summary of current session
imas-codex scout summarize tcv --session-id <uuid>
```

### New Python API

```python
from imas_codex.agentic import ScoutSession, ExplorationAgent

# Create session with explicit windowing
session = ScoutSession(
    facility="tcv",
    window_size=10,
    max_windows=5,
)

# Resume existing session
session = ScoutSession.load(facility="tcv", session_id="...")

# Use with agent
async with ExplorationAgent(facility="tcv", session=session) as agent:
    result = await agent.explore("Find equilibrium codes")
    # Automatically checkpoints at window boundaries
```

## Testing Strategy

### Unit Tests
- `tests/agentic/test_scout_session.py` - Session management
- `tests/agentic/test_summarizer.py` - Window summarization

### Integration Tests
- `tests/integration/test_scout_windowed.py` - Full windowed flow
- `tests/integration/test_scout_resume.py` - Session resume

### Regression Tests
- Ensure token count stays <100k per window
- Ensure all discoveries are persisted
- Ensure budget enforcement works

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Max context per step | ~900k | <100k |
| Discoveries persisted | ~50% | >95% |
| Session resumable | No | Yes |
| Agent limit awareness | None | Warning at 80% |
| Cost for 100 files | $0.10+ | <$0.05 |

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Summarization loses detail | Lost discoveries | Persist raw discoveries before summarizing |
| Session state corruption | Lost progress | Transaction-safe checkpoints |
| LLM summarization cost | Added cost | Use cheap model (flash) for summaries |
| Backward compatibility | Breaking changes | Keep old API, deprecate gradually |

## Timeline Estimate

- Phase 1: 2-3 days
- Phase 2: 1-2 days
- Phase 3: 1 day
- Phase 4: 1-2 days
- Phase 5: 1 day

**Total: 6-9 days**

## Related Files

- [agents/explore.md](../../agents/explore.md) - Explore agent guidelines
- [agents/ingest.md](../../agents/ingest.md) - Ingestion pipeline
- [imas_codex/agentic/explore.py](../../imas_codex/agentic/explore.py) - Current implementation
- [imas_codex/agentic/monitor.py](../../imas_codex/agentic/monitor.py) - Cost monitoring
