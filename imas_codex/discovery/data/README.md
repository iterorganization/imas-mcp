# Signal Discovery Workflow

This module discovers, classifies, and validates data signals from fusion facility MDSplus trees.

## Graphical Abstract

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         discover signals <facility>                       │
└────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           SCAN WORKER (1)                                │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │  Epoch Detection  │ → │  Signal Creation  │ → │   INTRODUCED_IN   │  │
│  │  (shot changes)   │    │  (from added_paths)│   │   + REMOVED_IN    │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                    │                                     │
│                          status: discovered                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                         (epochs must exist)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         ENRICH WORKER (2)                                │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Claim Batch    │ → │  LLM Classification│ → │   Update Graph    │  │
│  │   (25 signals)   │    │  (physics_domain)  │    │   + description   │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                    │                                     │
│                          status: enriched                                │
│                          (cost tracked)                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                       (requires epoch_id set)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          CHECK WORKER (1)                                │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐  │
│  │   Claim Batch    │ → │  Remote MDSplus    │ → │   CHECKED_VIA     │  │
│  │  (10 signals)    │    │  (run_python_script)│   │   relationship   │  │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘  │
│                                    │                                     │
│                          status: checked                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

## State Machine

Signals progress through states via worker transitions:

```
                  enrich_worker         check_worker
    discovered ──────────────────► enriched ──────────────► checked
        │                              │                        │
        │                              │                        │
        └──────────────────────────────┴────────────────────────┘
                           (failures → skipped/failed)
```

**States** (FacilitySignalStatus enum):
| State | Description | Next Worker |
|-------|-------------|-------------|
| `discovered` | Signal exists in tree, node_path known | enrich_worker |
| `enriched` | LLM classified physics_domain, description | check_worker |
| `checked` | MDSplus access confirmed, shape/dtype known | (terminal) |
| `skipped` | Excluded by filter or policy | (terminal) |
| `failed` | Error during processing | (terminal) |

## Worker Coordination

Workers coordinate via `claimed_at` timestamp pattern:

1. **Claim**: Worker sets `claimed_at = datetime()` while status unchanged
2. **Process**: Worker performs work (LLM call, SSH script, etc.)
3. **Complete**: Worker updates status to next state, clears `claimed_at = null`
4. **Orphan recovery**: Claims older than 5 minutes are automatically reclaimed

### Stopping Conditions

Each worker has independent stopping logic:

```python
# scan_worker stops when:
state.discover_idle_count >= 3  # No new trees to process

# enrich_worker stops when:
state.budget_exhausted          # cost_limit reached
state.signal_limit_reached      # signal_limit reached  
state.stop_requested            # User interrupt (Ctrl+C)

# check_worker stops when:
state.check_idle_count >= 3     # No pending work
AND enriching_done              # Enrich worker has finished
AND not has_pending_check_work  # No more enriched signals
```

### Worker Dependencies

```
scan_worker ─────► enrich_worker ─────► check_worker
    │                   │                    │
    │                   │                    └── Requires epoch_id on signals
    │                   └── Any discovered signal can be enriched
    └── Creates signals from epoch added_paths
```

**Critical dependency**: check_worker only claims signals with `epoch_id IS NOT NULL`. This ensures:
- We have a valid `TreeModelVersion.first_shot` for the MDSplus query
- The signal is known to exist in that shot range
- Epoch detection must run before check can proceed

## Epoch Detection

The scan worker detects "epochs" - periods where the tree structure remained constant.

### How Epoch Detection Works

1. **Binary search**: Scan shot range finding structure changes
2. **Version fingerprinting**: Compare node paths between shots
3. **Diff computation**: Track `added_paths` and `removed_paths`

### Signal Lifecycle Tracking

Epochs enable symmetric lifecycle tracking:

```cypher
// Signal introduced in v2
(signal)-[:INTRODUCED_IN]->(epoch_v2)

// Signal removed in v4
(signal)-[:REMOVED_IN]->(epoch_v4)

// Epoch lineage
(epoch_v3)-[:PRECEDED_BY]->(epoch_v2)-[:PRECEDED_BY]->(epoch_v1)
```

**Processing order**: Epochs are sorted by version and processed sequentially. For each epoch:
- `added_paths` → Create FacilitySignal + `INTRODUCED_IN` edge
- `removed_paths` → Create `REMOVED_IN` edge to existing signal

### Checkpoints

Epoch detection results are cached locally:
```
~/.local/share/imas-codex/checkpoints/data/{facility}_{tree}_epochs.json
```

Use `--force` to re-detect epochs:
```bash
discover signals tcv --force  # Re-runs epoch detection
```

## CLI Options

```bash
discover signals <facility> [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `--tree TEXT` | Specific tree(s) to process | All configured |
| `--shot INT` | Reference shot number | From settings |
| `--cost-limit FLOAT` | Max LLM cost in USD | 0.50 |
| `--signal-limit INT` | Max signals to enrich | Unlimited |
| `--scan-only` | Only run scan worker | False |
| `--enrich-only` | Only run enrich worker | False |
| `--force` | Re-detect epochs | False |
| `--no-rich` | Disable progress display | False |
| `--seed` | Run from epoch seeding | False |

## Expected Run Output

For a fresh TCV run (`discover signals tcv --no-rich --seed`):

```
Phase 1: Epoch Detection
- Scanning results tree shot range [1000, 80000]
- Binary search finds version boundaries
- Creating TreeModelVersion nodes with added/removed paths

Phase 2: Signal Creation  
- Creating FacilitySignal nodes from each epoch's added_paths
- Creating INTRODUCED_IN edges (signal → epoch)
- Creating REMOVED_IN edges for paths that disappeared

Phase 3: LLM Enrichment
- Claiming batches of 25 discovered signals
- Classifying physics_domain (magnetics, mhd, pellets, etc.)
- Generating descriptions from accessor paths
- Tracking token cost toward --cost-limit

Phase 4: MDSplus Checking
- Claiming batches of 10 enriched signals
- Executing check_signals.py via SSH
- Recording shape, dtype on success
- Creating CHECKED_VIA relationships to DataAccess

Summary:
- Epochs created: N
- Signals discovered: N  
- Signals enriched: N ($X.XX cost)
- Signals checked: N
```

## Graph Schema

### Nodes

```cypher
(:FacilitySignal {
    id: "tcv:general/results/ip",
    facility_id: "tcv",
    physics_domain: "magnetics",
    name: "IP",
    accessor: "data(\\results::ip)",
    tree_name: "results",
    node_path: "\\results::ip",
    status: "checked",
    epoch_id: "tcv:results:v3",  // Links to TreeModelVersion
    discovered_at: datetime,
    enriched_at: datetime,
    checked_at: datetime
})

(:TreeModelVersion {
    id: "tcv:results:v3",
    facility_id: "tcv",
    tree_name: "results",
    version: 3,
    first_shot: 45000,
    last_shot: 79999,
    node_count: 1523,
    nodes_added: 12,
    nodes_removed: 3
})
```

### Relationships

```cypher
// Signal lifecycle
(signal)-[:INTRODUCED_IN]->(epoch)  // Signal first appeared in this epoch
(signal)-[:REMOVED_IN]->(epoch)     // Signal removed in this epoch

// Epoch lineage  
(epoch_v3)-[:PRECEDED_BY]->(epoch_v2)

// Access verification
(signal)-[:CHECKED_VIA]->(data_access)
```

## Module Structure

```
imas_codex/discovery/data/
├── __init__.py       # Exports: get_data_discovery_stats, clear_facility_signals
├── models.py         # LLM output models (SignalClassification, etc.)
├── parallel.py       # Core worker implementation (1800+ lines)
├── progress.py       # DataProgressDisplay for rich output
└── README.md         # This file
```

## Clearing Data

To reset signal discovery for a facility:

```bash
discover clear --domain signals tcv
```

This removes:
- All FacilitySignal nodes for the facility
- All TreeModelVersion nodes for the facility
- All INTRODUCED_IN/REMOVED_IN/CHECKED_VIA relationships
