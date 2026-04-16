# ISN StandardNameKind Assessment

## Problem Statement

The `StandardNameKind` enum in ISN (imas-standard-names) has three values:
`scalar`, `vector`, `metadata`. The imas-codex project is expanding its DD node
classification to distinguish `physical_quantity` from `geometric_quantity`. This
plan assesses whether StandardNameKind needs corresponding changes.

## Current State

### StandardNameKind (ISN)

```yaml
StandardNameKind:
  scalar:     "Scalar quantity"
  vector:     "Vector quantity (R,Z or multi-component)"
  metadata:   "Non-measurable concept or classification"
```

**Used by**:
- ISN grammar validation: `metadata` names cannot have units
- ISN `create_standard_name_entry()`: validates kind against grammar structure
- imas-codex SN pipeline: `kind` field on StandardName graph nodes
- imas-codex MCP tools: `kind` filter on `search_standard_names`

### NodeCategory (imas-codex, proposed)

```yaml
NodeCategory:
  physical_quantity:   "Measurable physics quantity (T_e, I_p, B_tor)"
  geometric_quantity:  "Shape/configuration parameter (elongation, triangularity)"
  coordinate:          "Independent variable (time, rho_tor_norm)"
  structural:          "Storage artifact"
  identifier:          "Typed descriptor"
  error:               "Uncertainty bounds"
  metadata:            "Bookkeeping"
```

### Key Difference

| Axis | StandardNameKind | NodeCategory |
|------|-----------------|--------------|
| **What it classifies** | Standard name entries | DD path nodes |
| **Taxonomy** | Mathematical nature (scalar vs vector vs non-measurable) | Pipeline participation (embed/search/extract/enrich) |
| **"metadata" means** | Non-measurable concept (coordinate system, grid type) | Bookkeeping subtree (ids_properties, code) |
| **Physics/geometric distinction** | Not present | Explicit (physical_quantity vs geometric_quantity) |

## Fitness Assessment

### Is StandardNameKind Fit for Purpose?

**Yes, for its current scope.** StandardNameKind answers: "What is the mathematical
dimensionality of this named quantity?" This is the right question for:

1. **Grammar validation**: Vector names require component suffixes (R, Z, phi)
2. **Unit constraints**: Metadata names cannot carry physical units
3. **Catalog organization**: Group scalars vs vectors vs non-measurable

### What It Doesn't Cover (And Shouldn't)

| Missing Axis | Why ISN Shouldn't Own It |
|---|---|
| **Physical vs geometric** | This is a DD classification concern. Both types produce standard names using the same grammar. `elongation` and `electron_temperature` are both scalar — the grammar doesn't care if one is geometric. |
| **Coordinate vs measurement** | Coordinates rarely get standard names (filtered by SN_SOURCE_CATEGORIES in imas-codex). If they did, they'd be scalar — no special grammar needed. |
| **Derived vs primitive** | Interesting for physics but irrelevant to naming grammar. |

### How the Grammar Already Handles Geometric vs Physical

ISN grammar has **two distinct base branches** for quantities:

| Grammar Branch | Type | Example Standard Name |
|---|---|---|
| `physical_base` (open-vocab) | Physical | `electron_temperature`, `plasma_current` |
| `geometric_base` (restricted) | Structured geometric concepts | `position`, `outline`, `trajectory`, `surface_normal` |
| `physical_base` (open-vocab) | Geometric scalars | `elongation`, `triangularity`, `minor_radius` |

**Important nuance**: Not all geometric quantities use `geometric_base`. Shape parameters
like `elongation` and `triangularity` parse as open-vocab `physical_base` entries, while
structured geometric concepts (`position`, `outline`, `trajectory`) use the restricted
`geometric_base` branch with geometry-specific rules (object/geometry qualification,
orientation/path completeness, extent dimensionality).

**Key implication**: `geometric_quantity` (NodeCategory in DD) does NOT always map to
`geometric_base` (ISN grammar branch). The DD classification describes the *physics origin*
of the quantity (shape vs measurement), while the grammar branch describes the *naming
structure*. These are orthogonal axes:
- Structural geometry (`outline`, `trajectory`, `centroid`) → `geometric_base` grammar
- Shape scalars (`elongation`, `triangularity`, `minor_radius`) → `physical_base` grammar (open-vocab)
- Both are `geometric_quantity` in the DD NodeCategory

## Design Options

### Option A: Keep StandardNameKind Unchanged (Recommended)

- `scalar`, `vector`, `metadata` — no changes
- Geometric/physical distinction managed by NodeCategory on the DD side
- Filtering by geometric/physical uses a query-time DD join (no denormalization)
- If denormalization is ever justified by query performance, use an authoritative
  recomputation into a **set-valued** `source_categories` (not singular) because a
  StandardName can aggregate multiple DD sources with different NodeCategory values

**Pros**: No ISN API breakage. Clean separation of concerns. Grammar already handles it.
**Cons**: Filtering standard names by geometric/physical requires a DD join (see below).

### Option B: Add `geometric` Flag to StandardName

```yaml
StandardNameKind:
  scalar:     "Scalar quantity"
  vector:     "Vector quantity"
  metadata:   "Non-measurable concept"

# New property (not part of kind):
is_geometric: bool  # True for shape/configuration parameters
```

**Pros**: Direct filterability on StandardName.
**Cons**: Duplicates information from DD source. Adds maintenance burden.
  Who sets this flag? The LLM (unreliable) or the DD node_category (requires propagation).

### Option C: Split Kind into 5 Values

```yaml
StandardNameKind:
  physical_scalar:    "Physical scalar (temperature, pressure)"
  geometric_scalar:   "Geometric scalar (elongation, aspect_ratio)"
  physical_vector:    "Physical vector (velocity, B-field)"
  geometric_vector:   "Geometric vector (boundary outline R,Z)"
  metadata:           "Non-measurable concept"
```

**Pros**: Maximum expressiveness.
**Cons**: Combinatorial explosion. Geometric vectors are extremely rare.
  Grammar validation would need 5 code paths instead of 3. Overcomplicated.

### Option D: Add Tags Instead

```python
# On StandardName:
tags: list[str]  # e.g. ["equilibrium", "geometric", "shape"]
```

**Pros**: Flexible. No enum changes. Tags already exist on StandardName.
**Cons**: Uncontrolled vocabulary. No validation guarantees.

## Recommendation

**Option A: Keep StandardNameKind unchanged.**

Rationale:
1. StandardNameKind is a grammar-level classification. Geometric vs physical is a
   physics-level classification that belongs in the DD.
2. The ISN grammar already distinguishes geometric from physical through `physical_base`
   vocabulary — elongation, triangularity, etc. have their own grammar rules.
3. Adding geometric awareness to ISN would create a dependency on imas-codex's
   NodeCategory enum, violating the clean boundary between the two projects.
4. If imas-codex needs to filter standard names by geometric/physical, it can join
   through the DD source path: `(sn)<-[:HAS_STANDARD_NAME]-(node:IMASNode)` and
   filter on `node.node_category`.

### Minor Enhancement (Optional)

If filtering standard names by geometric/physical becomes a frequent need,
**prefer query-time joins** over denormalized properties:

```cypher
// Query-time: find geometric standard names via linked DD nodes
MATCH (node:IMASNode)-[:HAS_STANDARD_NAME]->(sn:StandardName)
WHERE node.node_category = 'geometric_quantity'
RETURN DISTINCT sn.id, sn.description
```

**Why not a denormalized property**: A StandardName aggregates many source paths.
After consolidation/attachment, a single `source_category` value would be unstable —
the "primary" source path is arbitrary. If denormalization is truly needed later
(e.g., for first-class semantic-search filtering where DD join is insufficient),
use a **multivalued** `source_categories` set and recompute it authoritatively from
linked nodes. Note: query-time join won't cover manual/reference-only names that
lack DD node links — accept this limitation or add denormalization at that point.

## ISN Changes Needed

**None for StandardNameKind.** The enum is fit for purpose.

### Potential Grammar Vocabulary Additions

If the imas-codex SN pipeline identifies new geometric quantities that lack grammar
vocabulary, file ISN issues for:
- New `physical_base` entries (e.g., `squareness`, `ovality`, `indentation`)
- New `subject` entries if geometric context matters (e.g., `plasma_boundary`)

These are normal vocabulary additions, not structural changes.

## Relationship to imas-codex Plan

The imas-codex plan (`dd-unified-classification.md`) handles:
- NodeCategory split (physical_quantity + geometric_quantity)
- Propagating category to SN pipeline
- MCP tool filtering
- Graph recovery

This ISN plan confirms: **no ISN changes needed to support the imas-codex classification work.**

The boundary remains clean:
- **ISN owns**: grammar structure, vocabulary, validation rules, StandardNameKind enum
- **imas-codex owns**: DD classification, pipeline participation, NodeCategory, enrichment strategy

### Validation Risk Note

ISN semantic checks distinguish geometry-related names (object/geometry qualification,
component vs coordinate, orientation/path completeness, extent dimensionality). The
SN compose prompt in imas-codex should handle geometric quantities prescriptively:

- **Structural geometry** (`outline`, `trajectory`, `centroid`, `surface_normal`) →
  guide the LLM toward `geometric_base` grammar patterns
- **Shape scalars** (`elongation`, `triangularity`, `minor_radius`) →
  keep in open-vocab `physical_base` (standard naming pattern)

**Action required in imas-codex**: The compose prompt (`compose_dd.md`) does not
currently surface `node_category`. Pass either `node_category` or a derived
`geometry_hint` into the compose context so the LLM can select the appropriate
grammar branch. Without this, geometric quantities may be misgenerated.

## RD Review History

### Round 1 (Findings → Changes)
| # | Finding | Resolution |
|---|---------|------------|
| 1 | ISN has `geometric_base` branch — plan didn't acknowledge | Rewrote grammar section to cover both branches |
| 2 | `geometric_quantity` (DD) ≠ `geometric_base` (ISN) | Added explicit mapping note: orthogonal axes |
| 3 | `source_category` should be multivalued or query-time | Changed to query-time join recommendation |
| 4 | Validation risk if geometric treated identically to physical | Added validation risk note |
| 5 | Boundary wording needs sharpening | Refined boundary statement |

### Round 2 (Findings → Changes)
| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | `geometric_base` labeled "Geometric vectors" — too narrow | Non-blocking | Reworded to "restricted branch for structured geometric concepts" |
| 2 | Stale `source_category` bullet conflicts with join recommendation | Blocking | Removed; cons now points to DD join |
| 3 | Cypher example uses `sn.name` (should be `sn.id`), missing DISTINCT | Blocking | Fixed query; added note about manual/reference-only names |
| 4 | Validation risk note too vague | Non-blocking | Made prescriptive: structural geometry → `geometric_base`, shape scalars → `physical_base`. Added action item for compose prompt. |
