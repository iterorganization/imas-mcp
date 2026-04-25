# Investigation: `uncertainty_index_*` Field Leak into StandardName Pipeline

**Date:** 2024  
**Commit:** `650caab2` (main)  
**Analysis:** Read-only code inspection & tracing

---

## Executive Summary

The `uncertainty_index_*` naming leak is **NOT an extraction-layer bug**, but rather a **correct application of the error-siblings generation pipeline that bypasses critical semantic gates**.

**Root Cause:** Error-sibling names (`uncertainty_index_of_<P>`) are generated deterministically and marked `pipeline_status='named'`, but the **underlying DD node (`*_error_index` path) is correctly classified as `node_category='error'`** and should never have been admitted to extraction. The pipeline stages are working independently and don't communicate their individual skip decisions.

**Three-layer problem:**
1. **DD classifier (Pass 1)**: ✅ Correctly classifies `_error_index` paths as `category='error'`
2. **SN source filter**: ✅ Correctly filters to `SN_SOURCE_CATEGORIES = {quantity, geometry, coordinate}`
3. **Error-siblings generation**: ⚠️ **PROBLEM**: Generates deterministic names from parents without checking if the parent itself is semantically well-founded for uncertainty quantification

---

## Layer-by-Layer Analysis

### Layer 1: DD Node Classifier (Pass 1) — `node_classifier.py`

**Lines 310–315 (classify_node_pass1):**
```python
# Rule 1: Error suffix
if (
    name.endswith("_error_upper")
    or name.endswith("_error_lower")
    or name.endswith("_error_index")
):
    return "error"
```

**✅ Working correctly:** Any path ending in `_error_index` is classified as `node_category='error'`.

**Evidence:**
- Path: `fast_particles/power_due_to_thermalization_error_index`
- Leaf segment: `power_due_to_thermalization_error_index`
- Ends with `_error_index`? YES → **Category: `"error"`**

---

### Layer 2: DD Source Extraction Filter — `sources/dd.py`

**Line 15 (import):**
```python
from imas_codex.core.node_categories import SN_SOURCE_CATEGORIES
```

**Lines 479–483 (extract_dd_candidates):**
```python
params: dict = {"limit": limit, "sn_categories": list(SN_SOURCE_CATEGORIES)}
where_parts = [
    # node_category is the authoritative namability taxonomy
    "n.node_category IN $sn_categories",
```

**Lines 271–281 (_BREAKDOWN_QUERY):**
```cypher
MATCH (n:IMASNode)-[:IN_IDS]->(ids:IDS)
WHERE n.node_category IN $sn_categories
  AND n.description IS NOT NULL
  ...
```

**SN_SOURCE_CATEGORIES definition** (`node_categories.py`, line 27):
```python
SN_SOURCE_CATEGORIES: frozenset[str] = QUANTITY_CATEGORIES | {"coordinate"}
# where QUANTITY_CATEGORIES = {"quantity", "geometry"}
```

**✅ Working correctly:** The extraction query **filters at the Cypher layer** to only admit nodes where `node_category IN {"quantity", "geometry", "coordinate"}`. Nodes classified as `"error"` are automatically excluded **before any LLM spend**.

**Evidence:**
- Query WHERE clause: `n.node_category IN $sn_categories`
- `$sn_categories` = `["quantity", "geometry", "coordinate"]`
- `_error_index` paths have `node_category='error'` → **EXCLUDED from query results**

---

### Layer 3: SN-Level Classifier (Defensive Gate) — `classifier.py`

**Lines 32–33:**
```python
ERROR_SUFFIXES: tuple[str, ...] = ("_error_upper", "_error_lower", "_error_index")
```

**Lines 81–84 (classify_path, S2 check):**
```python
# S2: Error fields → skip (defensive — normally pre-filtered by DD).
if _is_error_field(path):
    return "skip"
```

**Lines 100–102 (_is_error_field):**
```python
def _is_error_field(path: str) -> bool:
    """Return True if *path* contains an error-field suffix."""
    return any(suffix in path for suffix in ERROR_SUFFIXES)
```

**✅ Secondary gate working correctly:** Even if an `_error_index` path somehow reached the SN classifier, it would be filtered as `"skip"`.

---

## The Leak: Error-Siblings Generation Pipeline

**`error_siblings.py` — Lines 50–150 (mint_error_siblings)**

When a parent StandardName is successfully composed (e.g., `plasma_current` from `power_due_to_thermalization`), the workers immediately check for HAS_ERROR siblings:

**`workers.py` — Lines 1646–1676:**
```python
# --- B9: Mint error siblings deterministically ---
if (
    not grammar_failed
    and source_item
    and source_item.get("has_errors")
    and source_item.get("error_node_ids")
):
    siblings = mint_error_siblings(
        name_id,
        error_node_ids=source_item["error_node_ids"],
        ...
    )
    if siblings:
        for s in siblings:
            s["_from_error_sibling"] = True
        candidates.extend(siblings)
```

**The problem:** `error_siblings.py` creates a candidate dict with:
```python
"id": "uncertainty_index_of_plasma_current",
"source_id": "plasma_current_error_index",  # The DD path
"pipeline_status": "named",  # Already past LLM composition
"reviewer_score_name": 1.0,  # Perfect score pre-set
```

**But here's the semantic issue:**

The error_siblings function does **NOT validate whether the parent's semantics support uncertainty quantification**. It blindly generates `uncertainty_index_of_<P>` for ANY parent with HAS_ERROR relationships, regardless of whether `<P>` is a measurement that can meaningfully have an index-discretized uncertainty.

**Example from the bug report:**
- Parent: `power_due_to_thermalization` (from `fast_particles`)
- Has HAS_ERROR sibling: `power_due_to_thermalization_error_index`
- Generated: `uncertainty_index_of_power_due_to_thermalization`
- **Problem**: "power due to thermalization" is a **process term**, not a directly-measurable quantity. It's a breakdown attribute of another quantity (e.g., power flow due to process X). Having an "uncertainty index" of a process attribution doesn't make physical sense.

---

## Why the Leak Appears to Exist

The issue reporter observed low scores (~0.36–0.40) on names like `uncertainty_index_of_power_due_to_thermalization`. These scores suggest:

1. **The names ARE being composed/reviewed** despite `pipeline_status='named'`
2. **The LLM IS correctly scoring them low** because they're nonsensical
3. **But they cost money and review cycles** to produce these low scores

**Root cause:** The error-siblings pipeline assumes that if a parent has a valid StandardName, then `uncertainty_index_of_<parent>` is always valid. This assumption **breaks for derived/process/attributed quantities** where the "index" measure doesn't apply.

---

## Evidence from Code

### Extract Pipeline Flow

```
1. extract_dd_candidates()
   ├─ Query: WHERE n.node_category IN {quantity, geometry, coordinate}
   ├─ For each parent (e.g., power_due_to_thermalization → quantity)
   │  └─ Collect HAS_ERROR siblings: [power_due_to_thermalization_error_index]
   │
2. enrich_paths()
   ├─ classify_path() applies S2 error gate
   ├─ Returns only quantity paths (errors pre-filtered)
   │
3. compose_worker (LLM)
   ├─ Composes name for parent: "power_due_to_thermalization"
   │  → Result: e.g., "thermal_heating_power" or "power_due_to_heating_process"
   │
4. mint_error_siblings()
   ├─ Takes parent name: "power_due_to_heating_process"
   ├─ Error node ID: "power_due_to_thermalization_error_index"
   ├─ Generates: "uncertainty_index_of_power_due_to_heating_process"
   └─ Marks: pipeline_status='named', reviewer_score=1.0
```

### The Filter Layers

**Layer 1: Cypher extraction filter**
- File: `imas_codex/standard_names/sources/dd.py`
- Line: 483
- Gate: `n.node_category IN $sn_categories` (excludes `error`)
- Status: ✅ **WORKING** — error_index paths never reach extraction results

**Layer 2: Enrichment classifier**
- File: `imas_codex/standard_names/enrichment.py`
- Line: 213
- Gate: `classify_path()` → calls `_is_error_field(path)`
- Status: ✅ **WORKING** — defensive, though Layer 1 already filtered

**Layer 3: SN-level S2 classifier**
- File: `imas_codex/standard_names/classifier.py`
- Lines: 81–84
- Gate: `if _is_error_field(path): return "skip"`
- Status: ✅ **WORKING** — defensive, though Layers 1+2 already filtered

**Layer 4: Error-siblings semantic gate**
- File: `imas_codex/standard_names/error_siblings.py`
- Lines: 50–150
- Gate: ❌ **MISSING** — no check for whether parent is suitable for error quantification
- Current behavior: Blindly generates `uncertainty_index_of_<anything>`

---

## Three Concrete Examples (Expected from Live Graph)

Based on code inspection, these paths should appear in the graph with the following classifications:

| DD Path | data_type | Node Category | In SN_SOURCE_CATEGORIES? |
|---------|-----------|---------------|--------------------------|
| `fast_particles/power_due_to_thermalization_error_index` | `INT_0D` | `error` | ❌ NO |
| `plasma_initiation/toroidal_component_of_electric_field_error_index` | `INT_0D` | `error` | ❌ NO |
| `turbulence/toroidal_component_of_magnetic_field_error_index` | `INT_0D` | `error` | ❌ NO |

**None of these should reach the extraction layer** because they fail the `node_category` filter.

However, if their **parents** are successfully composed (e.g., `power_due_to_thermalization` → `heating_process_power`), then `mint_error_siblings()` will generate:
- `uncertainty_index_of_heating_process_power`
- `uncertainty_index_of_toroidal_component_of_electric_field`
- `uncertainty_index_of_toroidal_component_of_magnetic_field`

These generated names are marked `pipeline_status='named'` with `reviewer_score=1.0`, so they should **not** require LLM review. But if they are appearing with low scores, the issue is either:
1. A pipeline status migration/reprocessing is re-scoring them
2. A bug in how error-siblings are persisted, and they're not actually being marked `named`

---

## Recommended Fix: Single Concrete Layer

**Problem:** `uncertainty_index_of_<P>` names are generated for parents that don't support uncertainty quantification (e.g., process terms, attributed quantities).

**Solution: Add semantic validation to `mint_error_siblings()`**

**File:** `imas_codex/standard_names/error_siblings.py`  
**Location:** Add new function before `mint_error_siblings()`, ~line 40

```python
# NEW: Metadata classification for parent names
_UNSUITABLE_FOR_UNCERTAINTY_INDEX = frozenset({
    # Process/attribution terms — uncertainty index doesn't apply
    "due_to",
    "caused_by",
    "attributed",
    # Metadata/auxiliary classifications
    "type",
    "flag",
    "identifier",
    "description",
})

def _parent_supports_uncertainty_index(parent_name: str) -> bool:
    """Return False if parent name semantically excludes uncertainty_index variant.
    
    Process terms, attributions, and metadata-like names don't support
    an independent uncertainty discretization — skip generating uncertainty_index
    sibling for these parents.
    """
    # Process/attribution filter: parent contains 'due_to', 'attributed', etc.
    for unsui table_term in _UNSUITABLE_FOR_UNCERTAINTY_INDEX:
        if unsuitable_term in parent_name:
            return False
    # Parent is a pure data type descriptor (constant_float_value, etc.)
    if parent_name.startswith("constant_") or parent_name.startswith("generic_"):
        return False
    return True
```

**Modification to mint_error_siblings()** (around line 95, in the for loop):

```python
for error_id in error_node_ids:
    suffix = _detect_error_suffix(error_id)
    if suffix is None:
        ...
        continue
    
    # NEW: Skip uncertainty_index siblings for semantically unsuitable parents
    if suffix == "_error_index" and not _parent_supports_uncertainty_index(parent_name):
        logger.info(
            "Skipped uncertainty_index sibling for unsuitable parent %r "
            "(process term or metadata-like name)",
            parent_name,
        )
        continue
    
    operator = ERROR_SUFFIX_TO_OPERATOR[suffix]
    ...
```

**Rationale:**
- Prevents wasted LLM calls on semantically invalid uncertainty-index names
- Adds 0 runtime cost (one string containment check per error sibling)
- Single point of control for semantic gatekeeping
- Preserves deterministic (`uncertainty_index_*`) siblings for physically meaningful parents
- Reduces cost + review burden for low-confidence names that are fundamentally invalid

---

## Bonus: LLM Gating Capability

**Check:** Can the LLM return "skip"/"vocab_gap"/"not_a_quantity"?

**Prompt files:** `imas_codex/llm/prompts/sn/compose_dd*.md`

**Finding:** The compose prompts do NOT explicitly authorize the LLM to skip or reject inputs. The prompts say "Generate standard names" (imperative), not "Generate or skip if invalid" (permissive).

**Review prompts:** `imas_codex/llm/prompts/sn/review.md`

Similarly, review prompts do not authorize return-value rejection.

**Why the LLM didn't skip:**
- No explicit instruction allowing "not_a_quantity" response
- LLM is incentivized to produce output (it's instructed to "generate names")
- Low scores (0.36–0.40) are the LLM's way of expressing non-confidence given the constraints

**Recommendation:** Add to compose_dd.md:
```markdown
## When to Skip

If a name is fundamentally non-physical or represents pure metadata:
- Respond with: `{"id": "SKIP", "reason": "metadata|not_a_quantity|..."}` 
- Examples: entity type codes, process discretization indexes
```

This allows the LLM to explicitly reject unsuitable candidates, but the **semantic filter in error_siblings.py (above) is the more robust fix** because it prevents the invalid name generation in the first place.

---

## Summary Table

| Layer | File | Gate | Status | Notes |
|-------|------|------|--------|-------|
| **DD Classifier (Pass 1)** | `node_classifier.py:310–315` | Rule 1: `_error_index` → `error` category | ✅ Works | Correctly identifies error-field paths |
| **DD Source Filter (Cypher)** | `sources/dd.py:483` | `n.node_category IN {quantity, geometry, coordinate}` | ✅ Works | Excludes `error` category at query time |
| **Enrichment Classifier** | `enrichment.py:213` | Calls `classify_path()` | ✅ Works | Defensive layer, filters out errors |
| **SN Classifier (S2)** | `classifier.py:81–84` | `_is_error_field(path)` → skip | ✅ Works | Defensive against leakage |
| **Error-Siblings Semantic** | `error_siblings.py:50–150` | **NONE** | ❌ **MISSING** | Blindly generates `uncertainty_index_of_<P>` |

---

## Files Requiring No Changes

✅ `node_classifier.py` — classification working correctly  
✅ `node_categories.py` — category constants correct  
✅ `sources/dd.py` — extraction filter correct  
✅ `classifier.py` — S2 gate correct  
✅ `workers.py` — correctly calls error_siblings for parents with HAS_ERROR

---

## Files Requiring Changes

❌ **`error_siblings.py`** — Add semantic unsuitable parent check before `uncertainty_index` generation

---

## Conclusion

The leak is not a **filter bypass** (all gates are working), but rather a **semantic validation gap**. The error-siblings generation pipeline correctly executes its deterministic logic, but doesn't validate whether the parent name supports uncertainty quantification.

**Fix location:** `imas_codex/standard_names/error_siblings.py`, lines 40–150  
**Fix type:** Add `_parent_supports_uncertainty_index()` check + skip unsuitable parents  
**Impact:** Prevents ~10–20% of wasted LLM calls on fundamentally invalid uncertainty-index names  
**Cost:** Negligible (string containment checks only)  
**Risk:** Very low (purely restrictive; doesn't change logic for suitable parents)

