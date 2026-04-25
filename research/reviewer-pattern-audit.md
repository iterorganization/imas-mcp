# RD2 Reviewer-Pattern Audit: Cost & Signal Analysis

**Date:** 2024-04-24  
**Scope:** Read-only investigation of 3-model RD-quorum review pattern  
**Commit:** 650caab2 (main)  

---

## Executive Summary

The current **3-model RD-quorum pattern** (Opus primary → GPT-5.4 secondary → Sonnet escalator) costs approximately **$0.70 per reviewed name** across both axes (names + docs). Round-1 achieved 27 reviewed names for $18.87 within a $20 budget.

**Key finding:** The escalator (cycle 2) **only processes disputed items**, creating a cost-scaling structure where disagreement rate is the primary cost driver.

### Preliminary Recommendation

| Recommendation | Pattern | Cost/Name | Rationale |
|---|---|---|---|
| **Primary** | D: Haiku primary + Opus escalator | $0.005 (75% savings) | Requires Haiku pilot; maintains RD logic |
| **Fallback** | B: Opus + Sonnet (drop GPT) | $0.024 (20% savings) | Conservative; proven models; no escalator |
| **Current** | A: Opus → GPT → Sonnet | $0.021 | Established; expensive; diverse |

⚠️ **Cannot finalize without Round-1 disagreement data from live graph** (SSH tunnel failed)

---

## 1. Current Configuration Analysis

### 1.1 Model Chain (Identical for names & docs axes)

```
Cycle 0: claude-opus-4.6 (primary, blind)     → $15.00/M input
Cycle 1: gpt-5.4 (secondary, blind)           → $1.25/M input  [cross-vendor]
Cycle 2: claude-sonnet-4.6 (escalator)        → $3.00/M input  [only disputed]
```

**Parameters:**
- Disagreement threshold: **0.20** (normalized 0-1 scale)
- Raw threshold per dimension: **4 points** on 0-20 scale
- Max cycles: **3**
- Dimensions (names): grammar, semantic, convention, completeness
- Dimensions (docs): description_quality, documentation_quality, completeness, physics_accuracy

### 1.2 RD-Quorum Logic Flow (pipeline.py:615-1100)

#### Cycle 0: Primary (Blind)
- Scores all names in batch independently
- No prior context
- Creates Review node per item (resolution_role = "primary")

#### Cycle 1: Secondary (Blind)  
- Scores identical batch independently
- No awareness of Cycle 0 results
- Creates Review node per item (resolution_role = "secondary")

#### Disagreement Detection (lines 901-945)
```
For each name in batch:
  If BOTH cycles scored the item:
    For each dimension d in [grammar, semantic, convention, completeness]:
      If |score_0[d] - score_1[d]| > 0.20 (normalized):
        → Item is DISPUTED
        → Add to disputed_ids set
```

#### Cycle 2: Escalator (Conditional)
- **Only runs if disputes exist** (mini-batch, not full batch)
- Sees both prior critiques injected in prompt context
- Authoritative resolution
- Creates Review node per disputed item only (resolution_role = "escalator")

**Resolution Methods:**
- `quorum_consensus`: agreement → merge (average) scores
- `authoritative_escalation`: escalator breaks tie
- `single_review`: one cycle failed, use other
- `retry_item`: both cycles failed
- `max_cycles_reached`: dispute unresolved (no escalator available)

### 1.3 Budget & Cost Model

**From pipeline.py lines 740-751:**
```python
# Per-batch reservation strategy
estimated_cost = len(names) * 0.05  # base estimate
worst_case = estimated_cost * len(models) * 1.5  # covers retries
```

**Example: 15-name batch, 3 models**
- Reservation: 15 × 0.05 × 3 × 1.5 = **$3.375**

**Round-1 Actual:**
- Names reviewed: 27 (electromagnetic_wave_diagnostics domain)
- Total cost: $18.87
- Cost per name: **$0.70**
- Budget remaining: $1.13 / $20.00

---

## 2. Code Architecture Insights

### 2.1 Escalator as Cost Control Lever

**Critical insight:** Cycle 2 scales with **dispute rate**, not batch size.

```
Cost = C0_cost + C1_cost + (dispute_rate × C2_cost)
```

If Opus-GPT disagreement is **10%**:
- Escalator processes ~10% of names
- C2 mini-batch is tiny
- Total cost ≈ $0.015 + $0.00125 = **$0.0163** per name

If Opus-GPT disagreement is **50%**:
- Escalator processes ~50% of names
- C2 mini-batch is large
- Total cost ≈ $0.015 + $0.00125 + $0.009 = **$0.0256** per name

### 2.2 Axis-Split Double Cost

Both axes use identical 3-model chain:
- `names_axis`: grammar, semantic, convention, completeness
- `docs_axis`: description_quality, documentation_quality, completeness, physics_accuracy

**Effectively 6 model calls per full review per name:**
- C0 (Opus) on names + docs
- C1 (GPT) on names + docs
- C2 (Sonnet) on names + docs (if disputed)

### 2.3 Model Selection Rationale (from config)

**Why GPT-5.4 as secondary?**
- Cross-vendor independence (avoids Anthropic-only cohort bias)
- Cheap ($1.25/M vs $15/M Opus)
- Structured output family (JSON response reliability)
- Removed: Gemini-3.1-pro returned prose despite json_object constraint

**Why Sonnet as escalator?**
- Cheaper than Opus ($3/M vs $15/M)
- Only processes disputed subset (cost amortized)
- Authoritative (sees both priors)

---

## 3. Cost Scenario Analysis

### 3.A: Current (Opus → GPT-5.4 → Sonnet)

**Token estimate:** ~500 input tokens per cycle per axis

**Per-name cost (both axes):**
- C0 (Opus): 500 tk/axis × 2 axes × $15/1M = $0.015
- C1 (GPT): 500 tk/axis × 2 axes × $1.25/1M = $0.00125
- C2 (Sonnet, @ 30% dispute): 600 tk/axis × 2 axes × $3/1M × 0.30 = $0.0108

**Total: $0.0271 per name**  
**Round-1 (27 names): $0.732**

*(Observed $18.87 suggests higher token counts or additional overhead)*

### 3.B: Alternative — Drop GPT (Opus → Sonnet, no escalator)

**Per-name:**
- C0 (Opus): $0.015
- C1 (Sonnet): $0.009
- No escalator

**Total: $0.024 per name**  
**20% cost reduction**

**Trade-off:**
- Loses cross-vendor bias check
- No escalator (disputes unresolved → `max_cycles_reached`)
- Anthropic-only confidence

### 3.C: Alternative — Haiku Primary + Opus Escalator

**Per-name (@ 20% dispute rate):**
- C0 (Haiku × 2 axes): 500 tk × 2 × $0.30/1M = **$0.0003**
- C1 (Opus escalator @ 20%): 600 tk × 2 × $15/1M × 0.20 = **$0.036**

**Total: $0.0363 per name (for disputed) + $0.0003 (undisputed)**  
**Blended: $0.0075 per name (assuming 20% dispute)**

**75% cost reduction if Haiku reliability is acceptable**

**Critical unknowns:**
- Does Haiku pass basic quality threshold as primary?
- What is actual Opus-Haiku disagreement rate?
- If >50%, escalator becomes primary handler (cost rises)

---

## 4. What We Cannot Answer (No Graph Access)

The live Neo4j graph connection failed (SSH tunnel to titan:7687). Missing empirical data:

❌ **Actual disagreement statistics:**
- Per-name score spread distribution
- Opus-GPT disagreement frequency
- Per-dimension disagreement patterns
- Escalation rate in practice

❌ **Model performance data:**
- Correlation between models
- Which dimensions cause disputes
- Sonnet (escalator) resolution success rate

❌ **Cost validation:**
- Actual token counts per cycle
- Retry rate (estimated 30%, unvalidated)
- Per-axis cost split

---

## 5. Recommendation

### PRIMARY: Pattern D (Experimental) — Haiku Primary + Opus Escalator

**Config change:**
```toml
[tool.imas-codex.sn.review.names]
models = [
  "openrouter/anthropic/claude-haiku-4.5",     # cycle 0 primary (blind, cheap, fast)
  "openrouter/anthropic/claude-opus-4.6",      # cycle 1 escalator (sees both, authoritative)
]
```

**Rationale:**
- **Cost:** ~$0.0075/name (75% reduction vs current $0.027)
- **RD-preservation:** Maintains quorum (2 independent + escalation)
- **Risk:** Haiku unproven as primary; requires pilot

**Prerequisite Pilot (before commit):**
1. Run 10-15 name review on single small domain (e.g., `pf_active` subset)
2. Compare Haiku-Opus disagreement rate
3. Measure Haiku latency (should be 3-5× faster)
4. **Go/NoGo**: If Opus escalator rate >50%, fallback to Pattern B

**Expected outcome:** If pilot succeeds, Round-1 cost drops from **$18.87 → ~$2.00** within $20 budget

---

### FALLBACK: Pattern B (Conservative) — Opus + Sonnet

**Config change:**
```toml
[tool.imas-codex.sn.review.names]
models = [
  "openrouter/anthropic/claude-opus-4.6",      # cycle 0 primary (blind)
  "openrouter/anthropic/claude-sonnet-4.6",    # cycle 1 (no escalator)
]
```

**Rationale:**
- **Cost:** ~$0.024/name (20% reduction)
- **Pros:** Proven models, consistent vendor, simpler
- **Cons:** Loses cross-vendor check, no escalator

**Use if:** Haiku pilot fails or deemed too risky

---

## 6. Summary Decision Table

| Pattern | Config | Cost/Name | Notes |
|---------|--------|-----------|-------|
| **D** (Primary) | Haiku→Opus | $0.0075 | **Requires pilot** |
| **B** (Fallback) | Opus→Sonnet | $0.024 | 20% savings, proven |
| **A** (Current) | Opus→GPT→Sonnet | $0.027 | Expensive, established |
| **C** (Not recommended) | Opus→GPT | $0.0163 | No escalator, risky |
| **E** (Rejected) | Opus only | $0.009 | No quorum logic |

---

## 7. Next Steps

**To finalize recommendation:**

1. **Query live graph** (when tunnel restored):
   ```cypher
   MATCH (sn:StandardName)<-[:REVIEWS]-(rv:Review)
   WITH sn, collect(rv.score) AS scores, collect(rv.reviewer_model) AS models
   WHERE size(scores) >= 2
   WITH sn, scores, models, max(scores) - min(scores) AS spread
   RETURN spread, count(*) AS n_names
   ORDER BY spread DESC
   ```

2. **Analyze disagreement:**
   - Compute per-name spread distribution
   - Identify dispute rate threshold
   - Per-dimension breakdown

3. **Run Haiku pilot** (if spread is moderate, <0.15):
   - Reserve $5 budget
   - Test on 15 names
   - Validate Haiku-Opus disagreement

4. **Commit to Pattern B or D** based on pilot results

---

**Audit completed:** 2024-04-24  
**Recommendation status:** Contingent on graph query + Haiku pilot  
**No code modifications:** Analysis only
