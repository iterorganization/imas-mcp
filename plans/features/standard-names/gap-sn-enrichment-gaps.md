# Standard Names: DD Enrichment Remaining Gaps

> Consolidates unimplemented items from plans 20 (DD-enriched generation)
> and related enrichment work. Core architecture (classifier, enrichment,
> consolidation, COCOS injection) is complete — see `completed/` for
> implemented plans.

## Status: ~85% of plan 20 implemented

### What's done (in completed/)
- Phase 1: 11-rule path classifier (`classifier.py`)
- Phase 2: Cluster enrichment + global grouping (`enrichment.py`)
- Phase 3: DD-enriched prompts with unit, COCOS, cluster siblings (`compose_dd.md`)
- Phase 4: Cross-batch consolidation with 5 conflict checks (`consolidation.py`)
- COCOS injection: `cocos_transformation_type` in schema + code
- Vocab gap mechanism: `SNVocabGap` model, `vocab_gap_detail` property
- Unit safety: units flow from DD, never from LLM

### Remaining gaps

#### Gap 1: `sn regenerate` CLI command
**Priority:** P3 (workaround exists: `sn generate --force --paths`)
**Scope:** Dedicated CLI command for targeted regeneration of specific names.
Current workaround is `sn generate --force --paths <path1> <path2>` which
works but doesn't preserve regeneration context (previous name, score, reason).

#### Gap 2: Benchmark approach profiles
**Priority:** P4
**Scope:** `--profile baseline/dd-enriched` flag for `sn benchmark` to compare
generation quality with and without DD enrichment. Would quantify the value
of the enrichment pipeline.

#### Gap 3: Concept registry
**Priority:** P4
**Scope:** Cross-run concept lookup to detect when a new generation produces
a different name for an already-named concept. Auto-attach partially addresses
this but doesn't track concept drift across regeneration runs.

## Recommendation

These are low-priority refinements. The core DD enrichment architecture is
complete and working. Gap 1 has a functional workaround. Gaps 2-3 are
measurement/tracking improvements that don't affect generation quality.

Consider deferring until after quality parity (plan 23) and standalone
review (plan 25) gaps are closed.

## References
- `completed/19-benchmark-and-lifecycle.md` — benchmark harness (done)
- `completed/22-rename-and-regeneration-context.md` — regeneration context (done)
- `20-consistency-and-prompt-enrichment.md` — original full plan (keep for reference)
