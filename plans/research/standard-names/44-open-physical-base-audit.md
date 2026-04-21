# Open `physical_base` Audit — rc18 cleanup + decomposition rule

**Date:** 2026-04-20
**ISN version:** v0.7.0rc18
**Scope:** VocabGap hygiene, reviewer decomposition rubric, MCP grammar
filters.

## 1. Problem

Running the Standard Names pipeline against rc18 left ~900 `VocabGap`
nodes in the graph, mostly flagging the open-vocabulary `physical_base`
slot with "missing token" reports. Those reports are noise: the ISN
grammar defines `physical_base` as open — compounds like
`flow_damping_coefficient` are legitimate, not a vocabulary hole. At
the same time we had no mechanism to catch the reverse defect: closed
grammar tokens (like `toroidal` in `toroidal_torque`) absorbed into
`physical_base` instead of being lifted to their dedicated slot.

## 2. Segment openness in rc18

Introspected `imas_standard_names.grammar.constants.SEGMENT_TOKEN_MAP`
on a fresh install. 11 segments; only `physical_base` has an empty
token tuple. All others carry closed enum lists of the sizes below.

| Segment | Tokens | Open? |
|---|---:|---|
| `physical_base` | 0 | **yes** |
| `component` | 18 | no |
| `coordinate` | 18 | no |
| `subject` | 73 | no |
| `device` | 82 | no |
| `geometric_base` | 13 | no |
| `object` | 82 | no |
| `geometry` | 98 | no |
| `position` | 98 | no |
| `region` | 6 | no |
| `process` | 76 | no |

`transformation` is stored separately in `TRANSFORMATION_TOKENS` (37
tokens) and is not part of `SEGMENT_TOKEN_MAP`. The codex heuristic
"empty token tuple ⇒ open" therefore correctly includes only
`physical_base`.

The prompt context also exposes a composer pseudo-segment,
`grammar_ambiguity` (declared in
`imas_codex/llm/prompts/shared/sn/_grammar_reference.md`), used when
the composer flags structural ambiguity rather than a missing token.
We treat it as a pseudo-open segment for gap filtering purposes.

### Recommendation for ISN rc19

Relying on "empty token tuple ⇒ open" is fragile. We suggest ISN
expose one of:

- `SegmentSpec.is_open: bool` on each segment definition, or
- a module-level constant `OPEN_SEGMENTS: frozenset[str] = frozenset({"physical_base"})`.

The codex `imas_codex.standard_names.segments.open_segments()` helper
would then switch to the official accessor.

## 3. Cleanup results

Before:

| Segment | VocabGaps |
|---|---:|
| `physical_base` | 715 |
| `grammar_ambiguity` | 15 |
| `subject` | 40 |
| `transformation` | 39 |
| `position` | 39 |
| `object` | 38 |
| `geometry` | 21 |
| `process` | 9 |
| other closed | 6 |
| **total** | **922** |

After `MATCH (vg:VocabGap) WHERE vg.segment IN {"physical_base",
"grammar_ambiguity"} DETACH DELETE vg`:

- 730 nodes deleted (77% reduction).
- 192 VocabGap nodes remain, all on genuinely closed segments.
  These are real vocabulary holes the reviewer + worker pipeline can
  still act on.

The filter is now enforced at write time (`write_vocab_gaps` +
`_update_sources_after_vocab_gap`) so regressions cannot reintroduce
the noise.

## 4. Decomposition audit (reviewer-side)

The replacement rubric for the "missing physical_base token" signal
is an audit that runs on every candidate StandardName:

> For each candidate, scan the `physical_base` slot for
> underscore-delimited tokens that appear in the closed vocabularies
> (`subjects`, `components`, `coordinates`, `transformations`,
> `positions`, `processes`, `objects`, `geometries`,
> `geometric_bases`, `devices`). Any hit is a candidate decomposition
> defect unless the compound is a well-known lexicalised atomic
> quantity.

Examples observed in the current corpus that the rule targets:

| Name | Hits | Suggested decomposition |
|---|---|---|
| `toroidal_torque` | `toroidal (component)` | `component=toroidal, physical_base=torque` |
| `volume_averaged_electron_temperature` | `volume_averaged (transformation), electron (subject)` | `transformation=volume_averaged, subject=electron, physical_base=temperature` |
| `flux_surface_cross_sectional_area` | `flux_surface (position)` | `position=flux_surface, physical_base=cross_sectional_area` |

Allowed lexicalised atoms (do not flag): `poloidal_flux`,
`minor_radius`, `cross_sectional_area`, `safety_factor`.

Penalty is folded into the existing **Grammar** scoring dimension
(cap at −8 per candidate), so the rubric totals stay invariant
(name-only: 4×20=80; full: 6×20=120). A new schema dimension was
deliberately avoided.

Helper utility:
`imas_codex.standard_names.decomposition.find_absorbed_closed_tokens(haystack, closed_vocab)`
returns sorted `(token, segment)` tuples for a haystack string, using
word-boundary substring matching. Covered by 14 unit tests.

## 5. New MCP surface

### `search_standard_names`

Adds 11 optional kwargs for grammar-slot post-filtering:

```python
grammar_physical_base, grammar_subject, grammar_component,
grammar_coordinate, grammar_transformation, grammar_position,
grammar_process, grammar_object, grammar_geometry,
grammar_geometric_base, grammar_device
```

Each filter is a case-insensitive exact match against the persisted
`sn.grammar_<segment>` property. Filters are conjunctive — all
provided constraints must hold.

### `list_grammar_vocabulary(segment)`

New tool. Aggregates `sn.grammar_<segment>` across all StandardName
nodes and returns a markdown table of distinct tokens with their
usage counts, ordered by count desc. Segment argument is validated
against an allowlist so Cypher interpolation is safe.

Example output for `component` in the current graph:

| Token | Count |
|-------|------:|
| radial | 37 |
| parallel | 33 |
| toroidal | 32 |
| poloidal | 25 |
| vertical | 18 |
| diamagnetic | 3 |

### Latent bug fixed in the same commit

The existing Cypher for `search_standard_names` and
`fetch_standard_names` read `sn.physical_base`, `sn.subject`,
`sn.component`, etc. Those properties are never set — the schema
persists them under the `grammar_` prefix. The "Grammar:" line in
the search report was silently empty. Fixed to read
`sn.grammar_physical_base`, `sn.grammar_subject`, etc.

## 6. Test coverage added

- `tests/standard_names/test_vocab_gaps.py::TestOpenSegmentFilter` — 5
  regression tests for `filter_closed_segment_gaps` and the
  `write_vocab_gaps` hook.
- `tests/standard_names/test_decomposition_audit.py` — 14 tests over
  synthetic vocab, formatter, and real ISN vocabulary fixtures.
- `tests/standard_names/test_sn_tools_grammar_filters.py` — 9 tests
  for `_search_standard_names` post-filtering and
  `_list_grammar_vocabulary`.

## 7. Follow-ups

1. **ISN rc19 request:** expose `SegmentSpec.is_open` or
   `OPEN_SEGMENTS` constant (see §2). Open a ticket upstream once
   rc18 is superseded.
2. **Worker prompt hint:** downstream consider surfacing
   `find_absorbed_closed_tokens` output as a hint in the composer
   prompt so compounds are lifted before reaching the reviewer. Not
   done in this change — keep change surface minimal.
3. **Embedding refresh:** the physical_base filter only works after
   StandardName nodes are reparsed. Existing 900 nodes already have
   `grammar_physical_base` set, so no backfill required.
