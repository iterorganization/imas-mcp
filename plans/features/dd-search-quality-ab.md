# DD Search Quality — A/B Motivated Improvements

**Status**: Draft (pending RD review)
**Motivation**: A/B comparison vs older `imas-dd` MCP server revealed clear regressions and
missed opportunities in how the current codex server exposes recent schema richness
(LLM-refined `description`, `keywords`, `cocos_label_transformation`, `node_category`).

---

## A/B Summary (highlights)

| Probe | codex top-1 / behavior | imas-dd (old) top-1 / behavior | Verdict |
|-------|-----------------------|--------------------------------|---------|
| `search_standard_names("poloidal flux")` | 3 rich SN results with docs, COCOS, tags | **0 results** (no StandardName catalog in test graph) | codex **wins decisively** |
| `get_ids_summary("magnetics")` | Full physics paragraph + lifecycle breakdown | 2-sentence blurb, no lifecycle | codex **wins** |
| `get_dd_migration_guide(3.42→4.1)` | 1,315 path_added detected | 163 path_added | codex **wins** (8x fidelity) |
| `search_dd_paths("electron temperature")` | Rich physics descriptions w/ symbols | Short one-liners | codex **wins** on content |
| `search_dd_paths("safety factor q profile")` | top-1 `core_instant_changes/.../q` (niche) | top-1 `core_profiles/profiles_1d/q` (canonical) | codex **loses** — tracked in `search-quality-improvements.md` |
| `find_related_dd_paths("equilibrium/.../psi")` | 6 connections, **no unit companions** | 9 connections, 3 unit companions | codex **loses** — bug found below |
| `search_dd_paths` missing filters | No `cocos_transformation_type` filter despite 438 tagged nodes | N/A | codex has under-surfaced filter coverage |

## Gap Analysis

Three material, surgical improvements emerge from the A/B + schema probes:

### Gap 1 — `find_related_dd_paths` regressions

**Evidence (REPL):**

```cypher
MATCH (p:IMASNode {id: 'equilibrium/time_slice/profiles_1d/psi'})-[:HAS_UNIT]->(u:Unit)
      <-[:HAS_UNIT]-(sibling:IMASNode)
WHERE sibling.ids <> p.ids
RETURN count(sibling) AS total,
       sum(CASE WHEN sibling.physics_domain = p.physics_domain THEN 1 ELSE 0 END) AS same_domain
// → total=200+, same_domain=0
```

The `unit_companions` clause filters `sibling.physics_domain = p.physics_domain`, collapsing
any cross-domain unit peers (the common case for coordinate-like quantities such as `psi`).
The older server had no domain restriction and returned useful results.

**Second issue:** all three sub-queries (`cluster_siblings`, `unit_companions`) return
`sibling.documentation`, not `sibling.description`. The refined LLM description exists on
100% of quantity/geometry nodes (12,087/12,087) but is never surfaced in this tool.

### Gap 2 — `cocos_label_transformation` under-surfaced

438 IMASNodes carry a `cocos_label_transformation` (`psi_like`, `ip_like`, `b0_like`,
`q_like`, `tor_angle_like`, …). Today this is:

- **Not filterable** from `search_dd_paths` or `list_dd_paths`.
- **Not returned** as a field in search results.
- **Not traversable** via `find_related_dd_paths` (no "COCOS kin" section).

Adding a `cocos_transformation_type` parameter and a new `find_related_dd_paths` section
unlocks queries like "all psi_like quantities across IDSs" (essential for COCOS-safe code
porting).

### Gap 3 — `search_dd_paths` results hide `keywords`

6,244 unique keywords populate `IMASNode.keywords` (44,963 total refs). Current search
returns description + data_type + units but **never exposes keywords**. Agents consuming
search output cannot see the curated tag vocabulary that the enrichment pipeline already
computed. A single formatter change surfaces it.

---

## Scope

Three surgical code changes, each independently valuable and testable:

### Feature A — Fix `find_related_dd_paths` (minimal)

- **Files:**
  - `imas_codex/tools/graph_search.py` (`GraphPathContextTool`, L1961–2068)
  - `imas_codex/llm/search_formatters.py` (`format_path_context_report`, L1227+)
- **Changes (bug fix + enrichment, no re-ranking):**
  1. Drop the `sibling.physics_domain = p.physics_domain` clause in `unit_companions`.
  2. Exclude `node_category IN ['error','metadata']` from all sub-queries.
  3. Return **both** `sibling.description` and `sibling.documentation`; return
     `sibling.node_category`. Backend prefers `description` when non-empty.
  4. Update `format_path_context_report` to render description when present (fallback to
     documentation) and include a short category tag (e.g. `[coordinate]`).
- **Tests:** regression test — `find_related_dd_paths('equilibrium/.../psi')` returns
  ≥ 3 unit_companions from non-equilibrium domains, and `description` text is non-empty on
  cluster_siblings rows.

### Feature B — COCOS transformation filter (narrowed)

Already partly shipped: `cocos_label_transformation` is fetched in `search_dd_paths` /
`fetch_dd_paths` backends; a dedicated `get_dd_cocos_fields` tool exists. The remaining
gaps are **filtering** in search/list and **display** in the search formatter.

- **Files:** `imas_codex/tools/graph_search.py`, `imas_codex/llm/search_formatters.py`,
  `imas_codex/llm/server.py` (**both** MCP surfaces: simple L1032 + full L2489).
- **Changes:**
  1. Add `cocos_transformation_type: str | None = None` param to:
     - `GraphSearchTool.search_dd_paths` (filter in Cypher WHERE)
     - `GraphListTool.list_dd_paths` (filter in Cypher WHERE)
     - **Both** `server.py` wrappers for parity. **Not** added to `fetch_dd_paths`
       (fetch is exact lookup, filtering is meaningless there).
  2. In `search_formatters.format_search_dd_report` search-result block, add a
     `COCOS: psi_like` line after the keywords line when `hit.cocos_label_transformation`
     is populated. (Cocos is already rendered in `fetch_dd_paths` output — this brings
     search output to parity.)
  3. Add new `cocos_kin` section in `find_related_dd_paths`: peers with same
     `cocos_label_transformation`, different IDS. Excludes error/metadata categories.
- **Tests:**
  - `search_dd_paths(query, cocos_transformation_type='psi_like')` returns only psi_like
    nodes; mismatched query still works (empty result, no error).
  - `list_dd_paths(ids='equilibrium', cocos_transformation_type='psi_like')` returns a
    filtered list.
  - `find_related_dd_paths('equilibrium/.../ip')` now includes a `cocos_kin` section with
    ≥ 5 ip_like peers.
  - Formatter renders `COCOS: psi_like` on a hit that has the transformation.

### ~~Feature C — Surface `keywords` in search output~~ *(dropped: already shipped)*

Rubber-duck review confirmed `keywords` are already fetched in the backend
(`graph_search.py:479`) and rendered by the search formatter
(`search_formatters.py:1159-1160`). No work needed here.

---

## SN Pipeline Hook — downstream follow-on (scoped separately)

The SN compose/enrich pipeline (`imas_codex/standard_names/workers.py`) already consumes
`description` via `_DD_CONTEXT_QUERY` (workers.py L278). Two extensions after Features A–C
land:

1. **PRELINK cocos awareness:** when the source DD path has `cocos_label_transformation`,
   inject it into the compose prompt so the LLM produces names with matching COCOS
   provenance automatically (currently COCOS is post-hoc injected).
2. **Keyword harvesting for vocabulary-gap detection:** if the ISN grammar rejects a
   composed name, retry with keywords as candidate vocabulary hints. Flag genuine new
   terms as `VocabGap` nodes.

Both are additive and do not block Features A–C. Captured as open items in the SN plan
update (`28-sn-greenfield-pipeline.md`).

---

## Non-Goals

- **No changes to search scoring blend** — the `search-quality-improvements.md` plan
  (phases 1A/1B) owns canonical-path vs. accessor de-ranking and is orthogonal.
- **No new MCP tools** beyond surfacing existing rich fields.
- **No schema changes** — all fields used are already populated.

## Risks

- `unit_companions` without domain restriction can return noisy long-tail peers. Mitigated
  by (a) cluster-first ordering (shared cluster > shared unit) and (b) exclude
  error/metadata categories.
- Adding `cocos_transformation_type` to tool signatures is a surface expansion agents must
  discover — documented in tool docstrings, covered by `get_dd_catalog`-style guidance.

## RD Review Checklist

- [ ] Does Gap 1 need a broader redesign or is the surgical fix enough?
- [ ] Should `cocos_kin` be its own MCP tool instead of a section?
- [ ] Keyword display risk: some keywords are weak ("value", "d") — cap 5 OK or need
      stopword filter?
- [ ] Any conflict with `search-quality-improvements.md` phases 1A/1B?
