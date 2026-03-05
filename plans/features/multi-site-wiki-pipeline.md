# Multi-Site Wiki Pipeline: Pipelined Site Progression

## Problem

The `discover wiki` pipeline processes sites **strictly sequentially**: all workers for site N (SCORE, INGEST, FILE, IMAGE) must finish before any worker begins on site N+1. With 16 JET wiki sites, this creates a severe bottleneck — when 86 pog pages finish scoring and ingesting in minutes, the entire pipeline blocks for hours waiting for a single IMAGE worker to VLM-caption ~1,700 images before the ia/ek/open sites can begin.

```
Current: SITE-1 [SCORE → INGEST → IMAGE ████████████████████] → SITE-2 [SCORE → ...]
                                         ^^^ blocks everything
```

## Root Cause

The blocking chain spans three files:

1. **`cli/discover/wiki.py` — sequential site loop**: Both Rich and non-Rich modes iterate `_site_configs` in a `for` loop, `await`-ing `run_parallel_wiki_discovery()` for each site. Site N+1 cannot start until the `await` returns.

2. **`discovery/wiki/parallel.py` — single `WikiDiscoveryState` per call**: `run_parallel_wiki_discovery()` creates a `WikiDiscoveryState` bound to one site (`base_url`, `auth_type`, `ssh_host`, etc.). Workers read these fields to fetch content. The supervision loop (`run_supervised_loop`) blocks until `state.should_stop()` returns `True`.

3. **`discovery/wiki/state.py` — all-phases-done gate**: `should_stop()` requires **every** phase (scan, score, ingest, artifact_score, docs, **image**) to be idle/done. The `image_phase` waits for `ingest_phase` to be done before declaring no work, and then processes images one-by-one via VLM calls.

## Key Observations

### What already works across sites
- **Graph queries are facility-scoped, not site-scoped**: `claim_pages_for_scoring(facility)`, `has_pending_ingest_work(facility)` — all claim/check functions use `facility_id`, never filter by site URL. A worker pool that spans sites would naturally pick up work from any site.
- **WikiPage nodes carry full URL**: Each page's `url` property contains the site base URL, so content fetching doesn't require global state — the URL itself encodes where to fetch from.
- **Image nodes store `source_url`**: The image worker fetches from `source_url` on each Image node, not from `state.base_url`. It only needs `state.ssh_host` (which is shared across all JET sites) and `state.facility`.
- **`SupervisedWorkerGroup`** is generic — it's just a container of asyncio tasks with status tracking. Nothing prevents it from holding workers for multiple sites.

### What's tightly coupled to a single site
- **Workers read `state.base_url`**: score and ingest workers build fetch URLs relative to `state.base_url`. A page from site A cannot be processed by a worker bound to site B's state.
- **Auth clients are per-site singletons on state**: `state.get_async_wiki_client()`, `state.get_keycloak_client()`, etc. are lazily initialized for one site. Different sites may have different auth configurations (though for JET, all 16 sites share the same keycloak auth).
- **`PipelinePhase` completion is global**: There's one `score_phase`, one `ingest_phase`, etc. per state. No per-site phase tracking.

## Proposed Architecture: Pipelined Site Progression

### Core idea

Decouple **upstream workers** (SCORE) from **downstream workers** (IMAGE) across sites. When upstream work for site N is exhausted, immediately start upstream work on site N+1 — even if downstream workers from site N are still running. Downstream workers are facility-scoped and drain their queues across all sites.

```
Proposed: SITE-1 [SCORE → INGEST] → SITE-2 [SCORE → INGEST] → SITE-3 [SCORE → ...]
          IMAGE  [████████████████████████████████████████████████████████████████]
          (runs continuously, processes images from any site)
```

### Design: Two tiers of workers

**Tier 1 — Site-bound workers** (need `base_url` + auth to fetch raw content):
- `score_worker` — fetches page HTML from a specific wiki site, scores with LLM
- `ingest_worker` — fetches page HTML from a specific wiki site, chunks + embeds
- `docs_score_worker` — fetches artifact preview from a specific wiki site
- `docs_worker` — downloads artifact files from a specific wiki site

These workers need a site-specific `WikiDiscoveryState` because they must authenticate and fetch from a particular wiki URL.

**Tier 2 — Facility-scoped workers** (work from graph data, don't need site context):
- `image_score_worker` — fetches image bytes from `source_url` (stored on node), sends to VLM. Only needs `ssh_host` and `facility`, which are shared across all JET sites.
- `embed_description_worker` — embeds description text already stored on nodes. Fully graph-driven.

### Refactor plan

#### Step 1: Extract facility-scoped workers from the per-site lifecycle

Move IMAGE and EMBED workers out of `run_parallel_wiki_discovery()`. They become long-lived facility-level workers that start once and run across all sites.

- In `run_all_sites_unified()`: create IMAGE + EMBED workers **before** the site loop, attached to a facility-level `SupervisedWorkerGroup`.
- Modify `run_parallel_wiki_discovery()` to accept a flag like `skip_facility_workers=True` so it no longer creates IMAGE/EMBED workers internally.
- The facility-level workers use a lightweight state object (just `facility`, `ssh_host`, `cost_limit`, `stop_requested`) — no `base_url` or auth.

#### Step 2: Change the site loop to not wait for downstream phases

Modify the per-site `should_stop()` to ignore the image phase. The per-site supervision loop should exit when **site-bound phases** (score + ingest + artifact_score + docs) are done, regardless of pending images.

- Add a `should_stop_site_workers()` method that excludes image/embed phases.
- `run_parallel_wiki_discovery()` passes this as the stop condition to `run_supervised_loop()`.
- The site loop advances immediately once site-bound workers for site N are idle.

#### Step 3: Facility-level supervision for long-lived workers

After the site loop completes (all sites' upstream workers are done), await the facility-level worker group separately. This drains remaining images and embeds.

- The facility-level IMAGE worker's `should_stop_image_scoring()` needs adjustment: instead of waiting for `ingest_phase.is_idle_or_done` (which was site-scoped), it checks the graph directly for pending image work.
- A simple approach: image worker stops when `has_pending_image_work(facility)` returns False and it has been idle for N seconds (debounce to handle late-arriving work from the last site).

#### Step 4: Progress display updates

The Rich `WikiProgressDisplay` already queries the graph for global stats (`refresh_graph_state()`). No fundamental change needed — IMAGE progress is already facility-wide. The site banner (`display.advance_site()`) would update as upstream workers move between sites, while IMAGE counts continue accumulating across all sites.

### Edge cases to handle

- **Budget tracking**: The cost limit applies globally. Facility-scoped workers (IMAGE) must share the same budget counter. Consider passing a shared `cost_tracker` (an `asyncio`-safe counter or the existing `image_stats`) rather than per-site `cost_limit`.
- **Stop/shutdown**: `Ctrl+C` must stop both site-bound and facility-level workers. The existing `stop_event` pattern already supports this — both groups check the same event.
- **Different auth per site**: For JET, all sites share keycloak auth + same SSH host. If a facility had sites with different auth, image fetching would still work (images store `source_url` and the SSH host is facility-wide). Site-bound workers already handle per-site auth correctly.
- **Error isolation**: A crash in site N's score worker shouldn't affect site N+1. The existing `supervised_worker()` restart logic handles this per-worker. The site loop's `try/except/continue` already isolates site-level failures.

### What this does NOT change

- Graph schema — no new properties or node types needed
- Worker internals — score/ingest/docs workers are unchanged
- Claim logic — graph queries are already facility-scoped
- Bulk discovery — still runs per-site in the preflight phase
- CLI interface — no new flags needed (the change is transparent)

### Expected impact

For the JET scenario with 16 sites:
- pog finishes SCORE+INGEST in minutes → ia begins SCORE immediately
- IMAGE worker runs continuously in the background, processing images from pog while upstream workers are scoring ek, open, etc.
- Total wall-clock time drops significantly: upstream workers are never idle waiting for VLM calls
- The large `open` site (5,766 pages) can begin scoring as soon as the smaller sites' upstream work completes
