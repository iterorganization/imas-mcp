# Graph Data Quality Audit & Remediation

**Date:** 2026-02-13  
**Status:** Diagnostic Complete  
**Scope:** Unified discovery graph (ITER, TCV, JT-60SA, JET)

## Executive Summary

Comprehensive analysis reveals a **dual-graph conflict** where two independent Neo4j databases exist — one running locally on WSL and one on ITER. The overnight discovery batch (Feb 12) wrote to the **LOCAL graph**, not ITER. The SSH tunnel to ITER was only established this morning (Feb 13, 07:29), after discovery had completed.

The LOCAL graph is the authoritative "clean slate rebuild" from Feb 12 and contains fresh discovery data for all 4 facilities. The ITER graph is older (Feb 10), has more total nodes but different data distributions, and is missing ITER facility paths entirely.

**My analysis in the initial conversation was against the LOCAL graph and is current as of the Feb 12 discovery run.**

---

## 1. Graph Location Investigation

### The Conflict Explained

Both environments connect to `bolt://localhost:7687` by default. On WSL, two processes listen on this port:

| Listener | IP Family | PID | Started |
|----------|-----------|-----|---------|
| Local Neo4j (Java) | IPv4 (`::ffff:127.0.0.1:7687`) | 1118954 | Feb 12, 21:15:29 |
| SSH Tunnel | IPv6 (`[::1]:7687`) | 1432504 | Feb 13, 07:29 |

When Python's neo4j driver connects to `localhost:7687`, it resolves to IPv4 first (standard dual-stack behavior), hitting the **LOCAL Neo4j**. The SSH tunnel is only reachable via explicit IPv6 connection.

### Timeline Reconstruction

```
Feb 10, 18:01  │ ITER graph: DD built (hash: 10e1cd512b3df579)
               │ (ITER has had its own Neo4j running since at least Feb 10)
               │
Feb 12, 20:46  │ LOCAL: Neo4j data files created (clean slate)
Feb 12, 21:15  │ LOCAL: Neo4j process started (PID 1118954)
Feb 12, 22:14  │ LOCAL: DD graph built (hash: clean-slate-rebuild-20260212)
Feb 12, 22:00- │ Discovery batch runs → writes to LOCAL (no tunnel existed)
Feb 12, 23:53  │ LOCAL: Last data modification
               │
Feb 13, 07:14  │ SSH sessions established to ITER
Feb 13, 07:29  │ SSH tunnel to 7687 created (PID 1432504)
Feb 13, 07:23  │ User notices discrepancy between local and ITER results
```

### Graph Comparison

| Metric | LOCAL (WSL) | ITER | Delta |
|--------|-------------|------|-------|
| **Total nodes** | 171,364 | 244,344 | +73k on ITER |
| **Total relationships** | 368,122 | 580,996 | +213k on ITER |
| **dd_build_hash** | `clean-slate-rebuild-20260212` | `10e1cd512b3df579` | Different |
| **dd_graph_built** | 2026-02-12T22:14:54 | 2026-02-10T18:01:59 | LOCAL is 2 days newer |
| **Facility.created_at** | All Feb 12, 20:53 UTC | All NULL | LOCAL has timestamps |

### FacilityPath Distribution

| Facility | LOCAL | ITER | Notes |
|----------|-------|------|-------|
| iter | **12,542** | **0** | ITER graph has NO iter paths |
| tcv | 15,341 | 22,341 | ITER has +7k more |
| jet | 16,294 | 8,686 | LOCAL has +8k more |
| jt-60sa | 2,378 | 587 | LOCAL has +1.8k more |

### WikiPage Distribution

| Facility | LOCAL | ITER | Notes |
|----------|-------|------|-------|
| tcv | 7,778 | 7,777 | Nearly identical |
| jet | 10,546 | 12,038 | ITER has +1.5k more |
| jt-60sa | 5,800 | 7,155 | ITER has +1.4k more |

### FacilitySignal Distribution

| Facility | LOCAL | ITER |
|----------|-------|------|
| tcv | 322 | 339 |

### Root Cause

1. **No coordination mechanism** exists between local and remote Neo4j instances
2. The graph profile system defaults to `bolt://localhost:7687` — same on both WSL and ITER
3. When local Neo4j is running, it shadows any SSH tunnel to ITER
4. The `check_graph_conflict()` guard in `data db start` checks for tunnels before starting Neo4j, but:
   - If Neo4j was started first (before tunnel), no warning fires
   - If tunnel is created after Neo4j, no warning fires
5. Discovery commands have no way to verify which graph they're writing to

### Why Formatting Differs

The `discover status` output differs because:
1. **Content difference**: ITER graph has 0 FacilityPaths for `tcv`, so the paths section is omitted entirely (display logic skips when `total == 0`)
2. **No code difference**: Both environments run the same CLI code
3. **Cost tracking is per-graph**: LOCAL shows $0.61, ITER shows $3.00 — these are independent tallies

---

## 2. Authoritative Graph Decision

**The LOCAL graph is authoritative** based on:
- Clean slate rebuild with proper timestamps (Feb 12)
- Contains all 4 facilities including ITER paths
- Fresher data from most recent discovery run
- Proper `created_at` metadata on Facility nodes

**The ITER graph should be considered stale** and either:
1. Replaced with a dump from LOCAL
2. Or kept as a backup if it contains data not in LOCAL

---

## 3. Data Quality Analysis (LOCAL Graph)

The following analysis is based on the LOCAL graph (171,364 nodes, 368,122 relationships).

### 3.1 Node Inventory by Facility

| Node Type | ITER | TCV | JT-60SA | JET | Total |
|-----------|------|-----|--------|-----|-------|
| FacilityPath | 12,542 | 15,341 | 2,378 | 16,294 | 46,555 |
| WikiPage | 0 | 7,778 | 5,800 | 10,546 | 24,124 |
| WikiArtifact | 0 | 20,127 | 709 | 19,159 | 39,995 |
| WikiChunk | 0 | 7,634 | 212 | 1,826 | 9,672 |
| Image | 0 | 885 | 76 | 547 | 1,508 |
| FacilityUser | 998 | 852 | 256 | 285 | 2,391 |
| FacilitySignal | 0 | 322 | 0 | 0 | 322 |
| TDIFunction | 0 | 21 | 0 | 0 | 21 |
| DataAccess | 0 | 1 | 0 | 0 | 1 |
| SoftwareRepo | — | — | — | — | 168 |
| Person | — | — | — | — | 2,279 |
| IMASPath | — | — | — | — | 44,150 |

**Key observations:**
- **ITER has ZERO wiki infrastructure** — only FacilityPaths and FacilityUsers exist
- **Signals only exist for TCV** (322 signals, 21 TDI functions)
- **No TreeNodes, CodeChunks, SourceFiles** exist anywhere — ingestion pipeline hasn't run
- **Zero MAPS_TO_IMAS relationships** — signal-to-IMAS mapping not established

### 3.2 Paths Discovery Quality

#### Pipeline Completion

| Facility | Discovered | Scanned | Scored | Skipped | Excluded |
|----------|-----------|---------|--------|---------|----------|
| ITER | 3,293 (26%) | 624 (5%) | 918 (7%) | 7,707 (61%) | — |
| TCV | 4,298 (28%) | — | 907 (6%) | 10,136 (66%) | — |
| JT-60SA | 15 (1%) | 535 (22%) | 913 (38%) | 915 (38%) | — |
| JET | 6,639 (41%) | — | 947 (6%) | 8,668 (53%) | 40 |

#### Score Distribution (Scored Paths Only)

| Facility | Count | Avg Score | Median | P25 | P75 | High (≥0.7) |
|----------|-------|-----------|--------|-----|-----|-------------|
| ITER | 918 | 0.744 | 0.850 | 0.570 | 1.000 | 639 (70%) |
| TCV | 907 | 0.563 | 0.700 | 0.200 | 0.850 | 458 (50%) |
| JT-60SA | 913 | 0.713 | 0.850 | 0.500 | 0.950 | 642 (70%) |
| JET | 947 | 0.564 | 0.600 | 0.400 | 0.800 | 343 (36%) |

#### Issues
- **ITER has 624 paths stuck at `scanned`** — never progressed to scoring
- **TCV and JET skip `scanned` state** entirely — state machine inconsistency
- **All 3,685 scored paths have descriptions** (100%) — good quality
- **Zero embeddings on FacilityPath descriptions** — `facility_path_desc_embedding` index is empty

### 3.3 Wiki Discovery Quality

#### Pipeline Completion

| Facility | Scanned | Scored | Ingested | Skipped | Ingestion Rate |
|----------|---------|--------|----------|---------|----------------|
| ITER | — | — | — | — | N/A (no wiki) |
| TCV | 7,081 (91%) | 0 | 278 (3.6%) | 419 (5.4%) | 3.6% |
| JT-60SA | 5,370 (93%) | 19 | 99 (1.7%) | 312 (5.4%) | 1.7% |
| JET | 9,630 (91%) | 22 | 601 (5.7%) | 293 (2.8%) | 5.7% |

#### WikiChunk Quality
- **100% embedding coverage** on all 9,672 chunks
- Content stored in `content` field with rich metadata extraction
- **TCV chunks are richest**: 12.7% mention MDSplus paths, 65.5% mention units
- **JET chunks**: 71.9% mention units, 5.0% tool mentions, 0% MDSplus (uses PPF)
- **JT-60SA sparse**: only 212 chunks, 34.4% with units

#### Issues
- **~92% of wiki pages remain at `scanned`** — never scored/ingested
- **Zero LINKS_TO relationships** between wiki pages

### 3.4 WikiArtifact & URI Linking (CRITICAL)

#### Status Distribution

| Facility | Discovered | Scored | Ingested | Total |
|----------|-----------|--------|----------|-------|
| TCV | 19,425 (96.5%) | 170 (0.8%) | 532 (2.6%) | 20,127 |
| JT-60SA | 216 (30.5%) | 493 (69.5%) | 0 | 709 |
| JET | 18,531 (96.7%) | 320 (1.7%) | 308 (1.6%) | 19,159 |

#### CRITICAL: Artifact Linking is Broken

| Facility | Total Artifacts | Linked to Pages | Orphan Rate |
|----------|----------------|-----------------|-------------|
| TCV | 20,127 | **0** | **100%** |
| JET | 19,159 | 1,258 | **93.4%** |
| JT-60SA | 709 | 689 | **2.8%** |

**TCV has ZERO `WikiPage→HAS_ARTIFACT→WikiArtifact` relationships.** All artifacts are orphaned.

JT-60SA (TWiki-based) has excellent linking (97.2%). TCV and JET (MediaWiki-based) have catastrophic orphan rates. This is likely a bug in MediaWiki artifact discovery — filesystem scan finds artifacts but doesn't link them back to referencing pages.

#### URL Quality (Refetchability)

| Facility | Count | URL Pattern | Auth Required |
|----------|-------|-------------|---------------|
| TCV | 885 | `https://spcwiki.epfl.ch/wiki/images/...` | Wiki auth |
| JET | 547 | `https://wiki.jetdata.eu/.../images/...` | VPN + auth |
| JT-60SA | 76 | Mixed: `nakasvr23.iferc.org` + internal IPs | SSH tunnel |

All images have valid `source_url` — **re-retrieval is possible** with appropriate authentication.

### 3.5 Image Node Analysis

| Facility | Total | Captioned | Data Size | Avg Size |
|----------|-------|-----------|-----------|----------|
| TCV | 885 | 144 (16%) | 26.2 MB | 31 KB |
| JET | 547 | **0** | 37.0 MB | 71 KB |
| JT-60SA | 76 | **0** | 2.2 MB | 31 KB |
| **Total** | **1,508** | **144 (9.5%)** | **65.4 MB** | — |

#### Issues
- **JET and JT-60SA have ZERO VLM captions**
- **65.4 MB of base64 image data** stored in Neo4j
- All images have valid `source_url` — data is redundant

#### Recommendation: Remove `image_data` from Graph
Since all URLs are valid:
- Drop `image_data` property → saves 65 MB (~17% of database)
- Keep `caption`, `ocr_text`, `ocr_*_paths` fields
- Implement on-demand fetch from `source_url`
- Cache JT-60SA images locally (SSH tunnel dependency)

### 3.6 Signals Discovery (TCV Only)

| Status | Count | Has Description | Has Embedding |
|--------|-------|-----------------|---------------|
| checked | 73 | 73 (100%) | 0 |
| enriched | 121 | 121 (100%) | 0 |
| discovered | 50 | 0 | 0 |
| failed | 78 | 78 (100%) | 0 |

#### Issues
- **Zero embeddings** on any signal
- **78 failed signals (24%)** with no `check_error` populated
- **Zero MAPS_TO_IMAS relationships**
- Only 1 DataAccess node (TDI template)

### 3.7 Embedding Coverage (Critical Gap)

| Node Type | With Description | With Embedding | Gap |
|-----------|-----------------|----------------|-----|
| WikiChunk | N/A (content) | 9,672 (100%) | None |
| FacilityPath | 3,685 (7.9%) | **0** | 3,685 |
| FacilitySignal | 272 (84.5%) | **0** | 272 |
| WikiArtifact | 983 (2.5%) | **0** | 983 |
| Image | 144 (9.5%) | **0** | 144 |

**Five vector indexes are empty**: `facility_path_desc_embedding`, `facility_signal_desc_embedding`, `wiki_artifact_desc_embedding`, `image_desc_embedding`, `tree_node_desc_embedding`.

---

## 4. Cross-Facility Inconsistencies

| Issue | Affected | Severity |
|-------|----------|----------|
| ITER has zero wiki data | ITER | **High** |
| TCV artifacts 100% orphaned (no page links) | TCV | **Critical** |
| JET artifacts 93% orphaned | JET | **High** |
| FacilityPath `scanned` state inconsistency | ITER (624 stuck) | **Medium** |
| JET uses `excluded` status not used elsewhere | JET | **Low** |
| JET/JT-60SA have zero VLM captions | JET, JT-60SA | **Medium** |
| No embeddings on descriptions | All | **High** |
| WikiPage inter-page LINKS_TO empty | All | **Low** |
| Person nodes: 0 names, 0 emails | All | **Low** |

---

## 5. Remediation Steps

### Immediate (Graph Conflict Resolution)

1. **Stop SSH tunnel OR stop local Neo4j** — pick ONE authoritative graph
   - To use LOCAL: `ssh -O exit iter` or kill tunnel PID 1432504
   - To use ITER: `uv run imas-codex data db stop` locally
   
2. **If LOCAL is authoritative**: Dump LOCAL and load on ITER
   ```bash
   # On WSL
   uv run imas-codex data dump
   scp graph-dump-*.tar.gz iter:~/Code/imas-codex/
   
   # On ITER
   uv run imas-codex data db stop
   mv ~/.local/share/imas-codex/neo4j ~/.local/share/imas-codex/neo4j-backup-20260210
   uv run imas-codex data load graph-dump-*.tar.gz
   ```

3. **Add tunnel-conflict detection** — warn if both tunnel AND local Neo4j exist on same port

### Data Quality Fixes

4. **Fix MediaWiki artifact linking** — investigate `HAS_ARTIFACT` relationship creation for TCV/JET

5. **Generate description embeddings** — add post-step to score/enrich phases

6. **Run wiki discovery for ITER** — `uv run imas-codex discover wiki iter`

7. **Progress wiki ingestion** — score and ingest the ~22k scanned pages

8. **Run VLM captioning for JET/JT-60SA** — 623 images need captions

9. **Remove `image_data` from graph** — after confirming refetchability

10. **Fix path state machine** — standardize `discovered→scanned→scored` flow

11. **Backfill signal error diagnostics** — populate `check_error` on failed signals

### Verification Queries

```cypher
// TCV artifact linking (should be > 0 after fix)
MATCH (:WikiPage {facility_id:'tcv'})-[:HAS_ARTIFACT]->(wa) RETURN count(wa)

// Path embeddings (should be ≥ 3,685)
MATCH (fp:FacilityPath) WHERE fp.embedding IS NOT NULL RETURN count(fp)

// Image data removal (should be 0 after cleanup)
MATCH (img:Image) WHERE img.image_data IS NOT NULL RETURN count(img)

// ITER wiki (should be > 0)
MATCH (wp:WikiPage {facility_id:'iter'}) RETURN count(wp)

// VLM captions (should be ≥ 1,508)
MATCH (img:Image) WHERE img.caption IS NOT NULL RETURN count(img)
```

---

## 6. Decisions Required

| Decision | Options | Recommendation |
|----------|---------|----------------|
| Canonical graph location | LOCAL vs ITER | **LOCAL** (fresher, more complete) |
| Image data storage | Keep vs remove `image_data` | **Remove** (URLs valid, saves 65MB) |
| Embedding generation timing | Batch vs at-discovery | **At-discovery** (post score/enrich) |
| Tunnel management | Manual vs automatic | **Add conflict detection CLI** |

---

## Appendix: Raw Diagnostic Output

### Port 7687 Listeners (WSL)
```
LISTEN [::1]:7687               ssh (PID 1432504)     # IPv6 - tunnel to ITER
LISTEN [::ffff:127.0.0.1]:7687  java (PID 1118954)    # IPv4 - local Neo4j
```

### Neo4j Data File Timestamps (LOCAL)
```
Birth:  2026-02-12 20:46:33  (clean slate created)
Modify: 2026-02-12 23:53:42  (last write from discovery)
```

### Process Timeline
```
PID 1118954 (Neo4j Java): Started Feb 12, 21:15:29
PID 1432504 (SSH Tunnel): Started Feb 13, 07:29 (today)
```
