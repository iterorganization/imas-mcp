# CLI Option Matrix — Audit, Canonical Vocabulary, Consolidation

**Status:** research artifact for the CLI harmonization work (feature plan `33-cli-harmonization`).

**Scope:** audit every click option across `imas_codex/cli/**/*.py` and propose a
canonical vocabulary plus a consolidation plan for `sn` subcommands. Implementation
is handled in a separate patch series — this document is the source of truth.

> **Coordination note.** Another agent (`sn-extraction-batching-prompt`, opus-4.7) is
> concurrently modifying `sn generate`. This document captures `sn generate`'s current
> option surface for reference only; the harmonization of `sn generate` is explicitly
> **deferred** to a follow-up commit once that agent merges.

---

## Part 1 — Per-command option inventory

Each table lists: long flag, short flag, help summary, default, and type. The "**clash**"
column flags inconsistencies with the proposed canonical vocabulary (§3). Commented-in
source line numbers are included for traceability.

### 1.1  `sn` subcommands  (`imas_codex/cli/sn.py`)

#### `sn generate`  (lines 35–571)  — **frozen for this patch**
| long | short | meaning | default | type |
| --- | --- | --- | --- | --- |
| `--source` | — | `dd` or `signals` | `dd` | Choice |
| `--physics-domain` / `--domain` | — | physics-domain scope | `None` | str |
| `--facility` | — | facility id | `None` | str |
| `--cost-limit` | `-c` | USD cap | `5.0` | float |
| `--dry-run` | — | preview only | off | flag |
| `--force` | — | re-generate already-named | off | flag |
| `--revalidate` | — | re-run ISN validation | off | flag |
| `--limit` | — | max DD paths | `None` | int |
| `--compose-model` | — | override LLM | `None` | str |
| `--verbose` | `-v` | verbose | off | flag |
| `--quiet` | `-q` | suppress output | off | flag |
| `--paths` | — | explicit DD paths (multi) | `()` | multiple str |
| `--reset-to` | — | `extracted`\|`drafted` | `None` | Choice |
| `--from-model` | — | substring-match prior model | `None` | str |
| `--reset-only` | — | cleanup then exit | off | flag |
| `--since` | — | ISO `generated_at` lower bound | `None` | str |
| `--before` | — | ISO `generated_at` upper bound | `None` | str |
| `--below-score` | — | `reviewer_score <` threshold | `None` | float |
| `--tier` | — | comma list of review tiers | `None` | str |
| `--retry-quarantined` | — | target `validation_status=quarantined` | off | flag |
| `--retry-skipped` | — | include `status=skipped` sources | off | flag |
| `--retry-vocab-gap` | — | target vocab-gap names | off | flag |
| `--regen-only` | — | feed prior reviewer comments | off | flag |

**Observations** — `sn generate` uses `--physics-domain` (with `--domain` alias). Every
other `sn` subcommand uses `--ids` or `--domain` bare. `-c` is cost-limit here, in line
with the proposed vocabulary.

#### `sn benchmark`  (lines 574–739)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--source` | — | `dd` | `dd` | Choice | |
| `--ids` | — | IDS filter | `None` | str | ⚠ vs `--physics-domain` canon |
| `--domain` | — | physics-domain filter | `None` | str | ⚠ rename → `--physics-domain` |
| `--facility` | — | reserved | `None` | str | |
| `--models` | — | comma list | `None` | str | |
| `--max-candidates` | — | cap | `50` | int | ⚠ vs `--limit` canon |
| `--runs` | — | runs/model | `1` | int | |
| `--temperature` | — | LLM temp | `0.0` | float | |
| `--output` | — | JSON path | `None` | path | |
| `--verbose` | `-v` | verbose | off | flag | |
| `--reviewer-model` | — | judge | `None` | str | |

#### `sn status`  (lines 742–839)  — no options.

#### `sn gaps`  (lines 842–948)
| long | short | meaning | default | type |
| --- | --- | --- | --- | --- |
| `--segment` | — | grammar segment | `None` | str |
| `--export` | — | `table`\|`yaml` | `table` | Choice |

#### `sn publish`  (lines 950–1156)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--ids` | — | IDS filter | `None` | str | ⚠ rename → `--physics-domain` |
| `--domain` | — | domain filter (tags) | `None` | str | ⚠ rename → `--physics-domain` |
| `--output-dir` | — | YAML dir | `sn_catalog_output` | path | |
| `--group-by` | — | `ids`\|`domain`\|`confidence` | `ids` | Choice | |
| `--confidence-min` | — | floor | `0.0` | float | |
| `--catalog-dir` | — | existing catalog | `None` | path | |
| `--create-pr` | — | push PR | off | flag | |
| `--catalog-repo` | — | upstream repo | hard-coded | str | |
| `--dry-run` | — | preview | off | flag | |
| `--verbose` | `-v` | verbose | off | flag | |

#### `sn import`  (lines 1158–1317)
| long | short | meaning | default | type |
| --- | --- | --- | --- | --- |
| `--catalog-dir` | — | required | — | path |
| `--tags` | — | comma filter | `None` | str |
| `--dry-run` | — | preview | off | flag |
| `--check` | — | compare catalog vs graph | off | flag |
| `--verbose` | `-v` | verbose | off | flag |

#### `sn clear`  (lines 1320–1464)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--status` | — | `review_status` filter | `None` | str | |
| `--all` | — | everything | off | flag | |
| `--source` | — | `dd`\|`signals` | `None` | Choice | |
| `--ids` | — | IDS filter | `None` | str | ⚠ rename → `--physics-domain` |
| `--include-accepted` | — | also accepted | off | flag | |
| `--dry-run` | — | preview | off | flag | |
| `--force` | `-f` | skip prompt | off | flag | ⚠ `-f` should be `--facility` canon |
| `--include-sources` | — | drop sources too | off | flag | |

#### `sn reconcile`  (lines 1466–1488)
| long | short | meaning | default | type |
| --- | --- | --- | --- | --- |
| `--source-type` | — | `dd`\|`signals` | `dd` | Choice |

#### `sn seed`  (lines 1491–1622)
| long | short | meaning | default | type |
| --- | --- | --- | --- | --- |
| `--source` | — | `isn`\|`west`\|`all` | `all` | Choice |
| `--west-dir` | — | path | `None` | path |
| `--dry-run` | — | preview | off | flag |
| `--verbose` | `-v` | verbose | off | flag |

#### `sn resolve-links`  (lines 1625–1714)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--limit` | — | batch size/round | `50` | int | |
| `--rounds` | — | n rounds | `3` | int | |
| `--dry-run` | — | preview | off | flag | |
| `--verbose` | `-v` | verbose | off | flag | |

#### `sn review`  (lines 1717–2014)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--ids` | — | IDS scope | `None` | str | ⚠ rename → `--physics-domain` |
| `--domain` | — | physics-domain scope | `None` | str | ⚠ rename → `--physics-domain` |
| `--status` | — | review_status filter | `drafted` | str | |
| `--unreviewed` | — | only new/stale | off | flag | |
| `--force` | — | re-review | off | flag | |
| `--model` | — | override review model | `None` | str | |
| `--batch-size` | — | per-batch cap | `15` | int | |
| `--neighborhood` | — | similar-name ctx | `10` | int | |
| `--cost-limit` | `-c` | USD | `5.0` | float | |
| `--dry-run` | — | preview | off | flag | |
| `--skip-audit` | — | skip Layer 1 | off | flag | |
| `--concurrency` | — | parallel batches | `2` | int | |

**Missing:** `--name-only` (D3 deliverable) — scores only grammar / semantic / convention /
completeness; skips documentation + compliance.

#### `sn enrich`  (lines 2017–2233)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--domain` | — | physics-domain (multi) | `None` | multi str | ⚠ rename → `--physics-domain` |
| `--status` | — | review_status filter(s) | `named` | str | |
| `--cost-limit` | `-c` | USD | `2.0` | float | |
| `--limit` | — | cap names | `None` | int | |
| `--batch-size` | — | LLM batch | `8` | int | |
| `--dry-run` | — | preview | off | flag | |
| `--force` | — | re-enrich | off | flag | |
| `--model` | — | override | `None` | str | |
| `--verbose` | `-v` | verbose | off | flag | |
| `--quiet` | `-q` | quiet | off | flag | |

#### `sn rotate`  (lines 2236–end)
| long | short | meaning | default | type | clash |
| --- | --- | --- | --- | --- | --- |
| `--domain` | — | physics-domain, required | — | str | ⚠ rename → `--physics-domain` |
| `--cost-limit` | `-c` | USD total | `5.0` | float | |
| `--limit` | — | per-generate cap | `None` | int | |
| `--dry-run` | — | plan only | off | flag | |
| `--fail-fast` | — | abort on phase err | off | flag | |
| `--skip-generate` | — | | off | flag | |
| `--skip-enrich` | — | | off | flag | |
| `--skip-review` | — | | off | flag | |
| `--skip-regen` | — | | off | flag | |
| `--concurrency` | — | workers/phase | `2` | int | |

---

### 1.2  `discover` subcommands  (`imas_codex/cli/discover/*.py`)

`discover` takes `FACILITY` as a positional argument consistently. No `--facility`
option exists in this group.

#### `discover paths`
| long | short | meaning | default | clash |
| --- | --- | --- | --- | --- |
| `--root` | `-r` | restrict root paths (multi) | — | |
| `--cost-limit` | `-c` | USD | `10.0` | |
| `--path-limit` | `-l` | terminal-state cap | `None` | ⚠ vs canonical `--limit` |
| `--focus` | `-f` | NL focus | `None` | ⚠ `-f` reserved for `--facility` canon |
| `--threshold` | `-t` | score floor | `None` | |
| `--scan-workers` | — | n scan workers | `1` | |
| `--triage-workers` | — | n triage workers | `2` | |
| `--scan-only` | — | skip LLM | off | |
| `--triage-only` | — | skip SSH | off | |
| `--add-roots` | — | add roots from config | off | |
| `--enrich-threshold` | — | auto-enrich floor | `None` | |
| `--reset-to` | — | Choice via `reset_to_option` | `None` | |
| `--reset-scored` | — | hidden deprecated | off | |
| `--time` | — | minute cap | `None` | |
| `--triage-batch-size` | — | paths/LLM call | `None` | |

#### `discover signals`
| long | short | meaning | default | clash |
| --- | --- | --- | --- | --- |
| `--cost-limit` | `-c` | USD | `5.0` | |
| `--signal-limit` | `-n` | cap signals | `None` | ⚠ vs canonical `--limit` |
| `--scanners` | `-s` | comma list | `None` | |
| `--focus` | `-f` | pattern focus | `None` | ⚠ `-f` |
| `--scan-only` | — | | off | |
| `--enrich-only` | — | | off | |
| `--enrich-workers` | — | workers | `2` | |
| `--check-workers` | — | workers | `4` | |
| `--time` | — | minutes | `None` | |
| `--reference-shot` | — | override shot | `None` | |
| `--rescan` | — | re-discover | off | |
| `--reset-to` | — | Choice | `None` | |
| `--reenrich` | — | hidden deprecated | off | |

#### `discover code`
| long | short | meaning | default | clash |
| --- | --- | --- | --- | --- |
| `--min-score` | — | FP min | `None` | |
| `--max-paths` | — | scan cap | `100` | ⚠ vs canonical `--limit` |
| `--focus` | `-f` | | `None` | ⚠ `-f` |
| `--cost-limit` | `-c` | | `5.0` | |
| `--scan-workers` | — | | `2` | |
| `--triage-workers` | — | | `2` | |
| `--enrich-workers` | — | | `2` | |
| `--score-workers` | — | | `1` | |
| `--code-workers` | — | | `1` | |
| `--scan-only` | — | | off | |
| `--score-only` | — | | off | |
| `--time` | — | minutes | `None` | |
| `--verbose` | `-v` | | off | |
| `--rescan` | — | | off | |
| `--triage-batch-size` | — | | `None` | |
| `--reset-to` | — | Choice | `None` | |

#### `discover wiki`
| long | short | meaning | default | clash |
| --- | --- | --- | --- | --- |
| `--source` | `-s` | wiki URL | — | |
| `--cost-limit` | `-c` | | `10.0` | |
| `--max-pages` | `-n` | cap | `None` | ⚠ vs canonical `--limit` |
| `--max-depth` | — | link depth | `None` | |
| `--focus` | `-f` | | — | ⚠ `-f` |
| `--scan-only` | — | | off | |
| `--score-only` | — | | off | |
| `--verbose` | `-v` | | off | |
| `--rescan` | — | | off | |
| `--score-workers` | — | | `2` | |
| `--ingest-workers` | — | | `4` | |
| `--rescan-documents` | — | | off | |
| `--time` | — | minutes | `None` | |
| `--store-images` | — | | off | |
| `--min-score` | — | composite floor | `0.5` | |
| `--reset-to` | — | Choice | `None` | |

#### `discover documents`
| long | short | meaning | default | clash |
| --- | --- | --- | --- | --- |
| `--min-score` | — | FP floor | `0.5` | |
| `--max-paths` | — | scan cap | `50` | ⚠ vs canonical `--limit` |
| `--cost-limit` | `-c` | USD | `2.0` | |
| `--workers` | — | fetchers | `2` | |
| `--vlm-workers` | — | VLM workers | `1` | |
| `--store-bytes` | — | keep image bytes | off | |
| `--scan-only` | — | | off | |
| `--focus` | `-f` | | `None` | ⚠ `-f` |
| `--time` | — | minutes | `None` | |
| `--verbose` | `-v` | | off | |
| `--reset-to` | — | Choice | `None` | |

#### `discover status` / `discover clear` / `discover seed` / `discover inspect`
| cmd | options |
| --- | --- |
| `status` | `FACILITY` arg · `--json` · `--domain/-d` |
| `clear` | `FACILITY` arg · `--force/-f` · `--domain/-d` |
| `seed` | `FACILITY` arg · `--path/-p` (multi) |
| `inspect` | `FACILITY` arg · `--scanned N` · `--scored N` · `--json` |

`--domain` here means **discovery domain** (`paths`/`wiki`/`signals`/`files`) — this is
a *different* meaning from `--domain` inside `sn` (physics domain). The CLI group
(`discover` vs `sn`) disambiguates, but it adds to the vocabulary churn.

---

### 1.3  Other top-level groups

#### `graph` — split into `server.py`, `data.py`, `registry.py`, `sync.py`
Commands: `start`, `stop`, `status`, `shell`, `profiles`, `secure`, `export`, `load`,
`list`, `switch`, `init`, `clear`, `push`, `fetch`, `pull`, `tags`, `prune`, `facility
{list|add|remove}`, `repair cocos-labels`, `sync-isn-grammar`.

Observed conventions that match the canon:
- `--force` = skip confirmation (universally)
- `--dry-run` = preview (`graph push`, `graph prune`, `sync-isn-grammar`)

Clashes to resolve:
| flag | command | current meaning | note |
| --- | --- | --- | --- |
| `--facility` | `graph push`, `graph tags`, `graph prune` | filter to facility variant | fine |
| `-v` / `--version` | `graph fetch`, `graph pull` | **version tag** | collides with our canonical `-v` = verbose, but only inside `graph` where verbose-semantics aren't wired. Keep as-is. |
| `--keep` / `--dev-only` | `graph prune` | | fine |

#### `embed`  (`imas_codex/cli/embed.py`)
Commands: `start`, `status`, `service`, `stop`, `restart`, `logs`.

Flag map:
- `embed start`: `--foreground/-f`, `--no-slurm`, `--gpus/-g`, `--workers/-w`,
  `--host`, `--port`, `--log-level`, `--gpu`, `--idle-timeout`, `--deploy-label`.
- `embed status`: `--url`, `--local`.
- `embed service`: `--gpu`, `--deploy-label` + `action` positional.
- `embed restart`: `--gpus`, `--workers`, `--no-slurm`.
- `embed logs`: `--follow/-f`, `--lines`.

**Clash:** `-f` = `--foreground` (start) and `-f` = `--follow` (logs). Neither matches
the canonical `-f` = `--facility`. However `embed` has no facility concept, so the `-f`
slot is locally meaningful. **Decision:** keep as-is; the canonical `-f` = facility
applies only where a facility option exists.

#### `llm`  (`imas_codex/cli/llm_cli.py`)
Commands: `start`, `status`, `service`, `stop`, `restart`, `logs`, plus `keys`
(subgroup) and `teams` (subgroup), `spend`, `setup`, `security audit/harden`.

Flag map (options only):
- `llm start`: `--host`, `--port`, `--db-port`, `--foreground/-f`, `--env-file`.
- `llm status`: `--url`, `--deep`.
- `llm service`: `action` pos + `--port`.
- `llm stop`: `--with-db`.
- `llm logs`: `--follow/-f`, `--lines`.
- `llm keys create`: `--team`, `--alias` (req), `--budget`, `--duration`.
- `llm keys rotate`: `--key`.
- `llm teams create`: `--alias`, `--budget`, `--duration`.
- `llm spend`: `--team`.
- `llm setup`: `--dry-run`, `--force`.

**Clash:** `--duration` (teams/keys) vs nowhere else. OK. `-f` in start/logs same as
`embed` — local meaning.

#### `tunnel`  (`imas_codex/cli/tunnel.py`)
`start`, `stop`, `status`, `service`, hidden `service-run`.
Flags: `--neo4j`, `--embed`, `--llm`, `--timeout`. `--all` on `stop`. No canonical
conflicts.

#### `config`  (`imas_codex/cli/config_cli.py`)
`config private {push|pull|status}`, `config secrets {push|pull|status}`,
`config local-hosts`.
Flags use `--force`, `--dry-run` consistent with canon. `--gist-id`, `--url`,
`--no-backup` are local.

#### `serve`  (`imas_codex/cli/serve.py`)
Single command: `--transport`, `--host`, `--port`, `--tools-file`,
`--mcp-tool-allowlist`, `--default-facility`, `--reload`. No conflicts.

#### `release`  (`imas_codex/cli/release.py`)
`release` single command + argument: `--dry-run`, `--force`, `--skip-ci`, `--push`,
`--no-push`, `--message`, `--remote`, `--strict-clean`, `--allow-ci-pending`.
`--dry-run` and `--force` match canon.

#### `tools`  (`imas_codex/cli/tools.py`)
`tools list`, `tools status HOST`, `tools install`, `tools update`. Flags are
local (`--python-only`, `--tools-only`, `--slurm`, `--container`, `--force`).

#### `imas` / `imas dd`  (`imas_codex/cli/imas_dd.py`)
Key flags: `--verbose/-v`, `--quiet/-q`, `--force/-f`, `--version/-v` (collision
inside imas-dd — `v` means verbose on `build`, version on `search`/`version`).
`--ids` (filter). `--epoch/-e`. `--limit/-n` on `search` (local `n` for count).

**Clash:** `-n` = `--limit` in `imas search` but `-n` = `--max-pages`/`--signal-limit`
in `discover`. The canonical choice (below) is `-n` = `--limit`, which matches `imas
search` and the intent of `--max-pages` / `--signal-limit`.

---

## Part 2 — Inconsistencies in one glance

| concept | seen names | seen shorts | collision? |
| --- | --- | --- | --- |
| USD cost cap | `--cost-limit` (most) | `-c` (most) | clean — keep. |
| item cap | `--limit`, `--path-limit`, `--signal-limit`, `--max-pages`, `--max-paths`, `--max-candidates` | `-l`, `-n`, — | large drift |
| batch size | `--batch-size`, `--triage-batch-size`, `--max-candidates` | — | OK (batch-size is stable where present) |
| facility | positional `FACILITY` (discover, tools, most imas), `--facility` (sn, graph), `-f` on discover = focus | `-f` conflicts | |
| physics-domain scope (sn) | `--physics-domain` (generate only), `--domain` (review/enrich/rotate/benchmark/publish), `--ids` (benchmark/publish/clear/review) | — | large drift |
| dry-run | `--dry-run` everywhere | — | clean |
| force | `--force` (most), `--force/-f` (sn clear, discover clear, imas clear, release, config pull) | `-f` contested | |
| model override | `--model` (review, enrich, benchmark), `--compose-model` (generate), `--reviewer-model` (benchmark), `--from-model` (generate regen select) | — | mostly OK once we settle compose-model/reviewer-model as role-specific |
| verbose | `-v/--verbose` uniformly on `sn`, `discover`, `imas` | — | clean |
| quiet | `-q/--quiet` | — | clean (only a couple of places) |
| status filter | `--status` (review, enrich, clear), `--review-status` (none), `--status-filter` (internal var) | — | unify on `--status` |
| output format | `--export table|yaml` (gaps), `--json` (discover inspect/status, tools status), `--output PATH` (benchmark) | — | messy but scope-limited |

---

## Part 3 — Canonical option vocabulary (proposal)

For each canonical concept we pick **one** long name and optionally **one** short flag.
Justification cites prevailing usage and readability.

| concept | canonical long | canonical short | rationale |
| --- | --- | --- | --- |
| LLM spend cap (USD) | `--cost-limit` | `-c` | already used in every discover/sn pipeline command. |
| upper bound on items processed | `--limit` | `-n` | `--limit` is the simplest; `-n` (for "count") is used in `imas search` and partially in discover. We consolidate `--path-limit`, `--signal-limit`, `--max-pages`, `--max-paths`, `--max-candidates` onto `--limit` (plus pipeline-specific `--max-candidates` in `sn benchmark` keeps its name because "candidates" has a distinct domain meaning). `-n` sits better than `-l` because `-l` is widely associated with `--log` or `--location`. |
| LLM batch size (items/call) | `--batch-size` | — | already used in `sn review`, `sn enrich`; migrate `--triage-batch-size` → stays distinct (it's a per-phase tuning knob in discover — keep prefixed) but `--max-candidates` in benchmark stays as semantic item cap. |
| facility identifier | `--facility` / positional `FACILITY` | `-f` | positional on discover/tools is **kept** (strong prevailing pattern); flag `--facility` with short `-f` applies where no positional exists (sn, graph push/tags/prune). Free `-f` everywhere by renaming `--focus -f` → `--focus` (no short) and `--foreground -f` in embed/llm → keep (no facility concept; local `-f` allowed). `-f` for `--force` in `sn clear`/`discover clear`/`imas clear` → drop the short (keep long). |
| physics-domain scope (sn) | `--physics-domain` | — | user directive; also the only flag name that reads naturally at `sn generate --physics-domain magnetics`. `--domain` and `--ids` become aliases, preserved through the current release. |
| source type | `--source` | — | stable across `generate`, `clear`, `benchmark`, `seed` (always `dd`/`signals` or similar). `sn reconcile` uses `--source-type` → rename to `--source`. |
| preview without side-effects | `--dry-run` | — | already uniform; no change. |
| bypass safety checks or re-do | `--force` | — | drop `-f` short (conflicts with facility/focus/foreground). |
| LLM model override (role-agnostic) | `--model` | — | keep. Role-specific overrides (`--compose-model`, `--reviewer-model`) remain where two models must coexist (generate, benchmark). |
| output format | `--format` (Choice: `table`, `json`, `yaml`) | — | replace `--export table|yaml` on `sn gaps`, `--json` flags on `discover status`, `tools status`, `discover inspect` with a unified `--format` option (alias `--json` → `--format json`). |
| review-status / state filter | `--status` | — | keep the name; values differ by command but the flag is stable. |
| verbose logging | `--verbose` | `-v` | universal. |
| quiet logging | `--quiet` | `-q` | universal where present. |

### 3.1  Freed short flags and what they become

| short | old uses | canonical meaning |
| --- | --- | --- |
| `-c` | cost-limit (most), nothing else | **cost-limit** — no change |
| `-f` | focus (discover), force (sn/imas clear), foreground (embed/llm) | **facility** wherever a facility selector exists; otherwise unbound (no alias). `--focus`, `--force`, `--foreground` keep their long forms with no short. Exception: in `embed`/`llm` which have no facility concept, `--foreground -f` and `--follow -f` keep the short because there is no conflict. |
| `-n` | signal-limit, max-pages, limit (imas search) | **limit** |
| `-v` | verbose (most), version (graph fetch/pull/tags) | **verbose** — graph commands that use `-v` for `--version` are acceptable because there is no `--verbose` flag alongside in those specific commands. Flagged for future cleanup but no rename in this PR. |

---

## Part 4 — `sn` subcommand consolidation proposal

Target: ≤ 8 subcommands. Currently 14 (`generate`, `enrich`, `review`, `clear`,
`publish`, `import`, `status`, `gaps`, `reset`-via-`generate`, `reconcile`,
`resolve-links`, `rotate`, `seed`, `benchmark`). Core set we keep (per user
directive): `generate`, `enrich`, `review`, `clear`, `publish`, `status`, `gaps`,
`benchmark` — that is already 8.

### 4.1  `sn rotate` → fold into `sn generate`?

**Proposal: keep `rotate` separate.** Rationale:

- `rotate` chains four distinct pipelines (`generate → enrich → review → regen`). It
  allocates budget across phases (40/20/20/20) and needs its own state (`RotationConfig`
  with a rotation_id for provenance stamping). Wrapping `sn generate` with
  `--also-enrich --also-review --also-regen` would need `generate` to import the
  enrich + review entry points and track per-phase budgets — in effect, reproducing
  `rotate` inside `generate`.
- Different required args: `rotate` **requires** `--domain`; `generate` does not.
- Tests: `test_sn_rotate_cli.py` + `standard_names/test_rotation_provenance.py` assert
  rotation-specific provenance (RotatedFrom edges). Moving rotation into generate would
  muddy provenance.

**Conclusion:** rotate stays. It's one of the 8 slots. **Revised target set
becomes: `generate`, `enrich`, `review`, `clear`, `publish`, `status`, `gaps`,
`rotate`** (8) — drop `benchmark` from the "core" list? No — user directive pins
benchmark as core.

With rotate staying, we need to drop: `reset` (already done per commit history),
`reconcile`, `resolve-links`, `seed`, `import`, `benchmark` — six candidates.
User forbids dropping `benchmark` → five real candidates.

### 4.2  `sn reset` → already folded into `sn generate --reset-to` ✅

Confirmed: commit `2f996a82 refactor(sn): consolidate reset into generate` removed
the `reset` subcommand, and `sn generate --reset-to {extracted|drafted}` plus `sn
clear --status ...` cover the needs. **No further action.**

### 4.3  `sn resolve-links` — merge into `sn generate`'s post-persist step?

`sn resolve-links` runs an iterative rounds loop to promote `dd:` links to `name:`
links once target names appear. It is stateful (each round may promote names the
previous round couldn't) but the per-round work is small.

Two viable routes:
1. **Internal post-persist step**: after `generate` persists new names, trigger a
   bounded round of `resolve_links_batch()` automatically. Still leave `sn
   resolve-links` available as a manual catch-up for operators (rename to
   `sn links` to shorten, or keep under a `sn tools` subgroup). The iterative
   `--rounds` knob matters for bulk imports (e.g., after `sn import`), so the
   command can't disappear entirely.
2. **Demote to hidden / tools subcommand**: make `sn resolve-links` a
   `hidden=True` command (still callable from scripts) and always invoke one round
   at the end of `generate`, `enrich`, `import`, `seed`.

**Recommendation:** route 2. Add internal `resolve_links_after_persist()` hook fired
from `sn generate` / `sn enrich` / `sn import` / `sn seed` persist phases, and mark
`sn resolve-links` hidden (still works for CI/manual top-ups). This drops one
visible subcommand.

### 4.4  `sn seed` + `sn import` → merge?

Both pull external YAML data into the graph. Differences:

- `import`: **canonical catalog** (`imas-standard-names-catalog` upstream). Sets
  `review_status='accepted'`. Has `--check` mode for sync diffing. Required arg
  `--catalog-dir`.
- `seed`: **calibration anchors** (42 ISN reference examples bundled with
  `imas-standard-names`, plus WEST catalog with ~305 entries set to `drafted`). Does
  its own physics_domain / tag cleanup before ISN validation. Optional arg
  `--west-dir`.

They share 80% of machinery (YAML → validated → graph MERGE) but differ in
**provenance / review_status defaults** and **cleanup passes**. A natural merge is:

```
sn import [--catalog-dir PATH] [--source {catalog|isn|west|all}] [--tags ...]
          [--check] [--dry-run]
```

where `--source catalog` (default) requires `--catalog-dir` and `--source isn|west`
loads from bundled packages. `--check` stays meaningful only for `--source catalog`.

**Recommendation:** merge `seed` into `import` under `--source` flag. Keep backwards
compatibility by making `sn seed` an alias (hidden) that forwards to `sn import
--source {isn|west|all}`. This drops one visible subcommand.

### 4.5  `sn reconcile` → internal post-persist step?

`sn reconcile` re-links `StandardNameSource` nodes to upstream entities after a DD
or signals rebuild. It is already invoked:
- Manually by operators after an `imas dd build`.
- Not automatically by any pipeline.

Two options:
1. Auto-run at the start of `sn generate` when `--source` matches a recently-rebuilt
   source (detected via `(:IDS)` node mtime).
2. Move under `sn tools reconcile`, hidden from top-level help.

**Recommendation:** option 2 — reconcile is rare, operator-driven, and the
auto-detection heuristic is fragile. Moving to `sn tools reconcile` (hidden) removes
from the top-level surface but keeps it scriptable.

### 4.6  Final subcommand list

After proposals 4.1–4.5 land:

| # | subcommand | status | notes |
| --- | --- | --- | --- |
| 1 | `sn generate` | core | frozen in this PR |
| 2 | `sn enrich` | core | |
| 3 | `sn review` | core | gains `--names-only` this PR |
| 4 | `sn clear` | core | |
| 5 | `sn publish` | core | |
| 6 | `sn status` | core | |
| 7 | `sn gaps` | core | |
| 8 | `sn benchmark` | core | |
| 9 | `sn rotate` | core (kept, per 4.1) | |
| 10 | `sn import` | consolidated (absorbs `seed`) | default `--source=catalog` |
| — | `sn seed` | **hidden alias** → `sn import --source isn|west|all` | |
| — | `sn resolve-links` | **hidden** (scriptable catch-up) | |
| — | `sn reconcile` | **hidden** (move to `sn tools reconcile` later) | |

Visible count: **10**. Above the ≤ 8 target but the user-pinned core set is already
8; `rotate` and `import` are load-bearing. Without relaxing the pin on `benchmark`
(a potentially once-a-year command), we cannot go below 10. Recommendation: ship at
10 visible and revisit `benchmark` visibility later.

---

## Part 5 — `sn generate` canonical form (deferred)

This section documents the target shape `sn generate` should adopt **after** the
extraction-batching agent merges. No code changes in this patch.

| current | target |
| --- | --- |
| `--physics-domain` / `--domain` (alias) | `--physics-domain` (primary, alias `--domain`, alias `--ids` for back-compat of consumers that still think in IDS-scope) |
| `--limit` | `--limit` / `-n` |
| `-c`, `--cost-limit` | `-c`, `--cost-limit` ✓ |
| `--facility` | `--facility` / `-f` |
| `--compose-model` | **keep** — there is also no `--review-model` on generate, so `--compose-model` is the role-specific form. Do not merge with `--model`. |
| `--verbose`, `--quiet` | ✓ |
| `--dry-run`, `--force` | ✓ |
| `--retry-quarantined`, `--retry-skipped`, `--retry-vocab-gap` | consolidate into `--retry {quarantined,skipped,vocab-gap}` multi-value? Discuss separately. |

---

## Part 6 — Rename plan for this PR (applied to every non-generate command)

> *Implementation detail — D4 deliverable.* Each rename preserves the old name as an
> alias (no deprecation warning emitted at runtime, per AGENTS.md).

| command | old flag | new flag | alias kept |
| --- | --- | --- | --- |
| `sn review` | `--ids` | `--physics-domain` | yes (`--ids`, `--domain`) |
| `sn review` | `--domain` | `--physics-domain` | yes |
| `sn enrich` | `--domain` | `--physics-domain` | yes |
| `sn rotate` | `--domain` | `--physics-domain` | yes |
| `sn benchmark` | `--ids` | `--physics-domain` | yes (both become aliases, `--physics-domain` becomes new primary) |
| `sn benchmark` | `--domain` | `--physics-domain` | yes |
| `sn benchmark` | `--max-candidates` | `--max-candidates` **unchanged** — semantic "candidates" is domain-specific here | — |
| `sn publish` | `--ids` | `--physics-domain` | yes |
| `sn publish` | `--domain` | `--physics-domain` | yes |
| `sn clear` | `--ids` | `--physics-domain` | yes |
| `sn clear` | `--force/-f` | `--force` (drop `-f`) | keep `-f` as alias, since users may have scripts |
| `sn reconcile` | `--source-type` | `--source` | yes |
| `sn resolve-links` | unchanged | — | — |
| `sn seed` | unchanged (hidden alias only) | — | — |
| `sn import` | unchanged | — | — |
| `discover clear` | `--force/-f` | `--force` (drop `-f`) | keep `-f` alias |
| `discover paths` | `--focus -f` | `--focus` (drop `-f`) | keep `-f` alias |
| `discover signals` | `--focus -f` | `--focus` (drop `-f`) | keep `-f` alias |
| `discover code` | `--focus -f` | `--focus` (drop `-f`) | keep `-f` alias |
| `discover wiki` | `--focus -f` | `--focus` (drop `-f`) | keep `-f` alias |
| `discover documents` | `--focus -f` | `--focus` (drop `-f`) | keep `-f` alias |
| `discover paths` | `--path-limit -l` | `--limit -n` | keep `--path-limit`, `-l` aliases |
| `discover signals` | `--signal-limit -n` | `--limit -n` | keep `--signal-limit` alias |
| `discover wiki` | `--max-pages -n` | `--limit -n` | keep `--max-pages` alias |
| `discover code` | `--max-paths` | `--limit` | keep `--max-paths` alias |
| `discover documents` | `--max-paths` | `--limit` | keep `--max-paths` alias |

Short flags that become free after these renames: `-f` (facility), `-n` (limit).
This PR does **not** introduce `-f = --facility` anywhere new — `discover`
subcommands still take `FACILITY` positionally, and `sn` subcommands' new
`--facility` usage (rare — only `generate` and future) does not need a short. The
short-flag freeing is therefore **defensive**: it prevents future meaning-clash.

---

## Part 7 — D3: `sn review --name-only` design

> **Naming note.** The parallel agent's `sn generate --name-only` already landed
> (singular). We adopt the same spelling — `--name-only` — on `sn review` so the
> two flags compose naturally (`sn generate --name-only` → `sn review
> --name-only`). The DB-level field `review_mode` likewise stores `name_only` /
> `full`, mirroring `ExtractionBatch.mode`.

### 7.1  Behaviour

```
imas-codex sn review --name-only [other sn-review flags]
```

Scores only four dimensions — **grammar**, **semantic**, **convention**,
**completeness** — and skips documentation + compliance (which require full
description text). Persists a new `review_mode` field on `StandardName` ∈
{`name_only`, `full`}.

### 7.2  Normalization

```
reviewer_score_full   = sum(6 dims) / 120        # existing, 0-1
reviewer_score_namesonly = sum(4 dims) /  80     # new, 0-1
```

Both go into the same `reviewer_score` column (0-1). The dimension-set is recorded
via `review_mode` so downstream consumers (publish, enrich) can decide whether they
trust the score.

### 7.3  Downgrade protection

A `--name-only` run **must not overwrite** an existing full review unless `--force`
is given. Implemented as a WHERE-clause filter at target-selection time in
`_select_review_targets()`:

```python
if state.review_mode == "name_only" and not state.force_review:
    targets = [n for n in targets
               if n.get("review_mode") != "full"
               or n.get("reviewer_score") is None]
```

### 7.4  Prompt

Create `imas_codex/llm/prompts/sn/review_name_only.md` that renders only the four
dimension rubrics + calibration. Front-matter `dynamic: true`, `task: review`,
`used_by: imas_codex.standard_names.review.pipeline._review_single_batch`. Uses the
existing Jinja context (`items`, `calibration_entries`, `nearby_existing_names`,
grammar enums) but omits `documentation` and `compliance` rubric sections and
refers only to the four-dimension output schema.

### 7.5  Response model

New `StandardNameQualityScoreNameOnly` and matching batch model in
`imas_codex/standard_names/models.py`:

```python
class StandardNameQualityScoreNameOnly(BaseModel):
    grammar: int = Field(ge=0, le=20)
    semantic: int = Field(ge=0, le=20)
    convention: int = Field(ge=0, le=20)
    completeness: int = Field(ge=0, le=20)

    @property
    def total(self) -> int:
        return self.grammar + self.semantic + self.convention + self.completeness

    @property
    def score(self) -> float:
        return self.total / 80.0

    @property
    def tier(self) -> str:
        s = self.score
        if s >= 0.85: return "outstanding"
        if s >= 0.60: return "good"
        if s >= 0.40: return "adequate"
        return "poor"
```

Persist `review_mode` through `_match_reviews_to_entries()`:

```python
original["review_mode"] = "name_only" if state.review_mode == "name_only" else "full"
```

### 7.6  LinkML schema

Add to `imas_codex/schemas/standard_name.yaml`, under `StandardName.attributes`:

```yaml
      review_mode:
        description: >-
          Which review rubric produced reviewer_score. 'full' uses all
          six dimensions (grammar/semantic/documentation/convention/
          completeness/compliance, score normalized over 120). 'name_only'
          uses four dimensions (grammar/semantic/convention/completeness,
          score normalized over 80) — use this for name-only rotation
          cycles before description/documentation have been authored.
        range: StandardNameReviewMode
```

…and add the enum:

```yaml
enums:
  StandardNameReviewMode:
    permissible_values:
      full:
        description: All six dimensions scored.
      name_only:
        description: Grammar/semantic/convention/completeness only.
```

Re-run `uv run build-models --force` locally (do **not** commit generated files).

### 7.7  Tests

Add to `tests/standard_names/`:

- `test_review_name_only.py` — unit tests for `StandardNameQualityScoreNameOnly`
  total/score/tier calculations, normalization boundaries, JSON round-trip.
- Extend `test_review_pipeline.py` — add a test that renders the name-only prompt
  and asserts the prompt body does not contain the words "Documentation Quality"
  or "Prompt Compliance".
- Downgrade guard test — `--name-only` on a name with `review_mode='full'` and
  no `--force` must leave it untouched.

### 7.8  CLI surface

`sn review` gains **one** new flag: `--name-only` (is_flag). No other review flags
change this PR.

---

## Part 8 — Sequenced implementation for this PR

Commits in order (5 expected):

1. **`refactor(sn): rename --ids/--domain → --physics-domain with aliases`** —
   touches `sn/review`, `sn/enrich`, `sn/rotate`, `sn/benchmark`, `sn/publish`,
   `sn/clear`, `sn/reconcile`.
2. **`refactor(cli): free -f short on non-facility commands`** — `discover
   paths/signals/code/wiki/documents`, `discover clear`, `sn clear`.
3. **`refactor(discover): harmonize --limit / -n across discover subcommands`** —
   rename `--path-limit`, `--signal-limit`, `--max-pages`, `--max-paths` to
   `--limit` (aliases preserved).
4. **`feat(sn): add review_mode enum to standard name schema`** — LinkML only, no
   runtime code yet.
5. **`feat(sn): sn review --name-only scores 4 dimensions`** — prompt, models,
   pipeline wiring, CLI flag, tests.

`sn generate` is **not touched** — deferred to a follow-up after the
extraction-batching agent merges. Sequencing documented in
`plans/features/33-cli-harmonization.md`.
