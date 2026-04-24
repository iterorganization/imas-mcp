# OpenRouter Rate-Limit Ceilings

## 2026-04-24 16:01 UTC

N sweep: [32, 48, 64, 96, 128]
Total spend: $1.3712

| Model                         | Ceiling | Cost ($) | Notes           |
| ----------------------------- | ------- | -------- | --------------- |
| anthropic/claude-haiku-4.5    | ≥128    | 0.1240   | no 429 in sweep |
| anthropic/claude-sonnet-4.6   | ≥128    | 0.3952   | no 429 in sweep |
| anthropic/claude-opus-4.6     | ≥128    | 0.6127   | no 429 in sweep |
| openai/gpt-5.4                | ≥128    | 0.2393   | no 429 in sweep |
| google/gemini-3.1-pro-preview | 0       | 0.0000   | 429 at N=-1     |

### Interpretation (Wave 3 Extended Probe)

- **GPT-5.4 (NEW):** Temperature fix (`_build_kwargs` GPT-5 guard, commit `ce3849e6`) confirmed
  working — all 128 concurrent requests succeeded at every N level. No 429s.
  Peak RPM at N=128: **1521**.

- **Anthropic (Haiku 4.5 / Sonnet 4.6 / Opus 4.6):** All three cleared N=128 with zero
  429s and zero errors. The true OpenRouter concurrency ceiling is **≥128** for this
  account tier.  Peak observed RPM: Haiku=1314, Sonnet=1152, Opus=938.

- **Gemini 3.1 pro preview:** Persistent 100% parse-error failure — model returns prose
  ("Here is…") even with `response_format=json_object`. The `_sanitize_content` prose
  extractor cannot help because no JSON object is present in the response body.
  Gemini requires a different prompt or a raw-text-to-JSON post-parse pipeline.
  **Status: incompatible with current probe schema; rate ceiling unmeasured.**
  Gemini is NOT a compose candidate until structured-output compatibility is resolved.

### Sizing Decision (Case A — single semaphore)

All four working models produced **zero 429s at N=128** (sweep ceiling).  Variance
across models is <2× (all ≥128).  **Case A** applies:

| Decision | Value |
|---|---|
| Lowest clean ceiling | ≥128 |
| 75% headroom applied | floor(0.75 × 128) = **96** |
| `sn-compose.max-concurrency` updated | 24 → **96** |
| Per-model semaphore (`phase7-per-model-sem`) | **Skipped** — variance <2× |

**Pilot coordination:** No `~/.local/share/imas-codex/logs/sn.log` found at probe
start — pilot agent was not active; probe results reflect uncontested rate limits.

### Per-model level detail

#### anthropic/claude-haiku-4.5

| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0 | 0 | 1.7 | 10.0 | 193 | 0.0108 |
| 48 | 48 | 0 | 0 | 1.6 | 4.7 | 615 | 0.0162 |
| 64 | 64 | 0 | 0 | 1.7 | 3.3 | 1180 | 0.0216 |
| 96 | 96 | 0 | 0 | 2.4 | 10.0 | 578 | 0.0324 |
| 128 | 128 | 0 | 0 | 2.9 | 5.8 | 1314 | 0.0431 |

#### anthropic/claude-sonnet-4.6

| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0 | 0 | 1.7 | 3.3 | 585 | 0.0344 |
| 48 | 48 | 0 | 0 | 1.7 | 5.7 | 503 | 0.0516 |
| 64 | 64 | 0 | 0 | 2.0 | 8.4 | 456 | 0.0687 |
| 96 | 96 | 0 | 0 | 3.0 | 8.0 | 723 | 0.1031 |
| 128 | 128 | 0 | 0 | 3.3 | 6.7 | 1152 | 0.1375 |

#### anthropic/claude-opus-4.6

| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0 | 0 | 2.2 | 4.0 | 478 | 0.0533 |
| 48 | 48 | 0 | 0 | 2.2 | 3.1 | 940 | 0.0799 |
| 64 | 64 | 0 | 0 | 2.4 | 13.3 | 289 | 0.1066 |
| 96 | 96 | 0 | 0 | 3.0 | 6.7 | 855 | 0.1598 |
| 128 | 128 | 0 | 0 | 4.4 | 8.2 | 938 | 0.2131 |

#### openai/gpt-5.4

| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 32 | 0 | 0 | 2.2 | 4.1 | 472 | 0.0209 |
| 48 | 48 | 0 | 0 | 1.3 | 3.4 | 851 | 0.0312 |
| 64 | 64 | 0 | 0 | 1.4 | 3.8 | 1003 | 0.0416 |
| 96 | 96 | 0 | 0 | 1.8 | 4.9 | 1176 | 0.0624 |
| 128 | 128 | 0 | 0 | 2.5 | 5.0 | 1521 | 0.0831 |

#### google/gemini-3.1-pro-preview

| N | ok | 429s | errs | ttfb_med(s) | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 0 | 0 | 32 | 5.8 | 13.5 | 142 | 0.0000 |


---

## 2026-04-24 09:09 UTC

N sweep: [4, 8, 12, 16, 24, 32]
Total spend: $0.2953

| Model                         | Ceiling | Cost ($) | Notes                                        |
| ----------------------------- | ------- | -------- | -------------------------------------------- |
| anthropic/claude-haiku-4.5    | ≥32     | 0.0324   | no 429 in sweep; true ceiling unknown        |
| anthropic/claude-sonnet-4.6   | ≥32     | 0.1031   | no 429 in sweep; true ceiling unknown        |
| anthropic/claude-opus-4.6     | ≥32     | 0.1598   | no 429 in sweep; true ceiling unknown        |
| google/gemini-3.1-pro-preview | n/a     | 0.0000   | API compat: returns prose, not JSON schema   |
| openai/gpt-5.4                | n/a     | 0.0000   | API compat: rejects temperature=0.0          |

### Interpretation

- **Anthropic (Haiku 4.5 / Sonnet 4.6 / Opus 4.6):** All 3 completed N=32 parallel requests
  with zero 429s and zero errors. No rate ceiling was found within the sweep range.
  The true OpenRouter limit is >32 (likely per-account tier; OpenRouter reports ~600–1000 RPM
  for Opus-tier accounts). Observed peak RPM at N=32: Haiku=561, Sonnet=408, Opus=604.

- **Gemini 3.1 pro preview:** Every request failed with a Pydantic JSON parse error —
  the model returns prose ("Here is the standard name…") instead of JSON even when
  `response_format` is set to a JSON schema. This is a model-side structured-output
  incompatibility at the probe payload size, not a rate limit. Rate ceiling unmeasured.
  **Action required before Phase 8:** use a retry-aware JSON extraction wrapper or
  switch to Gemini Flash for compose tasks.

- **GPT-5.4:** Every request failed with `UnsupportedParamsError: temperature=0.0`.
  GPT-5.x reasoning models only support `temperature=1`. This is a client-side config
  error, not a rate limit. Rate ceiling unmeasured.
  **Action required before Phase 8:** strip `temperature` from GPT-5.x requests in
  `_build_kwargs` (model-family check) or default to `None` in the probe.

### Sizing decision

- Compose model is **Opus 4.6** (primary) or **Sonnet 4.6** (fallback).
- Anthropic ceiling: **≥32**; no 429s in sweep.
- Applied 75% headroom: **0.75 × 32 = 24** → `sn-compose.max-concurrency = 24`.
- Per-model ceiling variance among Anthropic trio: **0** (all ≥32) → **no per-model pools needed** (Phase 7b skipped).


### Per-model level detail

#### anthropic/claude-haiku-4.5

| N | ok | 429s | errs | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 0 | 0 | 9.2 | 26 | 0.0013 |
| 8 | 8 | 0 | 0 | 13.8 | 35 | 0.0027 |
| 12 | 12 | 0 | 0 | 3.9 | 186 | 0.0040 |
| 16 | 16 | 0 | 0 | 5.5 | 174 | 0.0054 |
| 24 | 24 | 0 | 0 | 2.9 | 498 | 0.0081 |
| 32 | 32 | 0 | 0 | 3.4 | 561 | 0.0108 |

#### anthropic/claude-sonnet-4.6

| N | ok | 429s | errs | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 0 | 0 | 16.0 | 15 | 0.0043 |
| 8 | 8 | 0 | 0 | 5.6 | 85 | 0.0086 |
| 12 | 12 | 0 | 0 | 3.7 | 192 | 0.0129 |
| 16 | 16 | 0 | 0 | 5.7 | 168 | 0.0172 |
| 24 | 24 | 0 | 0 | 4.8 | 299 | 0.0258 |
| 32 | 32 | 0 | 0 | 4.7 | 408 | 0.0344 |

#### anthropic/claude-opus-4.6

| N | ok | 429s | errs | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 4 | 0 | 0 | 3.0 | 80 | 0.0067 |
| 8 | 8 | 0 | 0 | 3.9 | 123 | 0.0133 |
| 12 | 12 | 0 | 0 | 3.8 | 191 | 0.0200 |
| 16 | 16 | 0 | 0 | 3.6 | 267 | 0.0266 |
| 24 | 24 | 0 | 0 | 3.7 | 393 | 0.0400 |
| 32 | 32 | 0 | 0 | 3.2 | 604 | 0.0533 |

#### google/gemini-3.1-pro-preview

| N | ok | 429s | errs | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0 | 0 | 4 | 8.8 | 27 | 0.0000 |
| 8 | 0 | 0 | 8 | 38.2 | 13 | 0.0000 |
| 12 | 0 | 0 | 12 | 35.8 | 20 | 0.0000 |
| 16 | 0 | 0 | 16 | 81.9 | 12 | 0.0000 |
| 24 | 0 | 0 | 24 | 13.7 | 105 | 0.0000 |
| 32 | 0 | 0 | 32 | 71.5 | 27 | 0.0000 |

#### openai/gpt-5.4

| N | ok | 429s | errs | wall(s) | rpm | cost($) |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 0 | 0 | 4 | 0.2 | 1133 | 0.0000 |
| 8 | 0 | 0 | 8 | 0.0 | 10473 | 0.0000 |
| 12 | 0 | 0 | 12 | 0.1 | 10720 | 0.0000 |
| 16 | 0 | 0 | 16 | 0.1 | 10263 | 0.0000 |
| 24 | 0 | 0 | 24 | 0.1 | 10466 | 0.0000 |
| 32 | 0 | 0 | 32 | 0.2 | 10612 | 0.0000 |

