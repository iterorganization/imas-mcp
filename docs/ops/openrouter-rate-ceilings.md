# OpenRouter Rate-Limit Ceilings

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

