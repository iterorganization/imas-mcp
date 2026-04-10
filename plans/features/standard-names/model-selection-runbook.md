# Model Selection Runbook — Standard Name Generation

Practical guide for running multi-model benchmarks and selecting the best
LLM for each role in the standard name pipeline.

## Quick Start

```bash
# Compare two models on equilibrium IDS (fast, <$0.50)
uv run imas-codex sn benchmark \
  --source dd \
  --ids equilibrium \
  --max-candidates 10 \
  --models google/gemini-3.1-flash-lite-preview,anthropic/claude-sonnet-4-6

# Full benchmark with reviewer scoring (~$1–2)
uv run imas-codex sn benchmark \
  --source dd \
  --max-candidates 50 \
  --models google/gemini-3.1-flash-lite-preview,anthropic/claude-sonnet-4-6 \
  --reviewer-model anthropic/claude-sonnet-4-6

# Export results to JSON for offline analysis
uv run imas-codex sn benchmark \
  --source dd \
  --ids equilibrium \
  --max-candidates 20 \
  --models google/gemini-3.1-flash-lite-preview \
  --output benchmark-results.json
```

## Cost Cap Guidance

| Scenario            | `--max-candidates` | Expected cost |
|---------------------|--------------------|---------------|
| Quick smoke test    | 5–10               | < $0.20       |
| Single-IDS compare  | 20–30              | $0.30–$0.80   |
| Full benchmark      | 50                 | $0.50–$1.50   |
| Production eval     | 100+               | $1.50–$4.00   |

> **Hard rule:** never exceed **$2.00** per benchmark execution during
> development and model selection.  Monitor cost in the output table and
> abort early if needed.

## Approved Model List

Models are accessed via the LiteLLM proxy running on the ITER login node.
The proxy routes requests through OpenRouter. Pass model identifiers as
configured in `pyproject.toml` (the proxy handles the `openrouter/` prefix).

### Compose Role (name generation)

| Model                                    | Tier     | Notes                              |
|------------------------------------------|----------|------------------------------------|
| `google/gemini-3.1-flash-lite-preview`   | Primary  | Current `language` config, cheapest|
| `google/gemini-2.5-flash`               | Alt      | Fast, good grammar                 |
| `anthropic/claude-sonnet-4-6`           | Fallback | Higher quality, 3–5× cost          |
| `anthropic/claude-haiku-4`              | Budget   | Fastest Anthropic, lowest cost     |

### Review Role (quality scoring)

| Model                                    | Tier     | Notes                              |
|------------------------------------------|----------|------------------------------------|
| `anthropic/claude-sonnet-4-6`           | Primary  | Best judgment, calibrated scoring  |
| `anthropic/claude-opus-4-6`             | Premium  | Highest quality, ~10× cost         |
| `google/gemini-2.5-pro`                | Alt      | Strong reasoning, competitive cost |

### Currently Configured (pyproject.toml)

```toml
[tool.imas-codex.language]
model = "google/gemini-3.1-flash-lite-preview"   # compose

[tool.imas-codex.reasoning]
model = "anthropic/claude-sonnet-4-6"             # review / complex

[tool.imas-codex.agent]
model = "anthropic/claude-opus-4-6"               # agent
```

## Decision Criteria

| Metric              | Weight   | Threshold | How to read                         |
|----------------------|----------|-----------|-------------------------------------|
| Grammar valid %      | Critical | ≥ 95%     | Names must parse→compose round-trip |
| Fields consistent %  | High     | ≥ 85%     | Decomposed fields reconstruct name  |
| Reference recall     | High     | ≥ 60%     | Overlap with human-curated set      |
| Avg quality score    | High     | ≥ 65      | 5-dimensional reviewer rating /100  |
| Cost per name        | Medium   | < $0.01   | Total API cost ÷ candidate count    |
| Names/min            | Medium   | > 30      | Throughput including API latency     |
| Cache hit rate       | Low      | > 50%     | OpenRouter prompt cache utilization  |

### Interpreting the Table

The benchmark outputs a Rich table with these columns:

```
Model    Names  Valid%  Fields%  Ref Match  Cost     Names/min  $/name  Cache%  Errors
gemini   47     100%    96%      28/50      $0.0312  62         $0.0007 78%     0
claude   45     98%     91%      31/50      $0.1450  18         $0.0032 65%     1
```

- **Valid %** — ratio of candidates whose `standard_name` survives grammar
  `parse_standard_name()` → `compose_standard_name()` round-trip.
- **Fields %** — ratio where decomposed grammar fields (`subject`,
  `physical_base`, etc.) recompose to the same name string.
- **Ref Match** — `overlap/total` against the reference dataset
  (`benchmark_reference.py`).  Higher recall = more agreement with human picks.
- **Cache %** — `cache_read / (cache_read + cache_creation)` — how much of
  the system prompt was served from OpenRouter's prompt cache. Higher values
  reduce cost and latency on subsequent batches.
- **Errors** — number of batches where the LLM call failed (timeout,
  rate limit, parse error).

When `--reviewer-model` is used, additional columns appear:

- **Avg Quality** — mean 5-dimensional score (0–100).
- **Avg Doc Len** — average documentation string length.
- **Fields Pop%** — ratio of optional fields populated.

## Recommended Selections

### For compose (name generation)

Use the **cheapest model that meets grammar + fields thresholds**:

1. Start with `google/gemini-3.1-flash-lite-preview`
2. If grammar valid < 95%, try `anthropic/claude-sonnet-4-6`
3. If cost per name > $0.01, try `google/gemini-2.5-flash`

### For review (quality scoring)

Use a **stronger model than compose** for unbiased evaluation:

1. Primary: `anthropic/claude-sonnet-4-6`
2. If budget allows: `anthropic/claude-opus-4-6`

> **Never use the same model for both compose and review** — self-review
> inflates quality scores.

## Prompt Cache Optimization

OpenRouter supports provider-level prompt caching. The SN benchmark system
prompt includes grammar rules and calibration examples that remain constant
across batches. Cache behavior:

- **First batch**: `cache_creation_tokens` > 0 (system prompt written to cache)
- **Subsequent batches**: `cache_read_tokens` > 0 (system prompt served from cache)
- **Cost savings**: cached tokens cost ~75% less than uncached tokens

To maximize cache hit rate:
- Run batches in rapid succession (caches expire after ~5 minutes idle)
- Use the `openrouter/` prefix (caching is provider-side, not proxy-side)
- Group batches by model to keep the cache warm

## Troubleshooting

### LLM proxy not running

```bash
# Check proxy status
uv run imas-codex hpc status

# Start proxy (requires SSH tunnel to ITER)
uv run imas-codex llm start
```

If the proxy is unavailable, the benchmark will fail with connection errors.
The cache reporting code and grammar validation are tested offline via
`uv run pytest tests/sn/test_benchmark.py -v`.

### Budget exhausted

If you see `ProviderBudgetExhausted`, either:
1. The OpenRouter account is out of credits — top up
2. The `--cost-limit` was exceeded — increase or reduce `--limit`
