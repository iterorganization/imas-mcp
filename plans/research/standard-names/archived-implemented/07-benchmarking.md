# Feature 07: Benchmarking

**Repository:** imas-codex  
**Complexity:** Medium  
**Depends on:** 05 (SN Build Pipeline)  
**Wave:** 4

---

## Overview

Comparative evaluation of different LLM models for standard name generation. Measures quality, cost, and speed to inform production model selection.

## Design

### Benchmark Command
```
imas-codex sn benchmark --source dd --ids equilibrium --models claude-sonnet,gpt-4o,gemini-pro
```

### Metrics
- **Quality:** Grammar validity rate, review acceptance rate, semantic accuracy score
- **Cost:** USD per name generated, tokens per name
- **Speed:** Names per minute, total pipeline time
- **Consistency:** Same input → same output across runs (determinism check)

### Benchmark Dataset
- Fixed set of DD paths / signals as input (reproducible)
- Known-good reference names for overlap comparison
- Edge cases: complex names, binary operators, multi-component

### Output
- Comparison table (stdout via Rich)
- JSON report for programmatic consumption
- Per-model breakdown with confidence intervals

## Deliverables

- [ ] `imas-codex sn benchmark` CLI command
- [ ] Benchmark runner with multi-model support
- [ ] Quality scoring against reference set
- [ ] Rich comparison table output
- [ ] JSON report export
- [ ] Benchmark dataset (fixed DD paths + expected outputs)
