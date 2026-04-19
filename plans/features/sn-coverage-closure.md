# SN Coverage Closure — all relevant DD nodes

## Goal

From the user: *"Our target is a robust standard name generation or linking for ALL
relevant nodes in the DD."*

## Current coverage

After 20+ rotation iterations, 335 valid SNs span 9 physics domains. The graph holds
`node_category IN ['quantity', 'geometry']` across 32 physics_domain values, but only
9 of those domains have SN coverage. Based on untouched-domain size estimates from
prior sessions:

| Domain | Paths (untouched) | Priority |
|---|---|---|
| magnetic_field_systems | 269 | P1 |
| gyrokinetics | 264 | P1 |
| stellarator_geometry | ~200 | P2 |
| wave_physics | ~150 | P2 |
| particle_measurement_diagnostics | ~180 | P2 |
| general_diagnostics | ~120 | P3 |
| ... (9+ smaller domains) | ~100 each | P3 |

## Protocol (one domain per session)

Same as the proven rotation in Plan 31:

```bash
# 1. baseline extract
uv run imas-codex sn generate --source dd --physics-domain D -c 2 --limit 60

# 2. review
mcp search_standard_names "D domain" --k 30
# + reviewer agent pass for quarantine analysis

# 3. fix pattern issues
#    prompt patches → commit in codex
#    vocab gaps     → ISN PR
#    audit gaps     → audits.py

# 4. iter 2 with fresh pending names + applied fixes
#    (if iter-1 quarantine > 10 %, iter again; else move on)
```

## Batch cost envelope

$2/domain × ~15 domains = $30 total for first-pass coverage. Iter-2 for plateau-busting
adds another $10-15. Total budget ~$45.

## Coverage stop rule

Stop when either:
- 90 % of `node_category='quantity'` paths per domain have a `PRODUCED_NAME` source edge, OR
- Remaining paths are flagged `skip/metadata` by the classifier.

## Order

1. **P1**: `magnetic_field_systems` (pure physics, largest untouched) → `gyrokinetics`
   (theory-heavy, good vocabulary stress test).
2. **P2**: `stellarator_geometry` (geometric canon test) → `wave_physics` (completes
   ICRH/ECRH/LH/wave family) → `particle_measurement_diagnostics` (Stokes/probe NC-30
   stress test).
3. **P3**: remaining smaller domains in any order.

## Success criteria

- All 32 populated physics domains have ≥ 1 valid SN after rotation.
- Median validation-pass rate across domains ≥ 90 %.
- Unique-name count ≥ 600 (2× current corpus).
- `sn status` shows zero `extracted` backlog for covered domains.

## Dispatch

Operator-driven per session, one domain per session. Background agent dispatches
acceptable for iter-2 fix application when the issue list is well-characterised.
