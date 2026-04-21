# 42 — Documentation Backfill Cycle

**Date**: 2025-07-22
**Objective**: Backfill textbook-quality `documentation` on StandardName nodes that are missing it, using `sn generate --target docs`.

## Phase 1 — Baseline

### Missing-docs counts (pre-backfill)

| Domain | Total | With Docs | Missing | Coverage |
|--------|-------|-----------|---------|----------|
| transport | 238 | 233 | 5 | 98% |
| general | 142 | 46 | 96 | 32% |
| edge_plasma_physics | 75 | 71 | 4 | 95% |
| equilibrium | 63 | 62 | 1 | 98% |
| magnetohydrodynamics | 33 | 32 | 1 | 97% |
| fast_particles | 29 | 26 | 3 | 90% |
| auxiliary_heating | 28 | 25 | 3 | 89% |
| plant_systems | 27 | 26 | 1 | 96% |
| plasma_wall_interactions | 17 | 16 | 1 | 94% |
| turbulence | 14 | 13 | 1 | 93% |
| waves | 14 | 11 | 3 | 79% |
| structural_components | 11 | 9 | 2 | 82% |
| (None) | 1 | 0 | 1 | 0% |

**Total missing: 122 docs** (out of ~900 total SNs)

### Priority order
1. `general` (96 missing — biggest gap by far)
2. `transport` (5)
3. `edge_plasma_physics` (4)
4. `waves` (3), `auxiliary_heating` (3), `fast_particles` (3)
5. Remaining single-missing domains

---

## Phase 2 — Generation Runs

### Run 1: general domain
```
uv run imas-codex sn generate --target docs --physics-domain general -c 5
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

### Run 2: transport domain
```
uv run imas-codex sn generate --target docs --physics-domain transport -c 1
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

### Run 3: edge_plasma_physics domain
```
uv run imas-codex sn generate --target docs --physics-domain edge_plasma_physics -c 1
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

### Run 4: waves domain
```
uv run imas-codex sn generate --target docs --physics-domain waves -c 0.5
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

### Run 5: auxiliary_heating + fast_particles
```
uv run imas-codex sn generate --target docs --physics-domain auxiliary_heating -c 0.5
uv run imas-codex sn generate --target docs --physics-domain fast_particles -c 0.5
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

### Run 6: Remaining small domains
Domains: structural_components, plant_systems, turbulence, plasma_wall_interactions, magnetohydrodynamics, equilibrium
```
# 1–2 missing each, loop with --limit 10
```
- **Status**: pending
- **Cost**: —
- **Docs generated**: —

---

## Phase 3 — Review & Quality

### Spot-check samples
_(to be filled after generation)_

### Review scores
_(to be filled after `sn review`)_

---

## Phase 4 — Quality Issues

### Flagged docs
_(to be filled after analysis)_

---

## Phase 5 — Summary

### Before/after
| Metric | Before | After |
|--------|--------|-------|
| Total missing docs | 122 | — |
| general coverage | 32% | — |
| Overall coverage | ~87% | — |

### Cost breakdown
| Domain | Cost |
|--------|------|
| general | — |
| transport | — |
| edge_plasma_physics | — |
| waves | — |
| auxiliary_heating | — |
| fast_particles | — |
| small domains | — |
| **Total** | **—** |

### Sample good docs
_(to be filled)_

### Sample bad docs
_(to be filled)_

### Prompt improvements for next cycle
_(to be filled)_
