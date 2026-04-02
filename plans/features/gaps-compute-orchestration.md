# Compute Session Orchestration — Outstanding Gaps

Consolidated from `pending/compute-session-orchestration.md`.

## Status Summary

The plan is ~10% implemented. Foundational pieces exist (cx script, SLURM job templates, settings) but the core session orchestration is unstarted. A Python CLI alternative (`imas-codex hpc`) provides basic HPC job management but not the reconnectable session pattern.

### What exists
- `cx` session manager (~/.local/bin/cx, 279 lines) with SSH+zellij session management
- SLURM job scripts: `slurm/codex-embed.sh`, `slurm/ollama-llm.sh`
- Settings infrastructure for LLM proxy URL, embed host discovery, port mapping
- Zellij layouts (claude.kdl, codex.kdl, facility.kdl, etc.)
- Python alternative: `imas-codex hpc` CLI with status, shell, run, submit, attach, cancel, info

### Different path note
The `imas-codex hpc` Python CLI covers HPC job management but **not** the persistent zellij session lifecycle or LiteLLM migration that the plan describes. The plan envisions a shell-based `cx compute` orchestration; the codebase has a Python CLI alternative. These solve **overlapping but different problems** — the Python CLI is adequate for job management but not for session reconnection.

---

## Outstanding Gaps

### Phase 1: SLURM Session Host (unstarted)
- `slurm/session-host.sh` — missing entirely
- `cx-slurm.sh` functions (`cx_find_job`, `cx_submit_job`, `cx_connect`) — not implemented
- Need: Allocate a persistent compute node via SLURM for interactive work

### Phase 2: cx compute mode (unstarted)
- `cx compute` subcommand not added to cx script
- No SLURM-backed session workflow (find/submit/connect pattern)
- Need: Single command to get an interactive compute session

### Phase 3: LiteLLM migration to compute (unstarted)
- `slurm/litellm-proxy.sh` — missing
- No HOSTALIASES DNS workaround deployed
- No service discovery file written
- Need: Move LiteLLM proxy from login node to compute node

### Phase 4: Tab integration (unstarted)
- `cx-tab` script — missing
- `compute-project.kdl` layout — missing
- Need: Zellij tab/pane management for compute sessions

### Phase 5: Status dashboard (unstarted)
- No `cx compute status` dashboard
- No end-to-end compute orchestration monitoring
- Need: Unified view of all compute services (embed, LiteLLM, Neo4j, sessions)

---

## Recommendation

This plan is large (~1000 lines) and mostly unstarted. Consider whether:
1. The `imas-codex hpc` Python CLI is sufficient for current needs (it may be)
2. The session orchestration features warrant a fresh, simplified plan
3. The LiteLLM-on-compute migration is still desired given current infrastructure

---

## Priority & Dependencies

**Priority: P4 — Low (Python CLI covers basic HPC needs)**

| Depends On | Enables |
|-----------|---------|
| None | Developer experience improvement (non-blocking) |

The existing `imas-codex hpc` Python CLI covers job submission, attachment, and status. This plan adds session persistence and LiteLLM migration which are quality-of-life improvements, not functional blockers.

## Overlap Notes

This gap document consolidates work from `pending/compute-session-orchestration.md` (the original detailed specification). The pending plan is reference material for detailed implementation specs.

**Architectural note:** The Python CLI (`imas-codex hpc`) and the shell-based `cx compute` approach solve overlapping problems with different tooling. Consider whether the Python CLI is sufficient before implementing the full shell-based plan.

## Documentation Updates

When this work is complete, update:
- [ ] `AGENTS.md` — Compute Infrastructure section with new commands/workflows
- [ ] `.claude/skills/` — Add compute session management skill if cx compute is implemented
- [ ] `plans/README.md` — mark this gap doc as complete
