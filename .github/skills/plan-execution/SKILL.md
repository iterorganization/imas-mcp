---
name: plan-execution
description: How to read and execute feature plans from the imas-codex plans/ directory. Use when implementing features from plan documents or when the fleet orchestrator dispatches work.
---

# Executing Feature Plans

## Plan Locations

| Location | Purpose |
|----------|---------|
| `plans/features/*.md` | Active plans (unstarted or in-progress) |
| `plans/features/pending/*.md` | Partially implemented (gaps documented in parent dir) |
| `plans/features/gaps-*.md` | Consolidated remaining work from related plans |

## Plan Structure

Plans typically contain:

- **Executive Summary** — what and why
- **Phases** — ordered implementation steps
- **Per-phase details** — files to modify, test cases, acceptance criteria
- **Dependencies** — what must exist before a phase can start
- **Documentation Updates** — what docs to update when complete

## Execution Workflow

1. **Read** the full plan to understand scope and dependencies
2. **Identify** which phases/tasks are ready (dependencies met)
3. **For each task**:
   a. Read the specific section carefully
   b. Note the listed files, test cases, and acceptance criteria
   c. Implement changes
   d. Run specified tests
   e. Verify acceptance criteria
4. **After completing a phase**, run the full test suite
5. **Commit** with conventional commit message (no plan phase references in title)

## Task Decomposition

When breaking a plan into tasks:

- **One task per phase subsection** (e.g., "1a", "1b", "2a")
- **Respect phase ordering** — later phases may depend on earlier ones
- **Identify parallel tasks** — items marked "independent" or "can be implemented in parallel"
- **Check for shared infrastructure** — multiple tasks may need a common utility first

## Complexity Assessment

A task is **straightforward** (→ `@engineer` agent) when:

- The plan specifies exact file paths and changes
- Changes are isolated to 1-3 files
- Test cases are provided in the plan
- It follows an established pattern in the codebase

A task is **complex** (→ `@architect` agent) when:

- The plan says "investigate", "determine", or "design"
- Changes span 4+ files across multiple modules
- New patterns or shared infrastructure are needed
- Integration with unfamiliar subsystems is required
- The plan has gaps that require codebase research to fill

## Using /fleet for Parallel Execution

The `/fleet` command automatically decomposes work into subtasks and dispatches to
custom agents based on their descriptions. When a plan has independent tasks:

1. Switch to plan mode (Shift+Tab) and create the plan
2. Use `/fleet` to execute — it will route simple tasks to `engineer` and complex
   tasks to `architect` based on the agent descriptions
3. You can also force routing: `@engineer fix the dd_version bug` or
   `@architect design the new pipeline`

## Scoring Dimensions

Rate each task 0-3 on each dimension:

| Dimension | 0 (Simple) | 1 (Moderate) | 2 (Complex) | 3 (Deep) |
|-----------|-----------|--------------|-------------|----------|
| **Files** | 1 file | 2-3 files | 4-6 files | 7+ files |
| **Ambiguity** | Fully specified | Minor gaps | Design choices | Research needed |
| **Dependencies** | None | 1 internal | 2-3 internal | External + internal |
| **Testing** | Add to existing | New test file | New test patterns | Integration tests |
| **Risk** | Local change | Module-scoped | Cross-module | Breaking change |

**Total ≤ 5** → `@engineer` (Sonnet 4.6 — fast, precise)
**Total ≥ 6** → `@architect` (Opus 4.6 — researches then builds)

## Plan Lifecycle

```
plans/features/<name>.md          → Active (implement from here)
plans/features/pending/<name>.md  → Reference material (gaps documented)
DELETE the plan file              → Fully implemented (code is the doc)
```

## Documentation Checklist

Every completed plan must update affected documentation:

| Target | When to Update |
|--------|----------------|
| `AGENTS.md` | New CLI commands, MCP tools, config, workflows |
| `README.md` | User-facing features, installation changes |
| `plans/README.md` | Plan status changes |
| `.claude/skills/` | New reusable workflows |
| `docs/` | Mature architecture documentation |
| Prompt templates | New or changed LLM prompts |

## Common Plan Types in This Project

- **Bug fix plans** — Phase 1 items, specific file + line references, test cases provided
- **Pipeline plans** — Multi-phase, schema → code → tests → docs progression
- **Server improvement plans** — Tool-level changes with A/B test criteria
- **Schema evolution plans** — LinkML YAML → build-models → migrate data → update code
