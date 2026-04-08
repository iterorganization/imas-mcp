# Current State: IMAS Standard Names Prompt Architecture

## Overview

The IMAS Standard Names project currently uses two distinct prompting approaches for standard name generation: **interactive MCP-mediated chat** and **scripted agent workflows**. Neither approach employs the structured prompt engineering patterns established in the partner `imas-codex` project.

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                  Standard Names Prompting                       │
├──────────────────────┬──────────────────────────────────────────┤
│   MCP Chat (primary) │   Agent Workflows (experimental)         │
│                      │                                          │
│  ┌────────────────┐  │  ┌────────────────┐  ┌───────────────┐  │
│  │ .github/prompts│  │  │ agent_loop_    │  │ agent_list_   │  │
│  │ (static .md)   │  │  │ workflow.py    │  │ generate.py   │  │
│  └───────┬────────┘  │  └───────┬────────┘  └──────┬────────┘  │
│          │           │          │                   │           │
│  Loaded by VS Code   │  pydantic_ai.Agent          │           │
│  or Copilot as       │  with MCP toolsets           │           │
│  prompt files        │  + inline prompts            │           │
│          │           │          │                   │           │
│  ┌───────▼────────┐  │  ┌──────▼─────────┐  ┌─────▼────────┐  │
│  │ Human operates │  │  │ Sequential     │  │ Review loop  │  │
│  │ chat manually  │  │  │ generate→      │  │ generate→    │  │
│  │ via MCP tools  │  │  │ review loop    │  │ AI review→   │  │
│  └────────────────┘  │  └────────────────┘  │ human review │  │
│                      │                      └──────────────┘  │
└──────────────────────┴──────────────────────────────────────────┘
```

## 1. MCP Chat Interaction Pattern (Primary)

### How it works

The primary method of minting standard names is through **interactive chat sessions** with an LLM that has access to MCP tools. The human operator drives the conversation, providing:

1. A natural-language request (e.g., "generate names for poloidal flux paths")
2. A list of IMAS DD paths to name
3. Iterative feedback on proposed names

The LLM calls MCP tools (`get_grammar`, `search_standard_names`, `compose_standard_name`, `create_standard_names`, `write_standard_names`) to research, compose, validate, and persist names.

### Prompt files

Static markdown files in `.github/prompts/` provide system-level instructions:

| File | Purpose | Lines |
|------|---------|-------|
| `generate_standard_names_from_imas_paths.prompt.md` | Workflow for generating names from IDS paths | ~68 |
| `workflows/list_generate_workflow/generate.prompt.md` | Phase-based generation system prompt | ~94 |
| `workflows/list_generate_workflow/review.md` | AI reviewer scoring rubric | ~80 |
| `workflows/list_generate_workflow/human.md` | Human interaction parsing | ~54 |

**Source**: `.github/prompts/generate_standard_names_from_imas_paths.prompt.md` [^1]

### Characteristics

- **Static prompts**: No templating, no dynamic content injection
- **No schema injection**: Grammar rules, vocabulary tokens, and Pydantic schemas are not embedded in prompts — the LLM must discover them via MCP tool calls
- **Sequential tool calling**: Each tool call happens in conversational turns, not batched
- **Human-in-the-loop throughout**: Every generation step requires human confirmation
- **No prompt caching**: System prompts are not designed for API-level cache reuse
- **No cost tracking**: No budget limits, no spend monitoring

### Strengths

- Flexible: can handle novel edge cases through conversation
- Simple implementation: no pipeline infrastructure needed
- Human quality control at every step

### Weaknesses

- **Not scalable**: Each name requires multiple chat turns (5-15 tool calls)
- **Inconsistent**: Prompt quality depends on operator skill
- **No context reuse**: Grammar, vocabulary, and schema are re-fetched every session
- **Expensive**: Redundant token usage from repeated context discovery
- **No batch processing**: Cannot process 50+ names efficiently
- **No reproducibility**: No structured prompt versioning or A/B testing

## 2. Agent Workflow Pattern (Experimental)

### Agent Loop Workflow

Located at `imas_standard_names/agents/agent_loop_workflow.py` [^2], this uses `pydantic_ai` to create a generate→review loop:

```python
# Agents defined with inline prompts
REVIEW_QUERY = "You are an expert in using the IMAS Data Dictionary..."

standard_name_agent = Agent[
    tuple[list[StandardNameEntry], StandardNameEntry | None, Review | None],
    StandardNameEntry,
](model=build_default_model(), output_type=StandardNameEntry, toolsets=SERVERS)

ai_review_agent = Agent[StandardNameEntry, Review](
    model=build_default_model(), output_type=Review, toolsets=SERVERS
)
```

**Key issues**:
- Inline string prompts with no structured context
- Sequential `await` calls — no parallel processing
- Hardcoded model selection (`openai/gpt-5`)
- No cost limits or retry logic
- MCP servers loaded from `.vscode/mcp.json` (development config)

### Agent List Generate Workflow

Located at `imas_standard_names/agents/agent_list_generage_workflow.py` [^3], this is more mature:

- Loads system prompts from `.github/prompts/workflows/` markdown files
- Implements a 2-tier review system (AI + human)
- Uses `build_regenerate_query()` and `build_review_query()` for dynamic prompt construction
- Has a retry loop with `MAX_RETRIES = 5`

**Key issues**:
- Prompts loaded via `Path(__file__).resolve()` — fragile path resolution
- String concatenation for dynamic content (`"\n".join(str(c) for c in candidates)`)
- No templating engine — prompt construction is ad-hoc Python string formatting
- No schema injection — review criteria are hardcoded prose
- No batch dispatch — candidates processed sequentially
- No cost tracking or budget limits

### Namelist Pattern

`imas_standard_names/agents/namelist.py` [^4] is a simpler script:

```python
agent = Agent(model, toolsets=servers, deps_type=Kind, output_type=list[StandardName])
result = agent.run_sync(
    "@imas-sn generate a list of standard names from the following IDS paths..."
)
```

This fires a single LLM call with IDS paths as inline text. No structured prompt, no review loop, no validation.

## 3. MCP Tool Architecture

The standard names MCP server provides 15+ tools [^5]:

| Category | Tools |
|----------|-------|
| Discovery | `search_standard_names`, `list_standard_names`, `fetch_standard_names` |
| Grammar | `get_grammar`, `compose_standard_name`, `parse_standard_name` |
| Vocabulary | `get_vocabulary`, `manage_vocabulary` |
| Creation | `create_standard_names`, `edit_standard_names`, `write_standard_names` |
| Validation | `check_standard_names`, `validate_catalog` |
| Schema | `get_schema`, `get_tokamak_parameters` |

These tools are well-designed for interactive use but are not optimized for batch pipeline consumption:
- Each tool call requires a full MCP request/response cycle
- No bulk query endpoints (e.g., "fetch grammar + vocabulary + schema in one call")
- No prompt-ready context assembly (tools return raw data, not prompt-formatted text)

## 4. Prompt Content Analysis

### Generate Prompt (`generate.prompt.md`)

The 94-line generation prompt follows a phased approach:

1. **Phase 0**: Input interpretation (parse user request)
2. **Phase 1**: Concept explanation (call `explain_concept` tool)
3. **Phase 2**: Exhaustive name survey (multiple `search_standard_names` calls)
4. **Phase 3**: Collision analysis
5. **Phase 4**: Name synthesis
6. **Phase 5**: Validation checklist
7. **Phase 6**: Output format

**Problem**: All domain knowledge (naming rules, token ordering, prohibited patterns) is encoded as natural language prose. The LLM must interpret and follow ~94 lines of text instructions, with no structured schema enforcement.

### Review Prompt (`review.md`)

The review prompt defines a weighted scoring model:

```
Score = 0.30*Uniqueness + 0.30*Descriptivity + 0.20*Generalizability + 0.20*Conventions
```

This is sound but relies entirely on LLM interpretation. There is no Pydantic schema enforcing output structure, no calibration examples, and no dimension-specific scoring guidance.

## 5. Gap Summary

| Dimension | Current State | Desired State |
|-----------|--------------|---------------|
| Prompt format | Static markdown, inline strings | Jinja2 templates with frontmatter |
| Context injection | LLM discovers via tool calls | Pre-assembled, schema-derived |
| Batch processing | Sequential, one-at-a-time | Parallel workers, batched dispatch |
| Schema enforcement | Natural language prose | Pydantic models + JSON schema |
| Cost tracking | None | Per-call, per-batch, budget limits |
| Prompt caching | None | Static-first layout, cache breakpoints |
| Reproducibility | None | Versioned prompts, deterministic rendering |
| Orchestration | Manual chat or simple loops | Engine with supervision, orphan recovery |
| Error handling | Basic try/except | Retry with backoff, structured errors |
| Calibration | None | Score calibration examples in prompts |

---

## Footnotes

[^1]: `.github/prompts/generate_standard_names_from_imas_paths.prompt.md`
[^2]: `imas_standard_names/agents/agent_loop_workflow.py`
[^3]: `imas_standard_names/agents/agent_list_generage_workflow.py`
[^4]: `imas_standard_names/agents/namelist.py`
[^5]: `imas_standard_names/tools/` directory (15+ tool files)
