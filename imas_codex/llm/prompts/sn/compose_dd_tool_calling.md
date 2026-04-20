---
name: sn/compose_dd_tool_calling
description: Variant C — lean compose prompt that fetches DD context on-demand via tool calls
used_by: scripts/prompt_ab.py (plan 32 Phase 2 research harness)
task: composition
dynamic: true
schema_needs: []
---

Generate Standard Names for the following IMAS Data Dictionary paths.

## Core rules

- **Unit is authoritative** and comes from the DD `HAS_UNIT` relationship.
  Do not include unit in your output — it is injected at persistence time.
  Use it only to disambiguate the physical quantity.
- **Name = `physical_base` [+ modifiers]**, lowercase snake_case, never
  include abbreviations, symbols, measurement methods, or processing
  adjectives (`filtered_`, `reconstructed_`, `averaged_`). Position tokens
  go after the physical base (`electron_temperature_core`, not
  `core_electron_temperature`).
- **No unit strings, no IDS names, no method names** in the Standard Name.
- **Follow controlled vocabulary**: use `poloidal_magnetic_flux` not
  `poloidal_flux`; `electron_temperature` not `electron_temp`; etc.

## Tool-calling policy (variant C)

You have access to three tools that fetch additional context **only when
you need it**. The goal is to keep the prompt lean and let you pull in
siblings, reference exemplars, or version history on demand rather than
front-loading all of it:

- `fetch_cluster_siblings(cluster_id)` — returns names already assigned
  to paths in the same semantic cluster. Use when you are unsure whether
  a similar name has been established.
- `fetch_reference_exemplar(concept)` — returns a published exemplar
  Standard Name that matches a concept (e.g. `"electron temperature"`).
  Use to confirm controlled-vocabulary choices.
- `fetch_version_history(path)` — returns DD version change history for
  one path. Use when the path description references a renamed or
  repurposed field.

**Budget:** at most 2 tool calls per batch. Prefer to emit the name
directly if the context is obvious.

## Output

Return a JSON array of `{path, standard_name, rationale}` objects. The
`rationale` must be one sentence citing the physical quantity and any
tool call evidence you used.

{{ paths_block }}
