---
name: metadata_population
description: Per-call context for IDS metadata population (dynamic, changes per call)
---

# Metadata Population: {{ ids_name }}

## Facility

{{ facility }}

## IDS

**{{ ids_name }}**: {{ ids_description }}

## DD Version

{{ dd_version }}

## Already Populated (Deterministic)

The following fields have been populated programmatically — do **not** override these:

{{ deterministic_fields_summary }}

## Mapped Signals Context

These signals have been mapped to this IDS:

{{ mapped_signals_summary }}

## Task

Populate the following metadata fields based on the context above:

1. **comment** — Summarize what this IDS mapping contains for {{ facility }}: which
   diagnostics contribute, what physical quantities are covered, and any notable
   characteristics or caveats.

2. **occurrence_type** — Classify this IDS occurrence:
   - `0` (machine_description) — static geometry or device parameters
   - `1` (experimental) — measured/acquired data from a pulse
   - `2` (simulation) — modeled or reconstructed data
   - `3` (composite) — mixed sources

3. **provenance_sources** — Describe the data provenance chain for {{ facility }}:
   the originating diagnostic or subsystem, any processing steps, and the storage
   system (e.g., MDSplus tree name, database).

4. **homogeneous_time** — Determine time homogeneity:
   - `0` (heterogeneous) — signals have different time bases
   - `1` (homogeneous) — all signals share a common time base
   - `2` (independent) — no time dependence (static data)
