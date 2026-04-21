# 43 — Vector grouping assertions for StandardName semantics

Investigates whether the three vector indexes that underpin Standard
Name semantic search (`standard_name_desc_embedding`,
`standard_name_source_desc_embedding`, `cluster_embedding` /
`cluster_label_embedding`) return coherent, physics-grounded groupings.

Implementation lives in
`tests/standard_names/test_vector_semantics.py`.

## Graph snapshot

Captured against the running `codex` graph on SLURM node
`98dci4-clu-2001`.

| label                   | nodes | with embedding | notes |
| ----------------------- | ----: | -------------: | ----- |
| StandardName            |   872 |            750 | 122 in `review_status='named'`, no description yet |
| StandardNameSource      |  6336 |          **0** | property unpopulated despite ONLINE index |
| IMASSemanticCluster     |  2808 |          2808 | `embedding`, `description_embedding`, `label_embedding` all populated |

Cluster scopes: `ids=1408`, `global=796`, `domain=604`.

Embedding model/dimensions: **Qwen/Qwen3-Embedding-0.6B @ 256-d**
(from `pyproject.toml → [tool.imas-codex.embedding]`).

## Test results (12 passed, 1 skipped, 1 xfailed)

| # | Test | Status | Notes |
| - | ---- | ------ | ----- |
| A1 | `test_sn_semantic_search_surfaces_expected_concept` | 5/5 PASS | top-10 contains the expected keyword for every concept |
| A2 | `test_sn_semantic_search_domain_coherence` | 4 PASS, 1 **xfail** | `plasma_current` top-5 spans 5 distinct physics domains |
| B  | `test_cluster_membership_groups_sns_semantically` | PASS | within-cluster mean 0.77 vs baseline 0.57 |
| C  | `test_standard_name_source_embeddings_group_by_producer` | SKIP | `StandardNameSource.embedding` is unpopulated |
| D  | `test_mcp_search_standard_names_returns_coherent_results` | PASS | `electron_density_at_separatrix` top-ranked; no unrelated magnetics result |
| E  | `test_cluster_labels_match_member_sns` | PASS | mean label↔member 0.72 across 30 sampled clusters |

## Empirical numbers

### Within-cluster vs random baseline cosine similarity

Random baseline (500 pairs from 700 SN embeddings): **0.574 ± 0.097**.

| cluster id | scope  | members |  mean | min   | max   | label |
| ---------- | ------ | ------: | ----: | ----- | ----- | ----- |
| `bd8d2148` | ids    |      55 | 0.726 | 0.395 | 0.987 | Transport Solver Profile Derivatives |
| `2f0dcab5` | ids    |      34 | 0.655 | 0.392 | 0.955 | — |
| `86df219b` | ids    |      19 | 0.778 | 0.481 | 0.960 | Gyrokinetic Eigenmode Fluxes Moments |
| `9fccb3b6` | ids    |      16 | 0.851 | 0.685 | 0.957 | — |
| `1c3f9b21` | domain |      16 | 0.851 | 0.685 | 0.957 | — |
| `a2a25e05` | domain |      14 | 0.860 | 0.701 | 0.966 | Radiation Emissivity and Radiated Power Profiles |
| `55a449ab` | global |      14 | 0.860 | 0.701 | 0.966 | — |
| `cf9e9745` | ids    |      14 | 0.860 | 0.701 | 0.966 | Plasma Radiation Emissivity and Power |
| `42561efc` | ids    |      12 | 0.843 | 0.728 | 0.966 | Edge Transport Radial Diffusion Coefficients GGD |
| `2a088219` | global |      12 | 0.843 | 0.728 | 0.966 | Radial Diffusivity GGD Values All Species |

Top-10 cluster overall mean: **0.813** — ≈ 2.5σ above baseline.

### Cluster label ↔ member cosine similarity (100 clusters, ≤5 members each)

| stat  | value |
| ----- | ----- |
| mean  | 0.720 |
| min   | 0.456 |
| max   | 0.858 |
| below 0.5 | 3 / 100 |
| below 0.6 | 10 / 100 |

### Per-concept top-5 vector search

| rank | id | physics domain | score |
| ---- | -- | -------------- | ----- |
| **Electron density** — `electron_density_at_separatrix` [general], `electron_density_at_divertor_target` [divertor_physics], `electron_radiated_power_density` [transport], `radial_component_of_electron_energy_flux` [transport], `fast_electron_number_density` [transport] | | | scores 0.889 – 0.873 |
| **Safety factor** — `safety_factor_at_magnetic_axis` [equilibrium] top (0.911), `normalized_poloidal_magnetic_flux_of_minimum_safety_factor` [equilibrium] (0.895), then three magnetic-field companions | | | |
| **Plasma current** — `toroidal_plasma_current_due_to_non_inductive_current_drive` [current_drive] top (0.936), `plasma_current` [plasma_control] (0.921), three more current-family SNs (scores ≥ 0.914) spread over 5 domains | | | |
| **Parallel current density** — `radial_component_of_diamagnetic_current_density` [transport] top (0.912); four `parallel_*` / conductivity neighbours | | | |
| **Magnetic axis** — `radial_component_of_magnetic_field` [magnetic_field_systems] (0.910), `major_radius_of_x_point` (0.908), `major_radius_of_magnetic_axis` (0.895), then two more field components | | | |

## Anomalies / findings

1. **StandardNameSource embeddings are unpopulated.** The
   `standard_name_source_desc_embedding` index is ONLINE at 100%
   (because zero nodes have the property — the index accepts the empty
   set as "fully populated"). All 6336 `StandardNameSource` nodes have
   `embedding IS NULL`. Any code path that relies on vector search over
   sources (e.g. retrieval of peer descriptions during compose) is
   currently a no-op. **Recommend**: add a CLI step / worker-pool task
   to embed `StandardNameSource.description`.
2. **122 StandardNames without embeddings are all unenriched.** Every
   one is in `review_status='named'` with both `description` and
   `documentation` NULL — embedding happens downstream in the enrich
   stage, so this is expected pipeline state, not a regression.
3. **`plasma_current` concept spans 5 physics domains** in the top-5
   search (current_drive, equilibrium, plasma_control, transport,
   waves). The vector search is semantically coherent (every hit is a
   plasma-current quantity) but the `physics_domain` taxonomy splits
   related current concepts very finely. Flagged as xfail with a
   pointer to this document.
4. **Short queries are unstable.** Bare keywords ("electron density",
   "safety factor") return different top-1 results than natural-language
   sentences — e.g. "electron density" ranks
   `shattered_pellet_species_number_density` top-1 while the sentence
   form correctly ranks `electron_density_at_separatrix` top-1. The
   tests standardise on the sentence form to reflect how the MCP tool
   is actually called.
5. **No documentation-free SN surfaces in search.** Because the
   indexed property is the embedding of `description` (absent on 122
   named-only nodes), those SNs are invisible to vector search. Users
   searching for them by concept will get no result. Consider fallback
   keyword matching on `id` tokens, or embed early on `id` + grammar
   fields.

## Recommendations

- **Embed StandardNameSource.description** so Test C can run meaningfully
  and compose-time neighbour retrieval can function.
- **Consider embedding at `named` state** using `id` tokens (they carry
  strong physics signal once run through the grammar decomposer) so
  unenriched SNs are at least partially searchable.
- **Revisit physics_domain granularity** for current-family SNs — or
  add a coarser `physics_concept` tag — so concept-level coherence
  checks (Test A2) can be strengthened beyond the modal-domain
  relaxation.
- Keep the `integration`-marked suite opt-in for unit runs but wire it
  into the nightly / post-deploy CI job to catch regressions in
  embedding quality or index population.

## Test commands

```bash
# Full vector-semantics suite (requires live graph + embedding server):
uv run pytest tests/standard_names/test_vector_semantics.py -v

# Only when explicitly running integration tests:
uv run pytest tests/standard_names -m integration

# Unit-only SN tests (default -m "not slow" still runs them):
uv run pytest tests/standard_names
```
