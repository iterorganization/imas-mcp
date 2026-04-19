# DD-06 — Spelling inconsistency `thermalisation` vs `thermalization`

## DD path(s) / pattern

Both spellings coexist under `distributions/distribution/`:

```
distributions/distribution/global_quantities/thermalisation/{particles,power,torque}
distributions/distribution/profiles_1d/thermalisation/{energy,momentum_phi,momentum_tor,particles}

distributions/distribution/global_quantities/thermalization/{particles,power,torque}
distributions/distribution/profiles_1d/thermalization/{energy,momentum_phi,particles}
```

Affected IDS: `distributions`.

Path counts: **9 paths** with `thermalisation`, **8 paths** with `thermalization`
— both with identical structure and units, differing only in spelling.

## Evidence

Rc13 standard names that quarantined because of this issue:

- `heating_power_due_to_thermalization_of_fast_particles` (audit: pydantic:scalar.name; unit: W) — `thermalization_of_fast_particles` is not in the Process vocabulary
- `thermal_particle_number_density_source_due_to_thermalization_of_fast_particles` (audit: pydantic:scalar.name; unit: m^-3.s^-1) — same grammar gap

Total: **2 quarantined names** directly attributable to the vocabulary gap for
`thermalization_of_fast_particles`. Two additional valid names use the same root
but pass because they use a shorter form:

- `thermal_energy_source_rate_density_due_to_thermalization` (unit: m^-3.W; vs: valid)
- `toroidal_angular_momentum_source_rate_density_due_to_thermalization` (unit: m^-2.N; vs: valid)

The spelling inconsistency complicates vocabulary work: should ISN add
`thermalization` or `thermalisation` as the Process token? The DD should settle
on one spelling so the grammar can match it.

## Proposed fix

**Standardize on `thermalization`** (American English, -ize suffix). Rationale:

1. IMAS uses American English conventions throughout (e.g., `ionization` not
   `ionisation`, `polarization` not `polarisation`).
2. The IUPAC recommendation is `-ize` for technical terms derived from Greek
   (`-ίζειν`).
3. `thermalization` already appears in the standard names vocabulary and in
   physics textbooks (Spitzer, NRL Plasma Formulary).

Actions:

1. Rename all `thermalisation` paths to `thermalization` in the DD.
2. Add `RENAMED_TO` records from the old paths to the new paths.
3. Mark `thermalisation` paths as `obsolescent` for one DD version before
   removal.

## Downstream impact

- **imas-codex SN extractor:** unblocks 2 quarantined names once
  `thermalization_of_fast_particles` is added to the ISN Process vocabulary
  (separate ISN issue). The DD spelling fix ensures only one spelling needs
  to be in the vocabulary.
- **pydantic audits:** no audit changes needed — the grammar vocabulary gap
  is the root cause, not an audit bug.
- **imas-standard-names:** can add `thermalization` (and
  `thermalization_of_fast_particles`) as a Process token without worrying
  about the British variant leaking through from the DD.

## References

- rc13 review: plans/research/standard-names/12-full-graph-review.md §2.13 (row 10)
- rc13 review: plans/research/standard-names/12-full-graph-review.md §4.3 (DD-05)
- Plan: plans/features/standard-names/31-quality-bootstrap-v2.md §7
