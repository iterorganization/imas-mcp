## Curated Enrichment Exemplars

These paired exemplars teach by contrast. Study the *reasons* — the weak
variants look plausible until you see the canonical form side by side. All
exemplars use American spelling (NC-17) in every prose field.

### Positive exemplars — imitate these patterns

#### E1. Physical-base scalar with governing equation

- **Name:** `electron_temperature` (unit `eV`)
- ✅ *Description:* `Thermal energy of the bulk electron population expressed as a temperature via $T_e = 2 \langle E_e \rangle / (3 k_B)$.`
- ✅ *Documentation excerpt:* `The bulk electron temperature quantifies the second
  moment of the electron velocity distribution under the Maxwellian assumption.
  $T_e$ is routinely inferred from Thomson scattering, electron cyclotron emission,
  and soft-X-ray tomography; typical core values span 1–20 keV in tokamak plasmas
  and 0.1–1 keV in the scrape-off layer. See also $p_e = n_e k_B T_e$
  (`electron_pressure`) and the kinetic profile family `electron_density`,
  `electron_pressure_gradient`.`
- *Why good:* one-sentence description states the physical meaning, not a
  tokenization of the name. Documentation covers (physics, measurement,
  range, relation to neighbouring SNs), defines every symbol, links two
  siblings, uses American spelling, no DD path leaked.

#### E2. Vector component with COCOS sign convention

- **Name:** `toroidal_component_of_magnetic_field` (unit `T`)
- ✅ *Description:* `Projection of the plasma magnetic-field vector onto the geometric toroidal direction $\hat{\phi}$.`
- ✅ *Documentation:* `The toroidal component $B_\phi = \mathbf{B} \cdot \hat{\phi}$
  dominates the total field magnitude in tokamak configurations and sets the
  ion gyrofrequency $\Omega_{ci} = Z_i e B_\phi / m_i$. Positive when $B_\phi$
  is in the $+\phi$ direction under COCOS-11 (counter-clockwise viewed from
  above). Measured by fibre-optic polarimetry or inferred from equilibrium
  reconstruction; typical on-axis values 1–6 T for current tokamaks. See also
  `poloidal_component_of_magnetic_field`, `magnetic_field_magnitude`.`
- *Why good:* sign-convention sentence begins literally with "Positive when"
  (satisfies the sanitizer); COCOS number is explicit; unit `T` is consistent
  with described physics; cross-refs target existing siblings.

#### E3. Constraint / measurement-weight path

- **Name:** `flux_loop_radial_magnetization_constraint_weight` (unit `1`)
- ✅ *Description:* `Relative weight applied to the radial magnetization signal from flux loops in the least-squares equilibrium cost functional.`
- ✅ *Documentation:* `Standard $\chi^2$ weight controlling the relative
  importance of this measurement in the equilibrium reconstruction. A value of
  1 applies the nominal measurement variance; larger values up-weight the
  constraint when the raw variance under-represents modelling error. The
  companion quantity is `flux_loop_radial_magnetization_constraint_measurement`.`
- *Why good:* uses the **one-line χ² reference** for the generic inverse-problem
  role (never re-derives it); focuses on WHAT the weight does, not the
  underlying physics of flux loops (already captured in the base name).

#### E4. Spectrum quantity — closure integral explicit

- **Name:** `toroidal_mode_spectrum_of_electron_temperature_perturbation`
  (unit `eV`)
- ✅ *Description:* `Fourier-mode amplitude of the electron-temperature perturbation along the toroidal coordinate.`
- ✅ *Documentation:* `Amplitude $|\tilde{T}_e(n_\phi)|$ of the complex Fourier
  decomposition of $\delta T_e$ with respect to the toroidal angle $\phi$.
  Integrating $|\tilde{T}_e|^2$ over toroidal mode number $n_\phi$ recovers the
  total perturbation variance. Typical $n_\phi$ ranges 0–50 for MHD-relevant
  modes; infinitesimal fluctuations resolved by ECE radiometer or soft X-ray
  arrays. See `electron_temperature`, `poloidal_mode_spectrum_of_*` siblings.`
- *Why good:* spelled-out closure integral (SR-spectrum rule — required when
  name ends in `_spectrum`); mode-number symbol defined; unit `eV` matches
  the amplitude (not a density — note: `_density_spectrum` would require a
  `eV/n_\phi` unit with the Parseval integral over $n_\phi$).

#### E5. Profile with typical-value range and diagnostic

- **Name:** `safety_factor` (unit `1`)
- ✅ *Description:* `Number of toroidal transits a field line makes per single poloidal circuit of a given flux surface.`
- ✅ *Documentation:* `The safety factor $q(\psi) = d\Phi_{\rm tor} / d\Phi_{\rm pol}$
  quantifies magnetic-field-line helicity and controls MHD stability. In conventional
  H-mode tokamaks $q_0 \approx 1$ on axis and $q_{95} = 3$–5 at the 95% flux
  surface; $q < 1$ regions drive sawtooth instabilities. Reconstructed from
  external magnetics plus motional-Stark-effect or polarimetry measurements.
  See `poloidal_magnetic_flux`, `toroidal_magnetic_flux`, `magnetic_shear`.`
- *Why good:* defines $q$ symbolically in terms of defined quantities; gives
  physically meaningful numeric ranges anchored to reactor-relevant scenarios;
  notes the stability connection; links to both flux variables it derives from.

#### E6. Geometric property — concise, no DD leak

- **Name:** `major_radius_of_magnetic_axis` (unit `m`)
- ✅ *Description:* `Major-radius coordinate of the magnetic axis, i.e. the extremum of the poloidal magnetic flux surface.`
- ✅ *Documentation:* `The major radius $R_{\rm axis}$ of the magnetic axis is
  the $R$ coordinate at which $\nabla \psi = 0$ inside the plasma, identifying
  the innermost nested flux surface. Typical values 1.6–3.0 m for medium-size
  tokamaks (JET, DIII-D) and 6.2 m for ITER. Obtained from equilibrium
  reconstruction (EFIT, LIUQE); pair with `vertical_coordinate_of_magnetic_axis`.`
- *Why good:* Definition anchored to a mathematical condition ($\nabla \psi = 0$);
  numeric ranges span multiple devices for context; pairs the coordinate with
  its Z-counterpart (NC R/Z consistency); no DD path mentioned.

### Anti-patterns — never emit these

#### AE1. Tautological description

- **Name:** `electron_temperature`
- ❌ `The temperature of the electrons.`
- *Why bad:* the description does not add information beyond the name tokens.
- *Fix:* describe the **physical meaning** (see E1 above — moment of the
  velocity distribution, units expressed via Boltzmann's constant).

#### AE2. DD-path leakage

- ❌ `Quantity stored at core_profiles/profiles_1d/electrons/temperature.`
- *Why bad:* standard names are IDS-agnostic. Referring to DD structure couples
  the reader to the source rather than the physics.
- *Fix:* describe the physics and its measurement; never mention IDS or DD
  paths anywhere in description, documentation, or tags.

#### AE3. Undefined LaTeX symbols

- ❌ `The safety factor $q = B_\phi R / (r B_\theta)$ ...` (no definition of
  $r$ or $B_\theta$)
- *Why bad:* introducing symbols without definition turns the documentation
  into shorthand; readers must consult other sources to parse it.
- *Fix:* `... where $r$ is the minor radius and $B_\theta$ is the poloidal
  magnetic-field magnitude on the given flux surface.`

#### AE4. Bracketed placeholder sign-convention

- ❌ `Positive when [condition].` or `Positive when the quantity is in the
  [direction] direction under COCOS-[N].`
- *Why bad:* the sanitizer will strip the entire sentence; the documentation
  loses the sign-convention payload required for vector and flux quantities.
- *Fix:* write the concrete condition — e.g. `Positive when $B_\phi$ is in
  the $+\phi$ direction under COCOS-11.` If the quantity is sign-invariant,
  omit the sentence entirely.

#### AE5. British spelling in prose

- ❌ `The normalised poloidal flux coordinate, used to parametrise flux
  surfaces ...`
- *Why bad:* the catalog is American-English only. British spellings become
  silent synonyms that break grep-based search and downstream consumers.
- *Fix:* `The normalized poloidal flux coordinate, used to parameterize flux
  surfaces ...`. Apply the same rule to `ionization`, `polarized`, `center`,
  `behavior`, `modeled`, `analyzed`, `fueling`, `color`, `fiber`, `meter`.
  SI unit symbols (`m`, `kg`, `V`) are unaffected.

#### AE6. Boilerplate inverse-problem derivation

- **Name:** `iron_core_segment_radial_magnetization_constraint_weight`
- ❌ `The iron core segment radial magnetization constraint weight is a
  parameter used in an inverse problem, where the forward model is ... and
  the cost function is $\chi^2 = \sum_i (y_i - F_i(x))^2 / \sigma_i^2 / w_i$.
  By adjusting this weight, the reconstruction emphasises the measurement
  more or less relative to other constraints ...`
- *Why bad:* every constraint-weight name gets the same two-paragraph
  derivation; the text is boilerplate, not physics, and bloats the catalog.
- *Fix:* single-line reference: `Standard $\chi^2$ weight controlling the
  relative importance of the iron-core segment radial magnetization
  measurement in the equilibrium reconstruction.` Link once to a canonical
  inverse-problem documentation target.

#### AE7. Documentation contradicts the name

- **Name:** `normal_component_of_magnetic_field` (claims scalar, unit `T`)
- ❌ *Doc says:* `The Fourier coefficients of the normal component of the
  magnetic field, expanded in poloidal mode number $m$ ...`
- *Why bad:* the name promises a scalar field-magnitude; the documentation
  describes a spectral coefficient. They disagree — either the name or the
  description is wrong.
- *Fix:* rename to `fourier_coefficient_of_normal_component_of_magnetic_field`
  (with unit `T` per-harmonic) **or** rewrite the documentation to describe
  the scalar field proper without Fourier language.

#### AE8. Tag proliferation / primary-tag leakage

- ❌ `tags: ["core-plasma", "transport", "edge-physics", "magnetic-field",
  "iter", "thomson-scattering", "measured", "reconstructed"]`
- *Why bad:* `transport`, `edge-physics`, `core-plasma` are **primary**
  physics domains managed by the pipeline — never in `tags`. Provenance
  (`measured`, `reconstructed`) is metadata, not a semantic tag. Device names
  (`iter`) are not tags.
- *Fix:* 2–5 lowercase hyphenated **secondary** tags — e.g.
  `["ece-diagnostic", "kinetic-profile", "flux-surface-average"]`.

#### AE9. Excessive / wrong cross-links

- ❌ `links: ["name:plasma_current", "name:plasma", "name:current",
  "https://en.wikipedia.org/wiki/Safety_factor_(plasma_physics)"]`
- *Why bad:* bare nouns like `plasma` and `current` are not valid SN ids;
  Wikipedia URLs are external noise; `plasma_current` is only weakly
  related to `safety_factor`.
- *Fix:* 2–6 bare-ID links to genuinely related **existing** SNs:
  `["poloidal_magnetic_flux", "toroidal_magnetic_flux", "magnetic_shear",
   "major_radius_of_magnetic_axis"]` (verify each exists in the catalog).

#### AE10. Generic filler prose

- ❌ `This is an important quantity in fusion research. It plays a
  significant role in plasma behaviour and is widely used ...`
- *Why bad:* zero information content; any three sentences must carry
  physics, measurement, or relational substance.
- *Fix:* every sentence earns its place with one of (definition, governing
  equation, typical-value range, diagnostic pathway, sign convention,
  cross-reference rationale).

### Pre-emit checklist for every enriched entry

1. Does the description add information beyond the name tokens?
2. Are all LaTeX symbols defined on first use, with units?
3. American spelling throughout every prose field?
4. For COCOS-dependent quantities: does the documentation contain a
   concrete `Positive when ...` sentence (no `[...]` placeholders)?
5. For spectrum quantities: is the closure integral / Parseval relation
   spelled out with the correct integration variable?
6. For constraint weights / measurement times: is the generic inverse-problem
   role referenced in one line rather than re-derived?
7. Do `tags` contain **only** secondary tags (no primary physics domains,
   no devices, no provenance verbs)?
8. Are `links` **bare SN ids** of existing names that genuinely enrich
   understanding (typically 2–6)?
9. Is the `description` ≤ 180 characters and the `documentation`
   ≥ 3 sentences?
10. Does the documentation avoid any mention of DD paths, IDS names, or
    structural database prefixes?
