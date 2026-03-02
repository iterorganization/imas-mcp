# DD v4 COCOS 17 Validation & Transformation Labels

> Exhaustive validation that IMAS Data Dictionary v4 complies with COCOS 17,
> and analysis of the COCOS transformation label system across DD versions.

**Reference:** Sauter & Medvedev, *CPC* **184** (2013) 293–302.

See also: [COCOS module documentation](cocos.md) for the `imas_codex.cocos`
API and [IMAS DD architecture](imas_dd.md) for graph schema details.

---

## 1. COCOS 17 Definition

COCOS is a single integer encoding four independent sign/normalization choices.
DD v3.x used COCOS 11; DD v4.x declares COCOS 17 in the top-level `<cocos>`
XML element. The two differ in exactly one parameter:

| Parameter | Symbol | COCOS 11 | COCOS 17 | Meaning |
|-----------|--------|----------|----------|---------|
| Poloidal flux sign | $\sigma_{B_p}$ | $+1$ | $-1$ | $\psi$ increasing (+1) or decreasing (−1) from axis to edge with positive $I_p$ |
| $\psi$ normalization | $e_{B_p}$ | 1 | 1 | Full $\psi$ (Wb), not $\psi/(2\pi)$ |
| Cylindrical handedness | $\sigma_{R\phi Z}$ | $+1$ | $+1$ | $(R,\phi,Z)$ right-handed, $\phi$ CCW from above |
| Poloidal handedness | $\sigma_{\rho\theta\phi}$ | $+1$ | $+1$ | $(\rho,\theta,\phi)$ right-handed, $\theta$ increasing CW in poloidal plane |

The **sole difference** is $\sigma_{B_p}$: flipped from $+1$ to $-1$. This
means $\psi$ decreases outward in COCOS 17 (for positive $I_p$ and $B_0$),
i.e. $\psi_{\text{axis}} > \psi_{\text{edge}}$.

---

## 2. Validation Evidence

Six independent lines of evidence were checked against 46,968 active DD v4
paths. Zero contradictions were found.

### 2.1 Toroidal angle $\phi$ — confirms $\sigma_{R\phi Z} = +1$

**257 paths** reference toroidal angle direction. Representative documentation
from the DD XML:

> *"Toroidal angle (oriented counter-clockwise when viewing from above)"*

This is the defining property of $\sigma_{R\phi Z} = +1$: the $(R,\phi,Z)$
system is right-handed with $\phi$ increasing counter-clockwise when viewed
from above.

### 2.2 Poloidal angle $\theta$ — confirms $\sigma_{\rho\theta\phi} = +1$

**30 paths** reference poloidal angle conventions. The DD defines $\theta$ as
increasing clockwise in the poloidal plane (when viewed with $\phi$ pointing
out of the page). Representative documentation:

> *"Poloidal angle (oriented clockwise when viewing the poloidal cross section
> from the low field side, such that theta=0 corresponds to the outboard
> midplane)"*

With $\phi$ CCW from above ($\sigma_{R\phi Z} = +1$), a CW poloidal angle
gives a right-handed $(\rho,\theta,\phi)$ system, confirming
$\sigma_{\rho\theta\phi} = +1$.

### 2.3 Poloidal flux $\psi$ units — confirms $e_{B_p} = 1$

**348 paths** carry units of `Wb` (Webers) for poloidal flux. **Zero** paths
use `Wb/rad` ($= \psi/(2\pi)$). This confirms $e_{B_p} = 1$: the DD uses
full $\psi$, not the $\psi/(2\pi)$ normalization of COCOS 1–8.

### 2.4 Poloidal flux sign — confirms $\sigma_{B_p} = -1$

The imas-python `_3to4_sign_flip_paths` dictionary lists **63 paths** across
22 IDS that require sign inversion ($\times -1$) when converting from DD v3
(COCOS 11, $\sigma_{B_p} = +1$) to DD v4 (COCOS 17, $\sigma_{B_p} = -1$).
**All 63 are $\psi$-related** — fields named `psi`, `psi_boundary`,
`psi_magnetic_axis`, `ggd/psi/values`, or `psi_external_average`.

Additionally, the DD v3 XML labels **115 paths** with
`cocos_label_transformation="psi_like"` or `"dodpsi_like"`, all of which
also require the sign flip. The total set of $\psi$-sensitive paths is **178**
(see [Section 5.4](#54-sign-flip-coverage-gap)).

No non-$\psi$ fields require sign changes, which is exactly what flipping only
$\sigma_{B_p}$ predicts.

### 2.5 Safety factor $q$ — confirms no sign change

The safety factor transforms as:

$$q_{\text{out}} = \frac{\sigma_{\rho\theta\phi,\text{out}} \cdot \sigma_{R\phi Z,\text{out}}}{\sigma_{\rho\theta\phi,\text{in}} \cdot \sigma_{R\phi Z,\text{in}}} \cdot q_{\text{in}}$$

Since both $\sigma_{\rho\theta\phi}$ and $\sigma_{R\phi Z}$ are identical
between COCOS 11 and 17, the factor is $+1$. Consistently, $q$ does **not**
appear in the sign-flip paths. The path
`equilibrium/time_slice/global_quantities/q_min/psi` does appear — but this is
the **$\psi$ value at $q_{\min}$** (a psi field), not $q$ itself.

### 2.6 Sauter Equation 23 algebraic consistency

Equation 23 from Sauter & Medvedev relates the COCOS parameters:

$$q = \frac{\sigma_{B_p}}{2\pi \cdot e_{B_p}} \oint \frac{\vec{B} \cdot \nabla\phi}{\vec{B} \cdot \nabla\theta} \, d\theta \cdot \sigma_{\rho\theta\phi}$$

For **all four physical scenarios** (signs of $I_p$ and $B_0$), COCOS 17
parameters yield $q > 0$, confirming internal consistency:

| Scenario | $I_p$ | $B_0$ | $\psi$ gradient | $q$ sign |
|----------|-------|-------|-----------------|----------|
| $I_p > 0, B_0 > 0$ | $+$ | $+$ | $\psi$ decreasing outward | $q > 0$ ✓ |
| $I_p > 0, B_0 < 0$ | $+$ | $-$ | $\psi$ decreasing outward | $q > 0$ ✓ |
| $I_p < 0, B_0 > 0$ | $-$ | $+$ | $\psi$ increasing outward | $q > 0$ ✓ |
| $I_p < 0, B_0 < 0$ | $-$ | $-$ | $\psi$ increasing outward | $q > 0$ ✓ |

### 2.7 Exhaustive search for contradictions

A graph query across all 46,968 active paths searched documentation fields for
any mention of conflicting conventions (e.g., "cocos=11", "clockwise toroidal",
"ψ increasing outward"). The **sole match** was
`magnetics/b_field_phi_probe/toroidal_angle`, which references "cocos=11
phi-like angle" — a benign description of how the probe angle relates to the
COCOS 11 convention, not a declaration that the field _uses_ COCOS 11.

The `edge_profiles/ggd/psi` path explicitly documents a surface normal
definition ("upward"), consistent with the COCOS 17 $\psi$ gradient direction.

### 2.8 Validation summary

| Evidence | Paths checked | COCOS 17 parameter confirmed | Contradictions |
|----------|--------------|------------------------------|----------------|
| $\phi$ direction | 257 | $\sigma_{R\phi Z} = +1$ | 0 |
| $\theta$ direction | 30 | $\sigma_{\rho\theta\phi} = +1$ | 0 |
| $\psi$ units | 348 | $e_{B_p} = 1$ | 0 |
| $\psi$ sign flip | 178 | $\sigma_{B_p} = -1$ | 0 |
| $q$ invariance | 36 | Consistent with $\sigma_{\rho\theta\phi} = \sigma_{R\phi Z}$ | 0 |
| Eq. 23 algebra | 4 scenarios | All yield $q > 0$ | 0 |
| Full-text search | 46,968 | No conflicting conventions | 0 |

**Conclusion:** DD v4 is fully consistent with COCOS 17. The $\psi$ sign
change from COCOS 11 is the only physical difference, and it is correctly
implemented across all 178 affected paths.

---

## 3. COCOS Transformation Labels in the DD XML

The DD XML annotates COCOS-sensitive fields with metadata that describes how
they transform under COCOS convention changes. This section analyzes the label
system, its coverage, and changes between DD v3 and v4.

### 3.1 XML attributes

Each COCOS-sensitive field carries up to five XML attributes:

| Attribute | Purpose | Example |
|-----------|---------|---------|
| `cocos_label_transformation` | Class label grouping fields by transform behavior | `psi_like` |
| `cocos_transformation_expression` | Algebraic transformation factor | `.fact_psi` |
| `cocos_leaf_name_aos_indices` | Qualified path template with AoS indices | `equilibrium.time_slice{i}.profiles_1d.psi` |
| `cocos_alias` | Template substitution marker | `IDSPATH` or `bpol` |
| `cocos_replace` | Context path for template expansion | `equilibrium.time_slice{i}.profiles_1d` |

The `cocos_label_transformation` groups fields into **transformation classes**
— sets of fields that all transform identically under any COCOS change. The
`cocos_transformation_expression` gives the formula in terms of the four
fundamental COCOS factors from Sauter Table 2.

### 3.2 Complete label inventory

The table below lists all 18 labels ever used in DD v3.22.0–v4.1.1, their
transformation expressions, the Sauter factor they encode, and path counts in
the last v3 and latest v4 releases.

| Label | Expression | Physics | Sauter Factor | v3 Paths | v4 Paths | v3→v4 |
|-------|-----------|---------|---------------|----------|----------|-------|
| `psi_like` | `.fact_psi` | $\psi$ and related | $\frac{\sigma_{B_p,\text{out}} (2\pi)^{1-e_{B_p,\text{out}}}}{\sigma_{B_p,\text{in}} (2\pi)^{1-e_{B_p,\text{in}}}}$ | 112 | 0 | Removed |
| `dodpsi_like` | `.fact_dodpsi` | $dX/d\psi$ derivatives | $(\texttt{psi\_like})^{-1}$ | 12 | 0 | Removed |
| `ip_like` | `.sigma_ip_eff` | $I_p$, driven currents | $\frac{\sigma_{R\phi Z,\text{out}} \cdot \sigma_{B_p,\text{out}}}{\sigma_{R\phi Z,\text{in}} \cdot \sigma_{B_p,\text{in}}}$ | 80 | 86 | Retained |
| `b0_like` | `.sigma_b0_eff` | $B_0$, toroidal field | $\frac{\sigma_{R\phi Z,\text{out}}}{\sigma_{R\phi Z,\text{in}}}$ | 99 | 90 | Retained |
| `tor_angle_like` | `.sigma_rphiz_eff` | $\phi$, toroidal angles | $\frac{\sigma_{R\phi Z,\text{out}}}{\sigma_{R\phi Z,\text{in}}}$ | 190 | 103 | Retained |
| `pol_angle_like` | `.fact_dtheta` | $\theta$, poloidal angles | $\frac{\sigma_{\rho\theta\phi,\text{out}}}{\sigma_{\rho\theta\phi,\text{in}}}$ | 12 | 10 | Retained |
| `q_like` | `.fact_q` | Safety factor $q$ | $\frac{\sigma_{\rho\theta\phi,\text{out}} \cdot \sigma_{R\phi Z,\text{out}}}{\sigma_{\rho\theta\phi,\text{in}} \cdot \sigma_{R\phi Z,\text{in}}}$ | 33 | 36 | Retained |
| `one_like` | `'1'` | COCOS-insensitive | $1$ | 10 | 8 | Retained |
| `grid_type_dim1_like` | `grid_type_transformation(…,1)` | Grid dim 1 | Grid-type dependent | 42 | 45 | Retained |
| `grid_type_dim2_like` | `grid_type_transformation(…,2)` | Grid dim 2 | Grid-type dependent | 42 | 45 | Retained |
| `grid_type_dim1_dim1_like` | `.fact_dim1*.fact_dim1` | Metric $g^{11}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_dim1_dim2_like` | `.fact_dim1*.fact_dim2` | Metric $g^{12}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_dim1_dim3_like` | `.fact_dim1*.fact_dim3` | Metric $g^{13}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_dim2_dim2_like` | `.fact_dim2*.fact_dim2` | Metric $g^{22}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_dim2_dim3_like` | `.fact_dim2*.fact_dim3` | Metric $g^{23}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_dim3_dim3_like` | `.fact_dim3*.fact_dim3` | Metric $g^{33}$ | Dim factor products | 6 | 0 | Removed |
| `grid_type_tensor_covariant_like` | `grid_type_transformation(…,4)` | Covariant tensor | Grid-type dependent | 6 | 6 | Retained |
| `grid_type_tensor_contravariant_like` | `grid_type_transformation(…,4)` | Contravariant tensor | Grid-type dependent | 6 | 6 | Retained |
| | | | **Total** | **680** | **435** | **−245** |

### 3.3 COCOS 11 → 17 transform factors

Between COCOS 11 and 17, only $\sigma_{B_p}$ changes. The remaining three
parameters ($e_{B_p}$, $\sigma_{R\phi Z}$, $\sigma_{\rho\theta\phi}$) are
identical. This means most labels have a transform factor of exactly $+1$:

| Label | Factor (11 → 17) | Explanation |
|-------|-------------------|-------------|
| `psi_like` | $-1$ | $\sigma_{B_p}$ flips |
| `dodpsi_like` | $-1$ | Inverse of `psi_like`, but $(-1)^{-1} = -1$ |
| `ip_like` | $-1$ | $\sigma_{R\phi Z} \cdot \sigma_{B_p}$ — $\sigma_{B_p}$ flips |
| `b0_like` | $+1$ | Only depends on $\sigma_{R\phi Z}$, unchanged |
| `tor_angle_like` | $+1$ | Only depends on $\sigma_{R\phi Z}$, unchanged |
| `pol_angle_like` | $+1$ | Only depends on $\sigma_{\rho\theta\phi}$, unchanged |
| `q_like` | $+1$ | $\sigma_{\rho\theta\phi} \cdot \sigma_{R\phi Z}$, both unchanged |
| `one_like` | $+1$ | Always 1 |

For general COCOS conversions (e.g., COCOS 17 → COCOS 2), many more labels
produce non-trivial factors.

---

## 4. What Changed Between v3 and v4

Eight labels were removed, falling into two distinct groups.

### 4.1 `psi_like` and `dodpsi_like` — convention made definitional

In DD v3 (COCOS 11), the $\psi$ sign was one of four variable COCOS
parameters. In DD v4 (COCOS 17), the sign is **baked into the definition**:
$\psi$ always decreases outward (for standard field/current orientation).

Since the convention is no longer variable within v4, the DD maintainers
removed the `psi_like` and `dodpsi_like` labels — eliminating 124 labeled
paths. The imas-python v3→v4 converter reads these labels from the **v3 XML**
at conversion time, so their absence from v4 does not break version conversion.

Of the 112 `psi_like` paths in v3, 91 still exist in v4 (with the label
removed) and 6 were structurally deleted. None received a replacement label.

### 4.2 Metric tensor components — structural removal

The individual metric tensor components (`g11_contravariant`,
`g12_contravariant`, …, `g33_contravariant`) were replaced in v4 by generic
`tensor_contravariant` / `tensor_covariant` array fields with the simpler
`grid_type_tensor_*_like` labels. The six compound cross-term labels
(`grid_type_dim1_dim1_like` through `grid_type_dim3_dim3_like`) disappeared
because the fields they described no longer exist — 36 paths removed.

### 4.3 Path count changes in retained labels

The remaining path count changes are due to IDS restructuring, not label
policy:

| Label | v3 → v4 | Cause |
|-------|---------|-------|
| `tor_angle_like` | 190 → 103 (−87) | `summary` IDS restructured (−81 paths) |
| `b0_like` | 99 → 90 (−9) | Deprecated field removal |
| `pol_angle_like` | 12 → 10 (−2) | Deprecated field removal |
| `one_like` | 10 → 8 (−2) | Deprecated field removal |
| `ip_like` | 80 → 86 (+6) | New current-related fields added |
| `q_like` | 33 → 36 (+3) | New safety factor fields added |
| `grid_type_dim1_like` | 42 → 45 (+3) | New grid fields added |
| `grid_type_dim2_like` | 42 → 45 (+3) | New grid fields added |

---

## 5. Assessment

### 5.1 Was removing `psi_like` from v4 a mistake?

**Not exactly, but it creates an asymmetry.** The retained labels (`ip_like`,
`b0_like`, `tor_angle_like`, `pol_angle_like`, `q_like`) all have transform
factor $+1$ between COCOS 11 and 17 — they didn't need to change for the
v3→v4 transition. They were kept because they remain useful for
**general COCOS-to-COCOS conversion**. By removing `psi_like` — the one label
that _did_ change — the DD created a gap: every COCOS-sensitive field class
is labeled except the most important one.

The logic appears to have been: "since v4 fixes the $\psi$ sign, the label is
redundant." But this reasoning is not consistently applied — v4 also fixes
$\phi$ direction (CCW from above), $\theta$ direction (CW in poloidal plane),
and $I_p$ sign convention, yet `tor_angle_like`, `pol_angle_like`, and
`ip_like` were all retained. The labels exist to describe **how fields
transform under arbitrary COCOS changes**, not just the v3→v4 transition.

**Purpose 1 — DD version conversion (v3↔v4):** Removing `psi_like` from v4 is
correct. The imas-python converter reads labels from the source (v3) XML.

**Purpose 2 — General COCOS conversion within v4:** Removing them is a loss.
Converting v4 data from COCOS 17 to COCOS 2 requires knowing which fields are
$\psi$-sensitive. All other field classes have machine-readable markers; $\psi$
does not. The sign is documented in prose but there is no programmatic way to
discover which of the 46,968 active paths carry $\psi$ semantics.

### 5.2 Functions beyond COCOS mapping

The labels serve four functions beyond their primary COCOS conversion role:

1. **Field classification** — grouping fields by physics behavior (all
   $\psi$-related fields share a label, enabling queries like "find all
   flux-related quantities")
2. **Conversion documentation** — the expression tells you exactly how to
   transform a field, without needing to understand the full Sauter formalism
3. **Validation** — a field's units and documentation can be checked for
   consistency with its label
4. **Semantic enrichment** — a `psi_like` label immediately tells downstream
   tools that a field is flux-related, aiding search and classification

### 5.3 Usage in imas-python

The labels have exactly **one programmatic use** in imas-python — in
`ids_convert.py:_apply_3to4_conversion()`. This function searches the v3 XML
for `psi_like`/`dodpsi_like`-labeled fields and registers sign-flip
post-processors. The `cocos_transformation_expression` values (`.sigma_b0_eff`,
`.fact_psi`, etc.) are **never evaluated** — they exist as declarative metadata
only.

### 5.4 Sign-flip coverage gap

The v3→v4 converter uses **two disjoint sources** for $\psi$ sign-flip paths,
with zero overlap:

| Source | Paths | Coverage |
|--------|-------|----------|
| DD XML `psi_like` + `dodpsi_like` labels | 115 | Profile arrays, constraint positions, error bounds |
| Hardcoded `_3to4_sign_flip_paths` | 63 | `psi_boundary`, `psi_magnetic_axis`, standalone $\psi$ values |
| **Total** | **178** | |

The hardcoded list catches 63 paths that the DD XML editors never annotated
with the `psi_like` label, including critical fields like `psi_boundary`,
`psi_magnetic_axis`, and `ggd/psi/values`. Without this fallback, these paths
would have incorrect signs after v3→v4 conversion.

This means the XML labeling was **never comprehensive** for $\psi$ fields —
it required a hardcoded supplement covering 35% of $\psi$-sensitive paths.

---

## 6. Recommendations for the IMAS Data Dictionary

### 6.1 Restore `psi_like` and `dodpsi_like` labels in DD v4

The `psi_like` and `dodpsi_like` labels should be restored on all
$\psi$-sensitive fields in the DD v4 XML. The removal creates an inconsistency:
all other COCOS-sensitive field classes (`ip_like`, `b0_like`,
`tor_angle_like`, `pol_angle_like`, `q_like`) retain their labels in v4 despite
their conventions also being fixed by COCOS 17. The labels describe how fields
transform under **arbitrary COCOS changes** — a property that is independent
of which convention the DD itself adopts.

Without these labels, any tool performing general COCOS conversion on v4 data
(e.g., COCOS 17 → COCOS 2 for CHEASE compatibility) cannot programmatically
identify which fields require $\psi$ sign or normalization adjustments. The
prose documentation ("$\psi$ decreasing outward") is not machine-readable.

### 6.2 Close the labeling coverage gap

Even in DD v3, the `psi_like` label was applied to only 115 of the 178
$\psi$-sensitive paths. The remaining 63 were never annotated, requiring
imas-python to maintain a hardcoded fallback list (`_3to4_sign_flip_paths`).
When restoring labels in v4, all 178 paths should be labeled —
including `psi_boundary`, `psi_magnetic_axis`, and `ggd/psi/values` fields
that were previously unlabeled.

The affected paths are well-defined: they are the union of the v3
`psi_like`/`dodpsi_like`-labeled fields and the paths in imas-python's
`_3to4_sign_flip_paths` dictionary. A complete list is available from
imas-python:

```python
from imas.dd_zip import dd_etree
from imas.ids_convert import _3to4_sign_flip_paths

# Source 1: XML-labeled paths (115)
tree = dd_etree("3.42.2")
for ids_el in tree.findall("IDS"):
    for field in ids_el.iter("field"):
        label = field.get("cocos_label_transformation", "")
        if label in ("psi_like", "dodpsi_like"):
            print(f'{ids_el.get("name")}/{field.get("path")}')

# Source 2: hardcoded paths (63)
for ids_name, paths in _3to4_sign_flip_paths.items():
    for path in paths:
        print(f"{ids_name}/{path}")
```

### 6.3 Apply labels consistently to new fields

Going forward, any new field added to the DD that is COCOS-sensitive should
receive a `cocos_label_transformation` and `cocos_transformation_expression`
attribute. This applies particularly to $\psi$-related fields, which are the
most common COCOS-sensitive class (178 paths) and the one most likely to be
extended as new IDS are added or existing ones gain new $\psi$ grids.

### 6.4 Consider evaluating transformation expressions

The `cocos_transformation_expression` values (`.sigma_b0_eff`, `.fact_psi`,
etc.) are currently never evaluated by any code — they exist as documentation
embedded in XML attributes. If the DD provided a specification for how to
evaluate these expressions given a source and target COCOS, they could enable
a generic, schema-driven COCOS conversion engine that requires no hardcoded
path lists. This would eliminate the class of bugs where a new $\psi$-related
field is added to the DD but not to the hardcoded sign-flip list.

---

## Appendix: Transformation Factor Reference

For converting data between any two COCOS conventions, the transformation
factor for each label class is:

| Label | Factor ($\text{in} \to \text{out}$) |
|-------|-------------------------------------|
| `psi_like` | $\frac{\sigma_{B_p,\text{out}} \cdot (2\pi)^{1-e_{B_p,\text{out}}}}{\sigma_{B_p,\text{in}} \cdot (2\pi)^{1-e_{B_p,\text{in}}}}$ |
| `dodpsi_like` | $(\texttt{psi\_like})^{-1}$ |
| `ip_like` | $\frac{\sigma_{R\phi Z,\text{out}} \cdot \sigma_{B_p,\text{out}}}{\sigma_{R\phi Z,\text{in}} \cdot \sigma_{B_p,\text{in}}}$ |
| `b0_like` | $\frac{\sigma_{R\phi Z,\text{out}}}{\sigma_{R\phi Z,\text{in}}}$ |
| `tor_angle_like` | $\frac{\sigma_{R\phi Z,\text{out}}}{\sigma_{R\phi Z,\text{in}}}$ |
| `pol_angle_like` | $\frac{\sigma_{\rho\theta\phi,\text{out}}}{\sigma_{\rho\theta\phi,\text{in}}}$ |
| `q_like` | $\frac{\sigma_{\rho\theta\phi,\text{out}} \cdot \sigma_{R\phi Z,\text{out}}}{\sigma_{\rho\theta\phi,\text{in}} \cdot \sigma_{R\phi Z,\text{in}}}$ |
| `one_like` | $1$ |

The four COCOS parameters for each convention are given in Sauter Table I,
or can be computed via `cocos_to_parameters()` from the
[COCOS module](cocos.md).
