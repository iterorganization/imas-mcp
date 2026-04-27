## IMAS Coordinate Conventions — Normative Rules

When referring to a coordinate system in documentation prose or in standard-name
descriptions, **ALWAYS specify the system explicitly.** NEVER use vague phrases
such as "the standard cylindrical system", "the standard frame", "the standard
coordinate system", or "machine cylindrical frame" alone (i.e. without the
explicit basis tuple).

### IMAS coordinate facts (COCOS-17 / IMAS DD)

1. **Cylindrical basis** is right-handed $(\hat{R}, \hat{\phi}, \hat{Z})$ with
   $\hat{R} \times \hat{\phi} = \hat{Z}$.
   - $R$ — major radius measured from the machine symmetry axis, in metres.
   - $\phi$ — toroidal angle increasing counter-clockwise when viewed from above,
     in radians.
   - $Z$ — vertical height, in metres.

2. **Cartesian basis** (used for sensor direction unit vectors, line-of-sight
   points, etc.) is right-handed $(\hat{x}, \hat{y}, \hat{z})$ with
   $\hat{z} = \hat{Z}$ (vertical) and $\phi = 0$ defined by the IDS or facility
   convention.

3. **Outline storage tuples** in IMAS sometimes appear as $(R, Z, \phi)$ — this
   is a tuple of coordinate **values** used for indexing 3-D points, NOT the
   basis-vector ordering. The basis is ALWAYS right-handed
   $(\hat{R}, \hat{\phi}, \hat{Z})$.

4. **Flux coordinates** must always be named explicitly: $(\rho, \theta, \phi)$,
   $(\psi, \theta, \phi)$, $(s, \theta^*, \phi)$, etc. NEVER call them
   "the standard" or "the usual" flux coordinates.

### Required phrasing — ❌ / ✓ examples

| ❌ Vague / wrong | ✓ Explicit / correct |
|---|---|
| "in the standard cylindrical system" | "in the IMAS cylindrical $(R, \phi, Z)$ frame" |
| "the cylindrical coordinate system" | "the right-handed cylindrical $(R, \phi, Z)$ system" |
| "machine cylindrical frame" (alone) | "machine cylindrical frame $(R, \phi, Z)$ where $R$ is major radius and $Z$ is vertical height" |
| "in cylindrical coordinates $(R, Z, \phi)$" (claiming basis ordering) | "in the cylindrical $(R, \phi, Z)$ frame; outlines may be stored as 3-tuples $(R, Z, \phi)$" |
| "the standard flux coordinate" | "the flux coordinate $\rho_\mathrm{tor,norm}$ (normalised toroidal flux radius)" |

### Enforcement

- Every documentation string that mentions a coordinate system MUST include the
  explicit basis tuple, e.g. $(R, \phi, Z)$ or $(\hat{x}, \hat{y}, \hat{z})$.
- The prohibition covers all output fields: `description`, `documentation`,
  `validity_domain`, and `constraints`.
- Do NOT use the word "standard" to qualify a coordinate system unless you
  immediately follow it with the explicit tuple and a brief definition.
