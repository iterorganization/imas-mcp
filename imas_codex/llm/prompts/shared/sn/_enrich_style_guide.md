## Enrichment Style Guide

### US-English Spelling — hard constraint

All output text (description, documentation, tags, links, validity_domain,
constraints) **MUST** use American (US) English spelling. This is a catalog-wide
convention enforced by automated validation.

Common UK → US pairs (US = required, UK = never use):

| US (use)          | UK (never)         |
|-------------------|--------------------|
| ionization        | ionisation         |
| normalized        | normalised         |
| polarized         | polarised          |
| magnetized        | magnetised         |
| behavior          | behaviour          |
| center            | centre             |
| analyze / analyzed | analyse / analysed |
| color             | colour             |
| modeled           | modelled           |
| labeled           | labelled           |
| fiber             | fibre              |
| fueling           | fuelling           |
| meter (prose)     | metre (prose)      |

SI unit symbols (`m`, `kg`, `eV`) are unaffected — only prose spelling matters.

### LaTeX Mathematics

- Use inline `$...$` for simple expressions: `$T_e$`, `$n_e$`, `$\nabla p = j \times B$`.
- Use display `$$...$$` only for key governing equations.
- Define ALL LaTeX variables with their units on first use in the documentation.

### Description Field

- **Maximum 2 sentences.**
- Must be physics-meaningful — not a mechanical restatement of the name tokens.
- Avoid tautology: if the name is `electron_temperature`, do NOT write
  "The temperature of the electrons." Instead describe what it physically represents,
  e.g., "Thermal energy of the electron population expressed as a temperature via
  $T_e = 2 \langle E_e \rangle / (3 k_B)$."

### Documentation Field

- **Minimum 3 sentences**, rich technical reference.
- Must cover:
  1. **Physical meaning** — what the quantity represents in the plasma.
  2. **Typical context / diagnostic** — how it is measured or computed
     (Thomson scattering, interferometry, equilibrium reconstruction, etc.).
  3. **Relationship to other quantities** — reference related standard names
     by their IDs in `links`.
- Include typical value ranges for fusion-relevant plasmas where applicable.
- For COCOS-dependent quantities, note the sign convention.
- Use LaTeX for equations; define variables on first use.

### Links

- Each entry is a **bare standard-name ID**: e.g., `electron_temperature`.
- **No** `dd:` prefixes, **no** URLs, **no** `name:` prefixes.
- Each link must name an existing standard name that genuinely enriches
  the reader's understanding of the current entry.
- Typical link count: 2–6 per entry.

### Tags

- Lowercase, hyphen-separated, no spaces: e.g., `core-plasma`, `thomson-scattering`.
- Typical tag categories: physics domain, diagnostic category, measurement type,
  plasma region.
- Include at least one physics-domain tag.

### Anti-Patterns to Avoid

- **Tautology**: the description must add information beyond what the name tokens
  already encode. Do not mechanically expand underscores into spaces.
- **DD-leak**: do NOT mention Data Dictionary path names, IDS names, or DD
  structural prefixes (e.g., `core_profiles/profiles_1d/...`) in the description
  or documentation. Standard names are IDS-agnostic physics identifiers.
- **Over-generic documentation**: avoid filler phrases like "This is an important
  quantity in fusion research." Every sentence must carry specific physics content.
