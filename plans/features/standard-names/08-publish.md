# Feature 08: Publish (YAML + PR)

**Repository:** imas-codex  
**Complexity:** Medium  
**Depends on:** 05 (SN Build Pipeline), 06 (Cross-Model Review)  
**Wave:** 4

---

## Overview

Convert validated standard name candidates into catalog YAML files and create batched pull requests to the imas-standard-names-catalog repository.

## Design

### YAML Generation
- Produce YAML files matching the `StandardNameEntry` JSON schema (Feature 04)
- One file per standard name (matching existing catalog convention)
- Include all fields: name, kind, unit, tags, status, description, provenance

### Batched PR Strategy
- Group names by IDS or physics domain
- Each PR contains a coherent batch (e.g., "Add 47 equilibrium standard names")
- Include confidence tier summary in PR description
- High-confidence names in separate PRs from medium-confidence (easier review)

### PR Content
- YAML files in correct directory structure
- Summary table in PR description (name, unit, source, confidence)
- Link back to source DD paths / signals
- Automated validation check (CI runs standard-names validation)

### Catalog Awareness
- Check existing catalog before generating PR
- Skip names that already exist (exact match)
- Flag potential conflicts (similar names) in PR description

## Deliverables

- [ ] YAML file generation from validated candidates
- [ ] GitHub PR creation via `gh` CLI or API
- [ ] Batching logic by IDS/domain
- [ ] Confidence tier separation
- [ ] Catalog dedup checking
- [ ] PR description template with summary table
- [ ] Dry-run mode: generate YAML locally without creating PR
