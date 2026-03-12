## Completion Criteria

### Minimum Viable Exploration (MVE)

Stop when you have:

- [ ] Environment characterized (OS, Python, IMAS version)
- [ ] At least 5 code directories identified with interest scores
- [ ] At least 20 source files queued for ingestion
- [ ] IMAS integration patterns documented (if any exist)

### Full Exploration Complete

Stop when:

- [ ] All major code directories surveyed (>100 files each)
- [ ] 100+ high-value source files (score >= 0.75) queued
- [ ] Key physics domains covered (equilibrium, profiles, transport)
- [ ] MDSplus trees/databases identified
- [ ] No new high-value patterns emerging from searches

### Diminishing Returns Signal

Stop exploring when:

- New searches return mostly config/build files
- Same code patterns appearing repeatedly
- Large directories contain mostly auto-generated code
- Searches take >30s with few relevant results
