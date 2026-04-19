# Gyrokinetics SN Extraction Debug

## Diagnosis

All 264 `quantity`-category IMASNodes in the `gyrokinetics` physics domain have
`node_type = 'constant'` (not `'dynamic'`).  The extraction query in
`imas_codex/standard_names/sources/dd.py` filtered on
`n.node_type = 'dynamic'`, unconditionally excluding every gyrokinetics path.
The `turbulence` domain (316 constant quantities) was similarly affected.

## Fix

Relaxed the `node_type` filter from `= 'dynamic'` to
`IN ['dynamic', 'constant']` in both `sources/dd.py` (primary extraction path)
and `graph_ops.py` (legacy helper).  `static` and `none` types are still
excluded — they represent machine/hardware parameters and unclassified nodes
respectively, which are less likely to be meaningful standard-name candidates.
