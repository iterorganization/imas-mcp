#!/bin/bash
# MDSplus Tree Ingestion Script for TCV/EPFL
# Run from imas-codex project directory with Neo4j running
#
# This script ingests all MDSplus trees using discover-mdsplus,
# which performs epoch discovery (finding when tree structure changed)
# and builds the "super tree" with shot-range applicability.
#
# Usage: 
#   cd /path/to/imas-codex
#   uv run imas-codex neo4j start  # If not already running
#   ./scripts/ingest_all_trees.sh 2>&1 | tee ingest_trees.log
#
# Expected runtime: Several hours (depends on tree complexity)
# - Simple trees (base, manual): ~5-10 minutes
# - Complex trees (tcv_shot, atlas): ~30-60 minutes each

set -e

# Trees to ingest - all valid TCV trees
# Trees already done with epochs: results (129 epochs), magnetics (22 epochs)
TREES=(
    "apcs"
    "atlas"
    "base"
    "diag_act"
    "diagz"
    "ecrh"
    "hybrid"
    "manual"
    "pcs"
    "power"
    "raw_bolo"
    "raw_ece"
    "raw_fild"
    "raw_mag"
    "raw_mhd"
    "tcv_shot"
    "thomson"
    "vsystem"
)

FACILITY="epfl"
LOGDIR="logs/mdsplus_ingestion"
mkdir -p "$LOGDIR"

echo "=========================================="
echo "MDSplus Tree Ingestion - $(date)"
echo "=========================================="
echo "Facility: $FACILITY"
echo "Trees to process: ${#TREES[@]}"
echo "Log directory: $LOGDIR"
echo ""

# Check Neo4j is running
if ! uv run python -c "from imas_codex.graph import GraphClient; GraphClient().query('RETURN 1')" 2>/dev/null; then
    echo "ERROR: Neo4j is not running. Start with: uv run imas-codex neo4j start"
    exit 1
fi
echo "✓ Neo4j connection verified"
echo ""

FAILED=()
SUCCEEDED=()

for tree in "${TREES[@]}"; do
    echo "----------------------------------------"
    echo "Processing: $tree"
    echo "Started: $(date)"
    
    logfile="$LOGDIR/${tree}_$(date +%Y%m%d_%H%M%S).log"
    
    # Run discover-mdsplus with full re-scan and legacy cleanup
    # --full to force complete re-scan (fixes epochs with old hash fingerprints)
    # --clean to merge and cleanup legacy nodes after ingestion
    if uv run discover-mdsplus "$FACILITY" "$tree" 2>&1 | tee "$logfile"; then
        echo "✓ $tree completed successfully"
        SUCCEEDED+=("$tree")
    else
        echo "✗ $tree FAILED (see $logfile)"
        FAILED+=("$tree")
    fi
    
    echo "Finished: $(date)"
    echo ""
done

echo "=========================================="
echo "All trees processed - $(date)"
echo "=========================================="
echo ""
echo "Succeeded: ${#SUCCEEDED[@]} trees"
echo "Failed: ${#FAILED[@]} trees"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed trees:"
    for tree in "${FAILED[@]}"; do
        echo "  - $tree"
    done
fi

# Summary
echo ""
echo "=== Ingestion Summary ==="
uv run python -c "
from imas_codex.graph import GraphClient
with GraphClient() as client:
    result = client.query('''
        MATCH (t:TreeNode)
        RETURN t.tree_name AS tree, count(t) AS nodes
        ORDER BY nodes DESC
    ''')
    total = 0
    for r in result:
        print(f'  {r[\"tree\"]}: {r[\"nodes\"]} nodes')
        total += r['nodes']
    print(f'  ─────────────────')
    print(f'  TOTAL: {total} nodes')
    
    print()
    result = client.query('''
        MATCH (v:TreeModelVersion)
        RETURN v.tree_name AS tree, count(v) AS epochs
        ORDER BY epochs DESC
    ''')
    print('Epochs by tree:')
    for r in result:
        print(f'  {r[\"tree\"]}: {r[\"epochs\"]} epochs')
"
