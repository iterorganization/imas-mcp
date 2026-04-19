#!/usr/bin/env bash
# rc14_bootstrap.sh — Full standard-name rotation for rc14.
#
# Plan 31 §G.2 — orchestrates clear → generate → resolve-links → enrich →
# review → corpus_health gate.  Uses only existing CLI verbs.
#
# Usage:
#   bash scripts/rc14_bootstrap.sh          # run full rotation
#   DOMAINS="equilibrium magnetics" bash scripts/rc14_bootstrap.sh  # subset
#
# Log output: ~/.local/share/imas-codex/logs/rc14_bootstrap.log
# Exit: non-zero on any failure (set -e).

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Physics domains to process (override with DOMAINS env var)
DOMAINS=( ${DOMAINS:-equilibrium core_profiles magnetics transport edge_profiles mhd pwi fast_particles} )

# Cost ceiling per LLM call (USD)
COST_LIMIT="${COST_LIMIT:-2.0}"

# Log directory
LOG_DIR="${HOME}/.local/share/imas-codex/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/rc14_bootstrap.log"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() {
    local ts
    ts="$(date -Iseconds)"
    echo "[${ts}] $*" | tee -a "${LOG_FILE}"
}

run_step() {
    local label="$1"; shift
    log "▶ ${label}"
    log "  cmd: $*"
    if "$@" 2>&1 | tee -a "${LOG_FILE}"; then
        log "✓ ${label} succeeded"
    else
        log "✗ ${label} FAILED (exit $?)"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Rotation pipeline
# ---------------------------------------------------------------------------

log "═══════════════════════════════════════════════════════"
log "rc14 bootstrap rotation — $(date)"
log "Domains: ${DOMAINS[*]}"
log "Cost limit per step: \$${COST_LIMIT}"
log "═══════════════════════════════════════════════════════"

# Step 1 — Clear all standard names (including sources and accepted)
run_step "clear" \
    uv run imas-codex sn clear --all --include-sources --include-accepted --force

# Step 2 — Generate names per domain
for domain in "${DOMAINS[@]}"; do
    run_step "generate [${domain}]" \
        uv run imas-codex sn generate --source dd --physics-domain "${domain}" -c "${COST_LIMIT}"
done

# Step 3 — Resolve dd: links to name: links
run_step "resolve-links" \
    uv run imas-codex sn resolve-links

# Step 4 — Enrich descriptions per domain
for domain in "${DOMAINS[@]}"; do
    run_step "enrich [${domain}]" \
        uv run imas-codex sn enrich --domain "${domain}" -c "${COST_LIMIT}"
done

# Step 5 — Review (3-layer pipeline)
run_step "review" \
    uv run imas-codex sn review

# Step 6 — Corpus health gate
run_step "corpus_health gate" \
    uv run pytest tests/standard_names/test_corpus_health.py -m corpus_health -v

log "═══════════════════════════════════════════════════════"
log "rc14 bootstrap rotation COMPLETE"
log "═══════════════════════════════════════════════════════"
