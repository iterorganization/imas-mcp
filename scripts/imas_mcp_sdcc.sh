#!/usr/bin/env bash
#
# imas_mcp_sdcc.sh
#
# Launch the IMAS MCP server over STDIO via Slurm. This lets MCP clients (VS Code,
# Claude, etc.) allocate a compute node transparently and communicate over
# stdin/stdout without opening network ports.
#
# Usage (direct):
#   scripts/imas_mcp_slurm_stdio.sh [<extra mcp args>...]
#
# In VS Code .vscode/mcp.json (JSONC):
#   {
#     "servers": {
#       "imas-sdcc": {
#         "type": "stdio",
#         "command": "scripts/imas_mcp_sdcc.sh",
#       }
#     }
#   }
#
# Optional environment variables (all unset => rely on cluster defaults):
#   IMAS_MCP_SLURM_TIME        (e.g. 02:00:00)
#   IMAS_MCP_SLURM_CPUS        (cpus per task)
#   IMAS_MCP_SLURM_MEM         (e.g. 4G)
#   IMAS_MCP_SLURM_PARTITION   (partition/queue)
#   IMAS_MCP_SLURM_ACCOUNT     (account)
#   IMAS_MCP_SLURM_EXTRA       (raw additional srun flags)
#
# Inside an existing allocation (SLURM_JOB_ID defined) the server starts directly.
# Outside an allocation it invokes srun --pty to obtain one.
#
# Notes:
# - We force unbuffered output (PYTHONUNBUFFERED=1, -u) to minimize latency.
# - Rich output is disabled to avoid protocol interference on stdio.
# - Pass-through CLI arguments are appended after the base command.

set -euo pipefail

ARGS=("$@")
# Build the base command to run the MCP server via stdio (uv is mandatory)
if ! command -v uv >/dev/null 2>&1; then
  echo "[imas-mcp-slurm] 'uv' is required but not found in PATH. Install https://github.com/astral-sh/uv and retry." >&2
  exit 1
fi
BASE_CMD=(uv run python -u -m imas_mcp.cli --transport stdio --no-rich --log-level INFO)

CMD=("${BASE_CMD[@]}" "${ARGS[@]}")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[imas-mcp-slurm] Detected existing allocation (JOB_ID=$SLURM_JOB_ID); launching directly." >&2
  exec env PYTHONUNBUFFERED=1 "${CMD[@]}"
fi

echo "[imas-mcp-slurm] Requesting Slurm allocation..." >&2

SRUN_OPTS=(--ntasks=1)

[[ -n "${IMAS_MCP_SLURM_CPUS:-}" ]] && SRUN_OPTS+=(--cpus-per-task="${IMAS_MCP_SLURM_CPUS}")
[[ -n "${IMAS_MCP_SLURM_TIME:-}" ]] && SRUN_OPTS+=(--time="${IMAS_MCP_SLURM_TIME}")
[[ -n "${IMAS_MCP_SLURM_MEM:-}" ]] && SRUN_OPTS+=(--mem="${IMAS_MCP_SLURM_MEM}")
[[ -n "${IMAS_MCP_SLURM_PARTITION:-}" ]] && SRUN_OPTS+=(--partition="${IMAS_MCP_SLURM_PARTITION}")
[[ -n "${IMAS_MCP_SLURM_ACCOUNT:-}" ]] && SRUN_OPTS+=(--account="${IMAS_MCP_SLURM_ACCOUNT}")

if [[ -n "${IMAS_MCP_SLURM_EXTRA:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${IMAS_MCP_SLURM_EXTRA})
  SRUN_OPTS+=("${EXTRA_ARR[@]}")
fi

echo "[imas-mcp-slurm] srun ${SRUN_OPTS[*]} ${CMD[*]}" >&2
exec srun "${SRUN_OPTS[@]}" env PYTHONUNBUFFERED=1 "${CMD[@]}"
