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
# Environment variable overrides (optional):
#   IMAS_MCP_SLURM_TIME        (default 01:00:00)
#   IMAS_MCP_SLURM_CPUS        (default 1)
#   IMAS_MCP_SLURM_MEM         (e.g. 4G) (unset => Slurm default)
#   IMAS_MCP_SLURM_PARTITION   (unset => scheduler default)
#   IMAS_MCP_SLURM_ACCOUNT     (unset => user default)
#   IMAS_MCP_SLURM_EXTRA       (any extra raw srun flags)
#   IMAS_MCP_USE_ENTRYPOINT    (if set to 1 use "imas-mcp" instead of python -m)
#
# Inside an existing allocation (SLURM_JOB_ID defined) the server starts directly.
# Outside an allocation it invokes srun --pty to obtain one.
#
# Notes:
# - We force unbuffered output (PYTHONUNBUFFERED=1, -u) to minimize latency.
# - Rich output is disabled to avoid protocol interference on stdio.
# - Pass-through CLI arguments are appended after the base command.

set -euo pipefail

# Defaults
: "${IMAS_MCP_SLURM_TIME:=08:00:00}"
: "${IMAS_MCP_SLURM_CPUS:=1}"
: "${IMAS_MCP_SLURM_MEM:=}"
: "${IMAS_MCP_SLURM_PARTITION:=}"
: "${IMAS_MCP_SLURM_ACCOUNT:=}"
: "${IMAS_MCP_SLURM_EXTRA:=}"
: "${IMAS_MCP_USE_ENTRYPOINT:=0}"

ARGS=("$@")

# Build the base command to run the MCP server via stdio
if [[ "${IMAS_MCP_USE_ENTRYPOINT}" == "1" ]] && command -v imas-mcp >/dev/null 2>&1; then
  BASE_CMD=(imas-mcp --transport stdio --no-rich --log-level INFO)
else
  # Use uv if available for environment management; else fallback to python directly.
  if command -v uv >/dev/null 2>&1; then
    BASE_CMD=(uv run python -u -m imas_mcp.cli --transport stdio --no-rich --log-level INFO)
  else
    BASE_CMD=(python -u -m imas_mcp.cli --transport stdio --no-rich --log-level INFO)
  fi
fi

CMD=("${BASE_CMD[@]}" "${ARGS[@]}")

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[imas-mcp-slurm] Detected existing allocation (JOB_ID=$SLURM_JOB_ID); launching directly." >&2
  exec env PYTHONUNBUFFERED=1 "${CMD[@]}"
fi

echo "[imas-mcp-slurm] Requesting Slurm allocation..." >&2

SRUN_OPTS=(--ntasks=1 --cpus-per-task="${IMAS_MCP_SLURM_CPUS}" --time="${IMAS_MCP_SLURM_TIME}" --pty)

[[ -n "${IMAS_MCP_SLURM_MEM}" ]] && SRUN_OPTS+=(--mem="${IMAS_MCP_SLURM_MEM}")
[[ -n "${IMAS_MCP_SLURM_PARTITION}" ]] && SRUN_OPTS+=(--partition="${IMAS_MCP_SLURM_PARTITION}")
[[ -n "${IMAS_MCP_SLURM_ACCOUNT}" ]] && SRUN_OPTS+=(--account="${IMAS_MCP_SLURM_ACCOUNT}")

if [[ -n "${IMAS_MCP_SLURM_EXTRA}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARR=(${IMAS_MCP_SLURM_EXTRA})
  SRUN_OPTS+=("${EXTRA_ARR[@]}")
fi

echo "[imas-mcp-slurm] srun ${SRUN_OPTS[*]} ${CMD[*]}" >&2
exec srun "${SRUN_OPTS[@]}" env PYTHONUNBUFFERED=1 "${CMD[@]}"
