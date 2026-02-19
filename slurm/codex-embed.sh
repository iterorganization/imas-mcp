#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=UNLIMITED
#SBATCH --job-name=codex-embed
#SBATCH --nodelist=98dci4-gpu-0002
#SBATCH --output=slurm-embed-%j.log
#
# Long-lived embedding server on the Titan node.
# Titan has 8x P100-PCIE-16GB; we use 4 GPUs with 4 workers.
#
# Submit:   sbatch ~/Code/imas-codex/slurm/codex-embed.sh
# Monitor:  squeue -n codex-embed
# Cancel:   scancel -n codex-embed
#
# Pre-requisite: sync GPU deps on login node (shared GPFS cache):
#   cd ~/Code/imas-codex && uv sync --extra gpu
#
# The server binds 0.0.0.0:18765 — reachable from:
#   - Titan (localhost):  curl http://localhost:18765/health
#   - Login node:         curl http://98dci4-gpu-0002:18765/health
#   - Compute nodes:      curl http://98dci4-gpu-0002:18765/health
#   - WSL (via tunnel):   ssh -L 18765:98dci4-gpu-0002:18765 iter
#                         curl http://localhost:18765/health

set -euo pipefail

# Use P100 GPUs 0-3 (half of 8)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Ensure log directory exists
mkdir -p ~/.local/share/imas-codex/logs

cd ~/Code/imas-codex

echo "Starting embed server on $(hostname) at $(date)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -4 | paste -sd, || echo 'unknown')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start embed server — exec replaces shell for clean signal handling
# --offline: compute nodes cannot reach PyPI; deps pre-synced on login node
# --workers 4 --gpus 0,1,2,3: one worker per GPU for parallel embedding
exec uv run --offline --extra gpu imas-codex serve embed start \
    --host 0.0.0.0 \
    --port 18765 \
    --gpus 0,1,2,3 \
    --workers 4 \
    --location titan
