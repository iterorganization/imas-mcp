#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=UNLIMITED
#SBATCH --job-name=codex-embed
#SBATCH --nodelist=98dci4-gpu-0002
#SBATCH --output=slurm-embed-%j.log
#
# Long-lived embedding server on the Titan node.
# Titan has 8x P100-PCIE-16GB; we use 1 GPU.
#
# Submit:   sbatch ~/Code/imas-codex/slurm/codex-embed.sh
# Monitor:  squeue -n codex-embed
# Cancel:   scancel -n codex-embed
#
# The server binds 0.0.0.0:18765 — reachable from:
#   - Titan (localhost):  curl http://localhost:18765/health
#   - Login node:         curl http://98dci4-gpu-0002:18765/health
#   - Compute nodes:      curl http://98dci4-gpu-0002:18765/health
#   - WSL (via tunnel):   ssh -L 18765:98dci4-gpu-0002:18765 iter
#                         curl http://localhost:18765/health

set -euo pipefail

# Use P100 GPU 0
export CUDA_VISIBLE_DEVICES=0

# Ensure log directory exists
mkdir -p ~/.local/share/imas-codex/logs

cd ~/Code/imas-codex

echo "Starting embed server on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Start embed server — exec replaces shell for clean signal handling
exec uv run --extra gpu imas-codex serve embed start \
    --host 0.0.0.0 \
    --port 18765 \
    --gpu 0 \
    --location titan
