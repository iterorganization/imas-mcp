#!/bin/bash
#SBATCH --partition=titan
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=codex-llm-local
#SBATCH --nodelist=98dci4-gpu-0002
#SBATCH --output=slurm-llm-%j.log
#
# Local LLM inference via Ollama on the Titan node.
# Uses 1x P100-PCIE-16GB (GPU 6 or 7, leaving 0-5 for embedding).
#
# Submit:   sbatch ~/Code/imas-codex/slurm/ollama-llm.sh
# Monitor:  squeue -n codex-llm-local
# Cancel:   scancel -n codex-llm-local
#
# Pre-requisites (run on login node which has internet):
#   1. Install Ollama:  curl -fsSL https://ollama.com/install.sh | sh
#   2. Pull model:      ollama pull qwen3:14b-q4_K_M
#   3. Model files are stored in ~/.ollama/models/ (shared GPFS)
#
# The server binds 0.0.0.0:11434 — reachable from:
#   - Titan (localhost):  curl http://localhost:11434/api/tags
#   - Login node:         curl http://98dci4-gpu-0002:11434/api/tags
#   - WSL (via tunnel):   ssh -L 11434:98dci4-gpu-0002:11434 iter
#
# LiteLLM integration:
#   Uncomment the ollama model entries in litellm_config.yaml and
#   restart the proxy: imas-codex llm restart

set -euo pipefail

# Use GPU 6 (leaving 0-5 for embedding service)
export CUDA_VISIBLE_DEVICES=6

# Ollama configuration
export OLLAMA_HOST=0.0.0.0:11434
export OLLAMA_MODELS=${OLLAMA_MODELS:-$HOME/.ollama/models}
export OLLAMA_NUM_PARALLEL=${OLLAMA_NUM_PARALLEL:-2}
export OLLAMA_MAX_LOADED_MODELS=${OLLAMA_MAX_LOADED_MODELS:-1}

# Ensure log directory exists
mkdir -p ~/.local/share/imas-codex/logs

echo "Starting Ollama LLM server on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --id=$CUDA_VISIBLE_DEVICES --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Models: $OLLAMA_MODELS"

# Check if Ollama is installed
if ! command -v ollama &>/dev/null; then
    echo "ERROR: ollama not found. Install on login node first:"
    echo "  curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Check if model files exist
if [ ! -d "$OLLAMA_MODELS" ]; then
    echo "ERROR: Model directory not found: $OLLAMA_MODELS"
    echo "Pull models on login node first: ollama pull qwen3:14b-q4_K_M"
    exit 1
fi

# Start Ollama server — exec replaces shell for clean signal handling
exec ollama serve 2>&1 | tee -a ~/.local/share/imas-codex/logs/llm-local.log
