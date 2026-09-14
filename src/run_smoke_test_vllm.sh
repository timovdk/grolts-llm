#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#
# Verify all four local models load and generate under vLLM before queueing the campaign.
#
#   sbatch run_smoke_test_vllm.sh                      # all four
#   sbatch run_smoke_test_vllm.sh --models qwen3-30b   # just one
#
# Two GPUs are requested because Llama-3.3-70B and Qwen3-Next-80B need them; the 1-GPU
# models run inside the same allocation, so one job covers all four.

set -euo pipefail

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0

export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/prjs1302/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"

source "${VLLM_VENV:-$HOME/venvs/vllm}/bin/activate"

cd "$(dirname "$0")"

python smoke_test_vllm.py "$@"

exit 0
