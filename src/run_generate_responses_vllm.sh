#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#
# Generate checklist answers with vLLM.
#
#   sbatch run_generate_responses_vllm.sh <model> <dataset> <qset> <chunk> [extra args...]
#
# The 70B and 80B models need two H100s, and --n-gpus must match the allocation:
#
#   sbatch --gpus=2 run_generate_responses_vllm.sh llama-3.3-70b ptsd 4 1000 \
#          --n-gpus 2 --runs 5 --also-greedy
#
# Runs resume: completed requests are read back from a partial output file and skipped, so
# a job killed by the wall clock can simply be resubmitted.

set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "usage: $0 <model> <dataset> <qset> <chunk> [extra args...]" >&2
    echo "models: qwen3-30b | qwen3-next-80b | llama-3.3-70b | magistral-small" >&2
    exit 1
fi

MODEL=$1
DATASET=$2
QSET=$3
CHUNK=$4
shift 4

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0

export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/prjs1302/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"

# vLLM lives in its own environment; override VLLM_VENV to point elsewhere.
source "${VLLM_VENV:-$HOME/venvs/vllm}/bin/activate"

cd "$(dirname "$0")"

python generate_responses_vllm.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --qset "$QSET" \
    --chunk "$CHUNK" \
    "$@"

exit 0
