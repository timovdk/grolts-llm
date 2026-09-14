#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_h100
#SBATCH --time=48:00:00
#
# Generate checklist answers with one locally hosted model.
#
#   sbatch run_generate_responses.sh <model> <dataset> <qset> <chunk> [extra args...]
#
# The 70B and 80B models need two H100s; override the default at submission time:
#
#   sbatch --gpus=2 run_generate_responses.sh llama-3.3-70b delinquency 4 1000
#   sbatch --gpus=2 run_generate_responses.sh qwen3-next-80b delinquency 4 1000
#   sbatch          run_generate_responses.sh qwen3-30b      delinquency 4 1000
#   sbatch          run_generate_responses.sh magistral-small delinquency 4 1000
#
# Sampled reruns for the run-to-run variability analysis (reviewer 2, major 2):
#
#   sbatch --gpus=2 run_generate_responses.sh llama-3.3-70b delinquency 4 1000 \
#          --runs 5 --temperature 0.7
#
# Completed runs are skipped, so a job that hits the wall time can just be resubmitted.

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
# Overridable so the script is not pinned to one cluster's project directory.
export HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/prjs1302/hf_cache}"

source "$HOME/venvs/grolts_embed/bin/activate"

python generate_responses.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --qset "$QSET" \
    --chunk "$CHUNK" \
    "$@"

exit 0
