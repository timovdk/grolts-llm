#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --partition=gpu_h100
#SBATCH --time=4:00:00
#SBATCH --array=0-7
#SBATCH --job-name=grolts-generations-2gpu
#SBATCH --output=logs/%x-%A_%a.log

set -euo pipefail

RUNS="${RUNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
CHUNK="${CHUNK:-1000}"
ENGINE="${ENGINE:-vllm}"
# vLLM's engine-side batch limit, exposed so a task can be retried with different
# batching without editing the runner:
#   MAX_NUM_SEQS=8 sbatch --array=0 run_generate_responses_2gpu.sh
# Keep it the same across the campaign: batch composition can perturb numerics.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"

# model dataset qset — one entry per array index.
TASKS=(
    "llama-3.3-70b ptsd 0"
    "llama-3.3-70b ptsd 4"
    "llama-3.3-70b delinquency 4"
    "llama-3.3-70b achievement 4"
    "qwen3-next-80b ptsd 0"
    "qwen3-next-80b ptsd 4"
    "qwen3-next-80b delinquency 4"
    "qwen3-next-80b achievement 4"
)

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "${#TASKS[@]} tasks (2 GPUs each) — submit with: sbatch $0"
    for i in "${!TASKS[@]}"; do printf "  [%d] %s\n" "$i" "${TASKS[$i]}"; done
    exit 0
fi

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#TASKS[@]}" ]; then
    echo "array index $SLURM_ARRAY_TASK_ID is past the end of ${#TASKS[@]} tasks" >&2
    exit 1
fi

read -r MODEL DATASET QSET <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[INFO] task $SLURM_ARRAY_TASK_ID: $MODEL $DATASET qset $QSET chunk $CHUNK"

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0
export PYTHONUNBUFFERED=1
# expandable_segments helps the transformers path, but it cannot be exported as a
# CUDA IPC handle, which is how vLLM's TP workers share tensors -- so set it only
# for the engine that wants it.
if [ "$ENGINE" != "vllm" ]; then
    export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
fi
# Quiet by default over a long campaign; submit with NCCL_DEBUG=INFO to diagnose a
# tensor-parallel failure. Warnings and errors print either way.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/prjs1302/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"

# SLURM runs a spool copy of this script, so $0 points into /var/spool;
# resolve the source dir from the submit dir instead (from $0 outside SLURM).
SRC_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$SRC_DIR/generate_responses_vllm.py" ] || SRC_DIR="$SRC_DIR/src"
if [ ! -f "$SRC_DIR/generate_responses_vllm.py" ]; then
    echo "cannot find generate_responses_vllm.py (looked in $SRC_DIR); submit from the repo or its src/ dir" >&2
    exit 1
fi
cd "$SRC_DIR"

if [ "$ENGINE" = "vllm" ]; then
    source "${VLLM_VENV:-$HOME/venvs/xling}/bin/activate"
    python generate_responses_vllm.py \
        --model "$MODEL" --dataset "$DATASET" --qset "$QSET" --chunk "$CHUNK" \
        --runs "$RUNS" --temperature "$TEMPERATURE" \
        --n-gpus 2 --also-greedy \
        --max-num-seqs "$MAX_NUM_SEQS"
else
    source "${GROLTS_VENV:-$HOME/venvs/grolts_embed}/bin/activate"
    python generate_responses.py \
        --model "$MODEL" --dataset "$DATASET" --qset "$QSET" --chunk "$CHUNK" \
        --runs "$RUNS" --temperature "$TEMPERATURE"
fi
