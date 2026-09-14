#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu_h100
#SBATCH --time=24:00:00
#SBATCH --array=0-7
#SBATCH --job-name=grolts-2gpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Rerun campaign for the models that need 2 GPUs.
#
#   sbatch run_reruns_2gpu.sh     # queue all 8 tasks, 2 at a time
#   ./run_reruns_2gpu.sh          # no SLURM: just list the tasks
#
# Each task loads its model once and produces a greedy pass plus RUNS sampled runs.
# Completed runs are skipped, so tasks killed by the wall clock can be resubmitted
# individually:
#
#   sbatch --array=3,5 src/run_reruns_2gpu.sh
#
# The %2 throttle keeps at most two tasks running at once; raise it if the queue allows.
# ENGINE=transformers falls back to the pre-vLLM path.
#
# Submit from this src/ dir: the SBATCH log paths above are relative to the
# submit dir, and logs/ lives here.

set -euo pipefail

RUNS="${RUNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
CHUNK="${CHUNK:-1000}"
ENGINE="${ENGINE:-vllm}"

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
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
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
        --n-gpus 2 --also-greedy
else
    source "${GROLTS_VENV:-$HOME/venvs/grolts_embed}/bin/activate"
    python generate_responses.py \
        --model "$MODEL" --dataset "$DATASET" --qset "$QSET" --chunk "$CHUNK" \
        --runs "$RUNS" --temperature "$TEMPERATURE"
fi
