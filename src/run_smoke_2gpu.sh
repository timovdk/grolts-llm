#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=2
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --array=0-1
#SBATCH --job-name=smoke-2gpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#
# Pre-flight check for the models that need 2 GPUs: does each load and generate
# under vLLM on the longest real prompts? Mirrors run_reruns_2gpu.sh, so submitting
# this also exercises the array structure the campaign depends on.
#
#   sbatch run_smoke_2gpu.sh     # check both models
#   ./run_smoke_2gpu.sh          # no SLURM: just list the tasks
#
# Then collate all four models into one table:
#
#   python smoke_test_vllm.py --summary
#
# Submit from this src/ dir: the SBATCH log paths above are relative to the
# submit dir, and logs/ lives here.

set -euo pipefail

QSET="${QSET:-4}"
CHUNK="${CHUNK:-1000}"
N_PROMPTS="${N_PROMPTS:-4}"

TASKS=(
    "llama-3.3-70b"
    "qwen3-next-80b"
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

MODEL="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[INFO] task $SLURM_ARRAY_TASK_ID: smoke test $MODEL"

module load 2025 Python/3.13.1-GCCcore-14.2.0 CUDA/12.8.0
export 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
export HF_CACHE_DIR="${HF_CACHE_DIR:-/projects/prjs1302/hf_cache}"
export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"

source "${VLLM_VENV:-$HOME/venvs/xling}/bin/activate"

# SLURM runs a spool copy of this script, so $0 points into /var/spool;
# resolve the source dir from the submit dir instead (from $0 outside SLURM).
SRC_DIR="${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
[ -f "$SRC_DIR/smoke_test_vllm.py" ] || SRC_DIR="$SRC_DIR/src"
if [ ! -f "$SRC_DIR/smoke_test_vllm.py" ]; then
    echo "cannot find smoke_test_vllm.py (looked in $SRC_DIR); submit from the repo or its src/ dir" >&2
    exit 1
fi
cd "$SRC_DIR"

python smoke_test_vllm.py \
    --model "$MODEL" \
    --qset "$QSET" \
    --chunk "$CHUNK" \
    --n-prompts "$N_PROMPTS" \
    --n-gpus 2
