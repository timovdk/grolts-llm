#!/bin/bash
#
# Submit every locally hosted model's generation for the variability campaign.
#
#   ./submit_all_reruns.sh                  # dry run: print the sbatch commands
#   ./submit_all_reruns.sh --submit         # queue them
#   ENGINE=transformers ./submit_all_reruns.sh   # use the transformers path instead
#
# 4 models x 4 (dataset, checklist) combinations = 16 jobs. Each job loads the model once
# and produces a greedy pass plus RUNS sampled runs from that single load.
#
# The greedy pass matters: without a baseline under the same engine, any difference between
# the published transformers runs and these sampled runs would confound sampling noise with
# the change of inference engine. With it, greedy-vs-sampled isolates sampling noise, and
# vLLM-greedy vs transformers-greedy is a separate engine-sensitivity check.
#
# gpt-5-mini is not here: its request files are in packages/openai_batches/ and are submitted
# from the account that holds the API key.
#
# Completed runs are skipped, so re-running this after a wall-time kill re-queues only what
# is missing.

set -euo pipefail

ENGINE="${ENGINE:-vllm}"
RUNS="${RUNS:-5}"
TEMPERATURE="${TEMPERATURE:-0.7}"
CHUNK="${CHUNK:-1000}"

# dataset:checklist version. PTSD is run on both checklists, the others on v2 only.
COMBOS=("ptsd:0" "ptsd:4" "delinquency:4" "achievement:4")

# model:gpus — the 70B and 80B need two H100s. Mirrors the registry in generate_responses.py.
MODELS=("qwen3-30b:1" "magistral-small:1" "llama-3.3-70b:2" "qwen3-next-80b:2")

case "$ENGINE" in
    vllm)         SCRIPT=run_generate_responses_vllm.sh ;;
    transformers) SCRIPT=run_generate_responses.sh ;;
    *) echo "ENGINE must be 'vllm' or 'transformers', got '$ENGINE'" >&2; exit 1 ;;
esac

SUBMIT=false
[ "${1:-}" = "--submit" ] && SUBMIT=true

cd "$(dirname "$0")"

count=0
for entry in "${MODELS[@]}"; do
    model="${entry%%:*}"
    gpus="${entry##*:}"
    for combo in "${COMBOS[@]}"; do
        dataset="${combo%%:*}"
        qset="${combo##*:}"

        cmd=(sbatch --gpus="$gpus" "$SCRIPT" "$model" "$dataset" "$qset" "$CHUNK"
             --runs "$RUNS" --temperature "$TEMPERATURE")
        if [ "$ENGINE" = "vllm" ]; then
            # --n-gpus must match the allocation; --also-greedy adds the same-engine baseline.
            cmd+=(--n-gpus "$gpus" --also-greedy)
        fi

        count=$((count + 1))
        if $SUBMIT; then
            "${cmd[@]}"
        else
            echo "${cmd[*]}"
        fi
    done
done

runs_each=$RUNS
[ "$ENGINE" = "vllm" ] && runs_each="$RUNS sampled + 1 greedy"

if $SUBMIT; then
    echo "[INFO] queued $count jobs via $ENGINE ($runs_each per job)"
else
    echo
    echo "[INFO] $count jobs would be queued via $ENGINE ($runs_each per job)."
    echo "[INFO] re-run with --submit to queue them."
    echo
    echo "[INFO] before queueing all of them, smoke-test one model end to end:"
    echo "         sbatch --gpus=1 $SCRIPT qwen3-30b achievement 4 $CHUNK --runs 1 --n-gpus 1"
    echo "[INFO] afterwards, capture the model revisions for pinning:"
    echo "         python generate_responses.py --model qwen3-30b --dataset ptsd --show-revisions"
fi
