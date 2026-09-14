"""Check that all four local models load and generate under vLLM before the real campaign.

Builds a tiny prompt batch from the **longest** real prompts, then runs each model through
``generate_responses_vllm.py`` end to end::

    sbatch --gpus=2 run_smoke_test_vllm.sh

Two details that make this a real test rather than a formality:

* The prompts are the longest ones in the actual batch, not short dummies. Prompts average
  ~13.3k tokens but reach 21,605, so a toy prompt would sail past a ``max_model_len`` that
  the real campaign would hit on a fifth of its requests.
* Each model runs in its own subprocess. vLLM pre-allocates a large share of GPU memory and
  does not reliably release it within a process, so loading four models in sequence in one
  process would fail for reasons that have nothing to do with the models.

Every model is attempted even if an earlier one fails, so one run tells you about all four.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from generate_responses import EMBEDDING_MODEL, INPUT_PATH, MODELS, OUTPUT_PATH

SMOKE_DATASET = "smoketest"

#: GPUs per model, mirroring submit_all_reruns.sh. The 1-GPU models run happily inside a
#: 2-GPU allocation, so the whole check fits in one job.
TENSOR_PARALLEL = {
    "qwen3-30b": 1,
    "magistral-small": 1,
    "llama-3.3-70b": 2,
    "qwen3-next-80b": 2,
}


def build_smoke_batch(args) -> Path:
    """Write a small batch of the longest real prompts, under a dataset name of its own."""
    source = (
        INPUT_PATH
        / f"{EMBEDDING_MODEL.replace('/', '_')}_generic_{args.dataset}"
        f"_{args.chunk}_{args.qset}.jsonl"
    )
    if not source.exists():
        raise SystemExit(f"[ERROR] no prompt batch at {source}")

    items = [json.loads(line) for line in source.open(encoding="utf-8")]
    items.sort(key=lambda item: len(json.dumps(item["body"]["messages"])), reverse=True)
    chosen = items[: args.n_prompts]

    target = (
        INPUT_PATH
        / f"{EMBEDDING_MODEL.replace('/', '_')}_generic_{SMOKE_DATASET}"
        f"_{args.chunk}_{args.qset}.jsonl"
    )
    with target.open("w", encoding="utf-8") as f:
        for item in chosen:
            f.write(json.dumps(item) + "\n")

    longest = len(json.dumps(chosen[0]["body"]["messages"])) // 4
    print(
        f"[INFO] {target.name}: {len(chosen)} of the longest prompts from {source.name}"
        f" (~{longest:,} tokens each)"
    )
    return target


def outputs_for(model: str, args) -> list[Path]:
    spec = MODELS[model]
    stem = (
        f"{EMBEDDING_MODEL.replace('/', '_')}_{spec.repo.replace('/', '_')}"
        f"_{SMOKE_DATASET}_{args.chunk}_{args.qset}_vllm"
    )
    return [OUTPUT_PATH / f"{stem}.jsonl", OUTPUT_PATH / f"{stem}_run1.jsonl"]


def run_model(model: str, args) -> dict:
    """Run one model in its own process and summarise what happened."""
    for path in outputs_for(model, args):
        path.unlink(missing_ok=True)
        path.with_suffix(".meta.json").unlink(missing_ok=True)

    command = [
        sys.executable, "generate_responses_vllm.py",
        "--model", model,
        "--dataset", SMOKE_DATASET,
        "--qset", str(args.qset),
        "--chunk", str(args.chunk),
        "--runs", "1",
        "--also-greedy",
        "--n-gpus", str(TENSOR_PARALLEL[model]),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    print(f"\n{'=' * 70}\n{model}\n{'=' * 70}\n$ {' '.join(command)}", flush=True)

    started = time.time()
    completed = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.time() - started

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout).strip().splitlines()[-12:]
        print("\n".join(tail))

    greedy, sampled = outputs_for(model, args)
    counts = [
        sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
        for path in (greedy, sampled)
    ]
    revision = None
    meta = greedy.with_suffix(".meta.json")
    if meta.exists():
        revision = json.loads(meta.read_text()).get("revision_loaded")

    # A completion that is empty, or that never reaches ANSWER, is a silent failure.
    answered = 0
    if greedy.exists():
        for line in greedy.open(encoding="utf-8"):
            if "ANSWER" in json.loads(line).get("completion", ""):
                answered += 1

    return {
        "model": model,
        "ok": completed.returncode == 0 and min(counts) == args.n_prompts,
        "greedy": counts[0],
        "run1": counts[1],
        "with ANSWER": answered,
        "revision": (revision or "?")[:12],
        "seconds": round(elapsed),
        "error": "" if completed.returncode == 0 else f"exit {completed.returncode}",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", default="ptsd", help="batch to draw the prompts from")
    parser.add_argument("--qset", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per model")
    parser.add_argument("--models", nargs="+", default=sorted(TENSOR_PARALLEL))
    parser.add_argument(
        "--keep", action="store_true", help="keep the smoke-test batch and outputs"
    )
    args = parser.parse_args(argv)

    batch = build_smoke_batch(args)
    results = []
    for model in args.models:
        try:
            results.append(run_model(model, args))
        except subprocess.TimeoutExpired:
            results.append({
                "model": model, "ok": False, "greedy": 0, "run1": 0, "with ANSWER": 0,
                "revision": "?", "seconds": args.timeout, "error": "timed out",
            })

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    header = f"{'model':<18}{'ok':<5}{'greedy':>7}{'run1':>6}{'ANSWER':>8}{'revision':>14}{'secs':>7}  error"
    print(header)
    for r in results:
        print(
            f"{r['model']:<18}{'yes' if r['ok'] else 'NO':<5}{r['greedy']:>7}{r['run1']:>6}"
            f"{r['with ANSWER']:>8}{r['revision']:>14}{r['seconds']:>7}  {r['error']}"
        )

    print(
        f"\nExpecting {args.n_prompts} completions in each column. A model that loads but "
        "produces\nfewer, or completions without an ANSWER section, has a problem worth "
        "finding now\nrather than 16 jobs from now."
    )

    if not args.keep:
        batch.unlink(missing_ok=True)
        for model in args.models:
            for path in outputs_for(model, args):
                path.unlink(missing_ok=True)
                path.with_suffix(".meta.json").unlink(missing_ok=True)
        print("\n[INFO] smoke-test files removed (--keep to retain them)")

    failed = [r["model"] for r in results if not r["ok"]]
    if failed:
        print(f"\n[ERROR] not ready: {', '.join(failed)}")
        return 1
    print("\n[OK] all models load and generate; safe to queue the campaign")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
