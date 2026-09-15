"""Check that a local model loads and generates under vLLM before the real campaign.

One model per invocation, so it slots straight into a SLURM job array::

    sbatch src/run_smoke_1gpu.sh      # qwen3-30b, magistral-small
    sbatch src/run_smoke_2gpu.sh      # llama-3.3-70b, qwen3-next-80b
    python src/smoke_test_vllm.py --summary

Each run writes ``logs/smoke_<model>.json``; ``--summary`` collates them into one table, so
the four results survive being spread across four array-task logs.

Two details that make this a real test rather than a formality:

* The prompts are the **longest** ones in the actual batch, not short dummies. Prompts
  average ~13.3k tokens but reach 21,605, and a toy prompt would sail past exactly the
  ``max_model_len`` problem that a quarter of the real campaign would hit.
* It checks for an ``ANSWER`` section, not merely a non-empty completion. A model that loads
  and emits text but never reaches ``ANSWER:`` is the failure behind the unparsed cells in
  the published runs, and would otherwise look like success.

Each model gets its own smoke batch file, so array tasks running concurrently cannot race on
it. Generation runs in a subprocess: vLLM pre-allocates GPU memory and does not reliably
release it within a process, which matters when models are checked locally in sequence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from pipeline_config import EMBEDDING_MODEL, MODELS, OUTPUT_PATH, prompt_batch

RESULT_DIR = Path("logs")

#: GPUs per model, from the single registry the runners also use.
TENSOR_PARALLEL = {name: spec.tensor_parallel for name, spec in MODELS.items()}


def smoke_dataset(model: str) -> str:
    """A dataset name unique to this model, so concurrent tasks never share a file."""
    return f"smoke-{model}"


def build_smoke_batch(model: str, args) -> Path:
    """Write a small batch of the longest real prompts under this model's own name."""
    source = prompt_batch(args.dataset, args.chunk, args.qset)
    if not source.exists():
        raise SystemExit(f"[ERROR] no prompt batch at {source}")

    items = [json.loads(line) for line in source.open(encoding="utf-8")]
    items.sort(key=lambda item: len(json.dumps(item["body"]["messages"])), reverse=True)
    chosen = items[: args.n_prompts]

    target = prompt_batch(smoke_dataset(model), args.chunk, args.qset)
    with target.open("w", encoding="utf-8") as f:
        for item in chosen:
            f.write(json.dumps(item) + "\n")
    print(
        f"[INFO] {target.name}: {len(chosen)} of the longest prompts from {source.name}",
        flush=True,
    )
    return target


def outputs_for(model: str, args) -> list[Path]:
    spec = MODELS[model]
    stem = (
        f"{EMBEDDING_MODEL.replace('/', '_')}_{spec.repo.replace('/', '_')}"
        f"_{smoke_dataset(model)}_{args.chunk}_{args.qset}_vllm"
    )
    return [OUTPUT_PATH / f"{stem}.jsonl", OUTPUT_PATH / f"{stem}_run1.jsonl"]


def run_model(model: str, args) -> dict:
    """Run one model end to end and summarise what happened."""
    batch = build_smoke_batch(model, args)
    for path in outputs_for(model, args):
        path.unlink(missing_ok=True)
        path.with_suffix(".meta.json").unlink(missing_ok=True)

    command = [
        sys.executable, "generate_responses_vllm.py",
        "--model", model,
        "--dataset", smoke_dataset(model),
        "--qset", str(args.qset),
        "--chunk", str(args.chunk),
        "--runs", "1",
        "--also-greedy",
        "--n-gpus", str(args.n_gpus or TENSOR_PARALLEL[model]),
        "--max-model-len", str(args.max_model_len),
        "--max-num-seqs", str(args.max_num_seqs),
    ]
    print(f"$ {' '.join(command)}", flush=True)

    started = time.time()
    try:
        returncode = subprocess.run(command, timeout=args.timeout).returncode
        error = "" if returncode == 0 else f"exit {returncode}"
    except subprocess.TimeoutExpired:
        returncode, error = -1, "timed out"
    elapsed = time.time() - started

    greedy, sampled = outputs_for(model, args)
    counts = [
        sum(1 for _ in path.open(encoding="utf-8")) if path.exists() else 0
        for path in (greedy, sampled)
    ]

    revision = None
    meta = greedy.with_suffix(".meta.json")
    if meta.exists():
        revision = json.loads(meta.read_text()).get("revision_loaded")

    # A completion that never reaches ANSWER is a silent failure, not a success.
    answered = 0
    if greedy.exists():
        for line in greedy.open(encoding="utf-8"):
            if "ANSWER" in json.loads(line).get("completion", ""):
                answered += 1

    result = {
        "model": model,
        "ok": returncode == 0 and min(counts) == args.n_prompts and answered == args.n_prompts,
        "greedy": counts[0],
        "run1": counts[1],
        "answered": answered,
        "expected": args.n_prompts,
        "revision": revision,
        "seconds": round(elapsed),
        "error": error,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"smoke_{model}.json").write_text(json.dumps(result, indent=2))

    if not args.keep:
        batch.unlink(missing_ok=True)
        for path in outputs_for(model, args):
            path.unlink(missing_ok=True)
            path.with_suffix(".meta.json").unlink(missing_ok=True)

    return result


def summarise() -> int:
    """Collate the per-model result files written by the array tasks."""
    files = sorted(RESULT_DIR.glob("smoke_*.json"))
    if not files:
        print(f"[WARN] no results in {RESULT_DIR}; has the smoke test run yet?")
        return 1

    results = [json.loads(f.read_text()) for f in files]
    print(
        f"{'model':<18}{'ok':<5}{'greedy':>7}{'run1':>6}{'ANSWER':>8}"
        f"{'revision':>14}{'secs':>7}  error"
    )
    for r in results:
        print(
            f"{r['model']:<18}{'yes' if r['ok'] else 'NO':<5}{r['greedy']:>7}{r['run1']:>6}"
            f"{r['answered']:>8}{(r['revision'] or '?')[:12]:>14}{r['seconds']:>7}  {r['error']}"
        )

    expected = results[0]["expected"]
    print(
        f"\nEvery count should be {expected}. A model that loads but produces fewer, or "
        "completions\nwithout an ANSWER section, has a problem worth finding now rather than "
        "16 tasks from now."
    )

    missing = sorted(set(TENSOR_PARALLEL) - {r["model"] for r in results})
    if missing:
        print(f"\n[WARN] no result yet for: {', '.join(missing)}")
    failed = [r["model"] for r in results if not r["ok"]]
    if failed:
        print(f"\n[ERROR] not ready: {', '.join(failed)}")
        return 1
    if missing:
        return 1
    print("\n[OK] all models load and generate; safe to queue the campaign")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(TENSOR_PARALLEL))
    parser.add_argument(
        "--summary", action="store_true", help="collate the per-model results and exit"
    )
    parser.add_argument("--dataset", default="ptsd", help="batch to draw the prompts from")
    parser.add_argument("--qset", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--n-prompts", type=int, default=4)
    parser.add_argument("--n-gpus", type=int, default=None, help="defaults to the model's tier")
    parser.add_argument("--max-model-len", type=int, default=32_768)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds for the model")
    parser.add_argument("--keep", action="store_true", help="keep the smoke batch and outputs")
    args = parser.parse_args(argv)

    if args.summary:
        return summarise()
    if not args.model:
        parser.error("--model is required (or pass --summary)")

    result = run_model(args.model, args)
    print(
        f"\n{result['model']}: {'OK' if result['ok'] else 'FAILED'} - "
        f"greedy {result['greedy']}/{result['expected']}, "
        f"run1 {result['run1']}/{result['expected']}, "
        f"with ANSWER {result['answered']}/{result['expected']}"
        + (f", {result['error']}" if result["error"] else "")
    )
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
