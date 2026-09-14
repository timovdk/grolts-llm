"""Generate checklist answers with vLLM instead of plain transformers.

Same inputs, same outputs, same CLI as ``generate_responses.py`` — it reads the prompt
batches from ``./batches`` and writes one JSONL of completions per run into
``../eval/batches_out``. The model registry is imported from ``generate_responses`` so the
two paths cannot drift apart.

Why vLLM: the transformers path pads every batch to its longest prompt and generates in
lockstep. With ~13k-token prompts of uneven length that wastes a great deal of compute;
vLLM's continuous batching does not.

One job produces the whole campaign for a model/dataset, from a single model load::

    python generate_responses_vllm.py --model qwen3-30b --dataset ptsd --qset 4 \
        --chunk 1000 --runs 5 --temperature 0.7 --also-greedy --n-gpus 1

Output names carry a ``_vllm`` tag, so nothing here can overwrite the published
transformers runs:

* ``..._<chunk>_<qset>_vllm.jsonl``        the greedy pass
* ``..._<chunk>_<qset>_vllm_run<i>.jsonl`` each sampled run

Keeping a greedy pass under the *same engine* is what makes the comparison interpretable.
Without it, any difference between the published greedy numbers and these sampled runs
would confound sampling noise with the change of inference engine. With it, greedy-vs-
sampled is a clean estimate of sampling noise, and vLLM-greedy vs transformers-greedy is a
separate, meaningful engine-sensitivity check.

Note that vLLM's greedy decoding is not bit-reproducible: continuous batching means a
request's numerics depend on which other requests share its batch. Do not describe these
runs as deterministic the way the transformers runs were.

Interrupted runs resume: completed ``custom_id``s are read back from the partial output
file and skipped, so a job killed by the wall clock picks up where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
from transformers import AutoConfig

from generate_responses import (
    CACHE_DIR,
    EMBEDDING_MODEL,
    INPUT_PATH,
    MODELS,
    NEW_MAX_TOKENS,
    OUTPUT_PATH,
    ModelSpec,
    normalize_messages,
)

#: The context window must hold the prompt plus what is generated. Prompts average ~13.3k
#: tokens but the tail is long: the largest measured is 21,605 (gpt-5-mini tokenizer), and
#: 21% of requests exceed 15k. 32k leaves room for that tail and for local tokenizers that
#: count somewhat differently. vLLM rejects anything longer rather than truncating silently,
#: so a value that is too small fails loudly -- but only after the model has loaded.
DEFAULT_MAX_MODEL_LEN = 32_768


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------


def load_prompts(path: Path) -> List[Dict]:
    """Read one prompt batch into ``[{custom_id, messages}, ...]``."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompts.append(
                {
                    "custom_id": item["custom_id"],
                    "messages": normalize_messages(item["body"]["messages"]),
                }
            )
    return prompts


def completed_ids(path: Path) -> set[str]:
    """Which requests a partial output file already holds."""
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["custom_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def output_name(spec: ModelSpec, args, run: int | None) -> Path:
    stem = (
        f"{EMBEDDING_MODEL.replace('/', '_')}_{spec.repo.replace('/', '_')}"
        f"_{args.dataset}_{args.chunk}_{args.qset}_vllm"
    )
    if run is not None:
        stem += f"_run{run}"
    return OUTPUT_PATH / f"{stem}.jsonl"


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def build_engine(spec: ModelSpec, args):
    """Start the vLLM engine for one model."""
    from vllm import LLM

    kwargs = dict(
        model=spec.repo,
        tensor_parallel_size=args.n_gpus,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        download_dir=CACHE_DIR,
        enforce_eager=args.enforce_eager,
    )
    if spec.revision:
        kwargs["revision"] = spec.revision
    if spec.family == "mistral":
        # Magistral needs its own tokenizer, exactly as the transformers path does.
        kwargs["tokenizer_mode"] = "mistral"
    return LLM(**kwargs)


def generate_run(llm, prompts: List[Dict], out_path: Path, sampling_params, batch_size: int) -> int:
    """Generate one run, appending as it goes so an interrupted job can resume."""
    done = completed_ids(out_path)
    pending = [p for p in prompts if p["custom_id"] not in done]
    if done:
        print(f"[INFO] resuming: {len(done)} already done, {len(pending)} to go", flush=True)
    if not pending:
        print(f"[INFO] {out_path.name} already complete")
        return 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out_path, "a", encoding="utf-8") as f_out:
        for start in tqdm(range(0, len(pending), batch_size), desc=out_path.stem[-28:]):
            batch = pending[start : start + batch_size]
            outputs = llm.chat([p["messages"] for p in batch], sampling_params)
            for prompt, output in zip(batch, outputs):
                f_out.write(
                    json.dumps(
                        {
                            "custom_id": prompt["custom_id"],
                            "completion": output.outputs[0].text,
                        }
                    )
                    + "\n"
                )
                written += 1
            f_out.flush()
    return written


def resolve_revision(spec: ModelSpec) -> str | None:
    """The commit hash of the snapshot actually being used.

    Read from the config alone, which is cheap, and matches what the transformers path
    records so the two are comparable.
    """
    try:
        config = AutoConfig.from_pretrained(
            spec.repo, cache_dir=CACHE_DIR, revision=spec.revision, trust_remote_code=True
        )
        return getattr(config, "_commit_hash", None)
    except Exception as error:  # provenance is best-effort; never fail a run over it
        print(f"[WARN] could not resolve revision: {error}")
        return None


def run_metadata(spec: ModelSpec, args, run: int | None, source: Path, revision: str | None) -> dict:
    """Everything needed to identify exactly what produced a run, recorded beside it."""
    import vllm

    return {
        "engine": "vllm",
        "vllm": vllm.__version__,
        "model": spec.repo,
        "revision_requested": spec.revision,
        "revision_loaded": revision,
        "dataset": args.dataset,
        "qset": args.qset,
        "chunk": args.chunk,
        "run": run,
        "seed": None if run is None else args.seed + run,
        "decoding": (
            {"temperature": 0.0}
            if run is None
            else {"temperature": args.temperature, "top_p": args.top_p}
        ),
        "max_new_tokens": NEW_MAX_TOKENS,
        "engine_args": {
            "tensor_parallel_size": args.n_gpus,
            "max_model_len": args.max_model_len,
            "max_num_seqs": args.max_num_seqs,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": args.enforce_eager,
        },
        "prompt_file": source.name,
        "prompt_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "n_prompts": sum(1 for _ in source.open(encoding="utf-8")),
        "host": platform.node(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qset", type=int, default=4, help="0 = GRoLTS v1, 4 = v2")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5, help="number of sampled runs")
    parser.add_argument(
        "--also-greedy", action="store_true",
        help="additionally produce a greedy pass from the same model load, so the sampled "
        "runs have a same-engine baseline to be compared against",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    # vLLM engine settings
    parser.add_argument("--n-gpus", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args(argv)

    if args.runs < 0:
        parser.error("--runs cannot be negative")
    if args.runs == 0 and not args.also_greedy:
        parser.error("nothing to do: pass --runs > 0, --also-greedy, or both")
    return args


def main(argv=None) -> int:
    from vllm import SamplingParams

    args = parse_args(argv)
    spec = MODELS[args.model]
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    source = (
        INPUT_PATH
        / f"{EMBEDDING_MODEL.replace('/', '_')}_generic_{args.dataset}"
        f"_{args.chunk}_{args.qset}.jsonl"
    )
    if not source.exists():
        print(f"[ERROR] no prompt batch at {source}", file=sys.stderr)
        return 1

    prompts = load_prompts(source)
    if not prompts:
        print(f"[ERROR] {source} is empty", file=sys.stderr)
        return 1

    # None marks the greedy pass; 1..n are the sampled runs.
    planned: List[int | None] = ([None] if args.also_greedy else []) + list(
        range(1, args.runs + 1)
    )
    if args.overwrite:
        for run in planned:
            output_name(spec, args, run).unlink(missing_ok=True)

    pending = [
        run
        for run in planned
        if len(completed_ids(output_name(spec, args, run))) < len(prompts)
    ]
    if not pending:
        print("[INFO] every requested run is already complete; nothing to do.")
        return 0

    print(f"[INFO] {spec.repo} | {source.name} | {len(prompts)} prompts")
    print(
        f"[INFO] pending: {['greedy' if r is None else f'run{r}' for r in pending]}"
        f" | max_model_len {args.max_model_len} | {args.n_gpus} GPU(s)"
    )

    revision = resolve_revision(spec)
    print(f"[INFO] revision: {revision or 'unknown'}")

    llm = build_engine(spec, args)

    for run in pending:
        out_path = output_name(spec, args, run)
        if run is None:
            sampling = SamplingParams(temperature=0.0, max_tokens=NEW_MAX_TOKENS)
        else:
            sampling = SamplingParams(
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=NEW_MAX_TOKENS,
                seed=args.seed + run,
            )
        print(f"[INFO] -> {out_path.name}", flush=True)
        generate_run(llm, prompts, out_path, sampling, args.batch_size)
        out_path.with_suffix(".meta.json").write_text(
            json.dumps(run_metadata(spec, args, run, source, revision), indent=2)
        )
        print(f"[INFO] done: {out_path.name}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
