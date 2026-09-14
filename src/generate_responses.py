"""Generate checklist answers with a locally hosted LLM.

Replaces the four near-identical ``generate_responses_{llama,mistral,qwen3,qwen3_next}.py``
scripts, which differed only in the model id, the batch token budget and (for Magistral)
the tokenizer path.

Reads the prompt batches written by ``generate_batches.py`` and writes one JSONL of
completions per (model, dataset, chunk size, question set, run) into ``eval/batches_out``,
where ``process_batch_result.py`` parses them into CSVs.

Greedy decoding, reproducing the original runs::

    python generate_responses.py --model qwen3-30b --dataset ptsd --qset 0 --chunk 1000

Sampled reruns, to quantify run-to-run variability (reviewer 2, major 2)::

    python generate_responses.py --model qwen3-30b --dataset delinquency --qset 4 \
        --chunk 1000 --runs 5 --temperature 0.7

With ``--runs`` the output filename gains a ``_run{i}`` suffix; a single greedy run keeps
the original name so existing analyses continue to resolve. Runs already on disk are
skipped unless ``--overwrite`` is given, so a job that hits its wall time can be resubmitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import transformers
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
)

INPUT_PATH = Path("./batches")
OUTPUT_PATH = Path("../eval/batches_out")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
NEW_MAX_TOKENS = 1500

# Where the HF weights live. Overridable so the script is not pinned to one cluster.
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/projects/prjs1302/hf_cache")


@dataclass(frozen=True)
class ModelSpec:
    """One generator model.

    ``batch_max_tokens`` is the packing budget (batch_size x padded_seq_len) tuned per
    model to fit an H100; ``family`` selects the tokenizer and loading path.
    """

    repo: str
    batch_max_tokens: int
    family: str = "causal"
    revision: str | None = None


MODELS: Dict[str, ModelSpec] = {
    "qwen3-30b": ModelSpec("Qwen/Qwen3-30B-A3B-Instruct-2507", 90_000),
    "qwen3-next-80b": ModelSpec("Qwen/Qwen3-Next-80B-A3B-Instruct", 70_000),
    "llama-3.3-70b": ModelSpec("meta-llama/Llama-3.3-70B-Instruct", 60_000),
    "magistral-small": ModelSpec("mistralai/Magistral-Small-2509", 80_000, "mistral"),
}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def load_model(spec: ModelSpec):
    """Load tokenizer and model for one spec, following that family's requirements."""
    if spec.family == "mistral":
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo,
            tokenizer_type="mistral",
            padding_side="left",
            fix_mistral_regex=True,
            cache_dir=CACHE_DIR,
            revision=spec.revision,
        )
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = 131_072 - NEW_MAX_TOKENS
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

        model = Mistral3ForConditionalGeneration.from_pretrained(
            spec.repo, dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR,
            revision=spec.revision,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=CACHE_DIR,
            revision=spec.revision,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
            revision=spec.revision,
        )

    model.eval()
    return tokenizer, model


def normalize_messages(raw_messages: List[Dict]) -> List[Dict]:
    """Flatten any structured 'text' content lists into plain strings."""
    norm: List[Dict] = []
    for m in raw_messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(part["text"] for part in content if part["type"] == "text")
        norm.append({"role": m["role"], "content": content})
    return norm


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------


def tokenize_batch(tokenizer, spec: ModelSpec, batch: List[List[Dict]], truncate: bool = True):
    """Tokenize a batch of message lists into padded model inputs."""
    if spec.family == "mistral":
        # Magistral's tokenizer applies its template and tokenizes in one step.
        out = tokenizer.apply_chat_template(
            batch, tokenize=True, return_tensors="pt", padding=True, truncation=truncate
        )
        input_ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return input_ids, attention_mask

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=truncate)
    return encoded["input_ids"], encoded["attention_mask"]


def context_limit(tokenizer, model) -> int | None:
    """The model's usable context length, or None if it cannot be determined.

    ``tokenizer.model_max_length`` is a huge sentinel on some repos, so fall back to the
    config's positional limit.
    """
    limit = getattr(tokenizer, "model_max_length", None)
    if not isinstance(limit, int) or limit > 10_000_000:
        limit = getattr(model.config, "max_position_embeddings", None)
    return limit if isinstance(limit, int) and limit > 0 else None


def check_prompt_lengths(tokenizer, model, spec: ModelSpec, lines) -> dict:
    """Fail before generating if any prompt would be silently truncated.

    Generation passes ``truncation=True``, which would quietly cut an over-long prompt and
    produce a plausible-looking answer to a question the model never fully saw. Measuring
    the untruncated lengths up front turns that into a loud error.
    """
    lengths = []
    for item in tqdm(lines, desc="Checking prompt lengths"):
        messages = normalize_messages(item["body"]["messages"])
        ids, _ = tokenize_batch(tokenizer, spec, [messages], truncate=False)
        lengths.append(int(ids.shape[1]))

    longest = max(lengths)
    limit = context_limit(tokenizer, model)
    budget = None if limit is None else limit - NEW_MAX_TOKENS

    print(
        f"[INFO] prompt tokens: max {longest}, median {sorted(lengths)[len(lengths) // 2]}"
        f" | context limit {limit} - {NEW_MAX_TOKENS} generated = {budget}"
    )
    if budget is not None and longest > budget:
        raise SystemExit(
            f"[ERROR] the longest prompt is {longest} tokens but only {budget} fit alongside "
            f"{NEW_MAX_TOKENS} generated tokens. Generation would silently truncate it; "
            "shorten the retrieved context or lower NEW_MAX_TOKENS rather than proceeding."
        )
    if budget is None:
        print("[WARN] could not determine the context limit; truncation cannot be ruled out.")
    return {"max_prompt_tokens": longest, "context_limit": limit, "prompt_budget": budget}


def run_metadata(spec: ModelSpec, args, model, run, source: Path, length_check: dict) -> dict:
    """Everything needed to identify exactly what produced a run, recorded beside it.

    The original runs recorded none of this, which is why the gpt-5-mini snapshot had to be
    recovered from the response payloads and the local model revisions cannot be recovered
    at all.
    """
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "model": spec.repo,
        # Resolved by transformers from whatever snapshot was actually loaded.
        "revision_requested": spec.revision,
        "revision_loaded": getattr(model.config, "_commit_hash", None),
        "dataset": args.dataset,
        "qset": args.qset,
        "chunk": args.chunk,
        "run": run,
        "seed": None if run is None else args.seed + run,
        "decoding": (
            {"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
            if args.temperature > 0
            else {"do_sample": False}
        ),
        "max_new_tokens": NEW_MAX_TOKENS,
        "prompt_file": source.name,
        "prompt_sha256": digest,
        "n_prompts": sum(1 for _ in source.open(encoding="utf-8")),
        **length_check,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "host": platform.node(),
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def process_batch(
    tokenizer, model, spec: ModelSpec, args, batch_ids, batch_messages, out_file
) -> None:
    """Generate for one packed batch and append the completions to ``out_file``."""
    with torch.inference_mode():
        input_ids, attention_mask = tokenize_batch(tokenizer, spec, batch_messages)
        input_ids = input_ids.to(model.device)
        attention_mask = attention_mask.to(model.device)

        generation = dict(
            max_new_tokens=NEW_MAX_TOKENS,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        if args.temperature > 0:
            generation.update(
                do_sample=True, temperature=args.temperature, top_p=args.top_p
            )
        else:
            generation.update(do_sample=False)

        outputs = model.generate(input_ids, attention_mask=attention_mask, **generation)

    prompt_length = input_ids.shape[1]
    completions = tokenizer.batch_decode(
        [o[prompt_length:] for o in outputs], skip_special_tokens=True
    )
    for cid, completion in zip(batch_ids, completions):
        out_file.write(json.dumps({"custom_id": cid, "completion": completion}) + "\n")


def generate_responses(tokenizer, model, spec: ModelSpec, args, lines, out_file) -> None:
    """Work through every prompt, packing batches up to the model's token budget."""
    batch_ids: List[str] = []
    batch_messages: List[List[Dict]] = []

    for item in tqdm(lines, desc="Processing"):
        messages = normalize_messages(item["body"]["messages"])

        # Would adding this prompt push the padded batch over budget?
        projected, _ = tokenize_batch(tokenizer, spec, batch_messages + [messages])
        if batch_messages and projected.numel() > spec.batch_max_tokens:
            process_batch(
                tokenizer, model, spec, args, batch_ids, batch_messages, out_file
            )
            batch_ids, batch_messages = [], []

        batch_ids.append(item["custom_id"])
        batch_messages.append(messages)

    if batch_messages:
        process_batch(tokenizer, model, spec, args, batch_ids, batch_messages, out_file)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def output_name(spec: ModelSpec, args, run: int | None) -> Path:
    stem = (
        f"{EMBEDDING_MODEL.replace('/', '_')}_{spec.repo.replace('/', '_')}"
        f"_{args.dataset}_{args.chunk}_{args.qset}"
    )
    if run is not None:
        stem += f"_run{run}"
    return OUTPUT_PATH / f"{stem}.jsonl"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--qset", type=int, default=4, help="0 = GRoLTS v1, 4 = v2")
    parser.add_argument("--chunk", type=int, default=1000)
    parser.add_argument(
        "--runs", type=int, default=1,
        help="number of independent runs; >1 adds a _run{i} suffix to the output",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 selects greedy decoding (the original setting); >0 enables sampling",
    )
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--seed", type=int, default=1000,
        help="base seed; run i uses seed + i so runs differ but stay reproducible",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--show-revisions", action="store_true",
        help="read the .meta.json sidecars already written and print a MODELS block with the "
        "loaded revisions filled in, ready to paste back into this file; runs nothing",
    )
    args = parser.parse_args(argv)

    if args.show_revisions:
        return args
    if args.runs > 1 and args.temperature == 0:
        parser.error(
            "--runs > 1 with greedy decoding would repeat the same output; "
            "pass --temperature 0.7 (or another value > 0) to sample."
        )
    return args


def show_revisions() -> int:
    """Print a MODELS block with each model's loaded revision filled in.

    The commit hash of the snapshot that produced a run cannot be recovered after the fact --
    that is why the original runs' local model versions are unknown. Once a run has written its
    sidecar, paste this back in to pin future runs to the same weights.
    """
    found: Dict[str, str] = {}
    for meta_file in sorted(OUTPUT_PATH.glob("*.meta.json")):
        meta = json.loads(meta_file.read_text())
        revision = meta.get("revision_loaded")
        if revision:
            found.setdefault(meta["model"], revision)

    if not found:
        print(f"[WARN] no revisions recorded yet in {OUTPUT_PATH}/*.meta.json")
        return 1

    print("MODELS: Dict[str, ModelSpec] = {")
    for name, spec in MODELS.items():
        revision = found.get(spec.repo)
        family = f', "{spec.family}"' if spec.family != "causal" else ""
        pin = f',\n        revision="{revision}",\n    ' if revision else ""
        print(f'    "{name}": ModelSpec("{spec.repo}", {spec.batch_max_tokens:_}{family}{pin}),')
    print("}")
    missing = [s.repo for s in MODELS.values() if s.repo not in found]
    if missing:
        print(f"\n[WARN] no run recorded yet for: {', '.join(missing)}")
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.show_revisions:
        return show_revisions()
    spec = MODELS[args.model]
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    input_path = (
        INPUT_PATH
        / f"{EMBEDDING_MODEL.replace('/', '_')}_generic_{args.dataset}"
        f"_{args.chunk}_{args.qset}.jsonl"
    )
    if not input_path.exists():
        print(f"[ERROR] no prompt batch at {input_path}", file=sys.stderr)
        return 1

    with open(input_path, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    if not lines:
        print(f"[ERROR] {input_path} is empty", file=sys.stderr)
        return 1

    runs = [None] if args.runs == 1 else list(range(1, args.runs + 1))
    pending = [r for r in runs if args.overwrite or not output_name(spec, args, r).exists()]
    if not pending:
        print("[INFO] every requested run already exists; nothing to do.")
        return 0

    print(f"[INFO] {spec.repo} | {input_path.name} | {len(lines)} prompts")
    print(
        f"[INFO] decoding: "
        + (
            f"sampling (temperature={args.temperature}, top_p={args.top_p})"
            if args.temperature > 0
            else "greedy"
        )
        + f" | runs pending: {pending}"
    )

    tokenizer, model = load_model(spec)
    resolved = getattr(model.config, "_commit_hash", None)
    print(f"[INFO] loaded revision: {resolved or 'unknown'}")
    length_check = check_prompt_lengths(tokenizer, model, spec, lines)

    for run in pending:
        if run is not None:
            # Per-run seed: runs differ from each other but the set is reproducible.
            torch.manual_seed(args.seed + run)
        output_path = output_name(spec, args, run)
        print(f"[INFO] -> {output_path}", flush=True)
        with open(output_path, "w", encoding="utf-8") as f_out:
            generate_responses(tokenizer, model, spec, args, lines, f_out)

        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(run_metadata(spec, args, model, run, input_path, length_check), indent=2)
        )
        print(f"[INFO] done: {output_path}  (provenance -> {meta_path.name})", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
