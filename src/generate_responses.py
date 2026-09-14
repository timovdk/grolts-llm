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
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

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
        from transformers import Mistral3ForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo,
            tokenizer_type="mistral",
            padding_side="left",
            fix_mistral_regex=True,
            cache_dir=CACHE_DIR,
        )
        tokenizer.padding_side = "left"
        tokenizer.model_max_length = 131_072 - NEW_MAX_TOKENS
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

        model = Mistral3ForConditionalGeneration.from_pretrained(
            spec.repo, dtype=torch.bfloat16, device_map="auto", cache_dir=CACHE_DIR
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            spec.repo,
            trust_remote_code=True,
            padding_side="left",
            cache_dir=CACHE_DIR,
        )
        model = AutoModelForCausalLM.from_pretrained(
            spec.repo,
            dtype="auto",
            device_map="auto",
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
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


def tokenize_batch(tokenizer, spec: ModelSpec, batch: List[List[Dict]]):
    """Tokenize a batch of message lists into padded model inputs."""
    if spec.family == "mistral":
        # Magistral's tokenizer applies its template and tokenizes in one step.
        out = tokenizer.apply_chat_template(
            batch, tokenize=True, return_tensors="pt", padding=True, truncation=True
        )
        input_ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
        attention_mask = (input_ids != tokenizer.pad_token_id).long()
        return input_ids, attention_mask

    prompts = [
        tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch
    ]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
    return encoded["input_ids"], encoded["attention_mask"]


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
    args = parser.parse_args(argv)

    if args.runs > 1 and args.temperature == 0:
        parser.error(
            "--runs > 1 with greedy decoding would repeat the same output; "
            "pass --temperature 0.7 (or another value > 0) to sample."
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
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

    for run in pending:
        if run is not None:
            # Per-run seed: runs differ from each other but the set is reproducible.
            torch.manual_seed(args.seed + run)
        output_path = output_name(spec, args, run)
        print(f"[INFO] -> {output_path}", flush=True)
        with open(output_path, "w", encoding="utf-8") as f_out:
            generate_responses(tokenizer, model, spec, args, lines, f_out)
        print(f"[INFO] done: {output_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
