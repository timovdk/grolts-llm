"""Shared configuration for the generation pipeline.

Importing this loads the repo-root ``.env`` (see ``.env.example``), which is where
``HF_TOKEN`` lives.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

INPUT_PATH = Path("./batches")
OUTPUT_PATH = Path("../eval/batches_out")
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
NEW_MAX_TOKENS = 1500
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"

load_dotenv(ENV_FILE)

# Where the HF weights live. Overridable so the scripts are not pinned to one cluster.
CACHE_DIR = os.environ.get("HF_CACHE_DIR", "/projects/prjs1302/hf_cache")


@dataclass(frozen=True)
class ModelSpec:
    """One generator model.

    ``batch_max_tokens`` is the transformers packing budget (batch_size x padded_seq_len),
    tuned per model to fit an H100; vLLM manages batching itself and ignores it.
    ``family`` selects the tokenizer and loading path.
    """

    repo: str
    batch_max_tokens: int
    family: str = "causal"
    #: Optional commit hash
    revision: str | None = None
    #: GPUs this model needs, mirroring the run_reruns_*gpu.sh arrays.
    tensor_parallel: int = 1


MODELS: dict[str, ModelSpec] = {
    "qwen3-30b": ModelSpec("Qwen/Qwen3-30B-A3B-Instruct-2507", 90_000),
    "qwen3-next-80b": ModelSpec(
        "Qwen/Qwen3-Next-80B-A3B-Instruct", 70_000, tensor_parallel=2
    ),
    "llama-3.3-70b": ModelSpec(
        "meta-llama/Llama-3.3-70B-Instruct", 60_000, tensor_parallel=2
    ),
    "magistral-small": ModelSpec("mistralai/Magistral-Small-2509", 80_000, "mistral"),
}


def normalize_messages(raw_messages: list[dict]) -> list[dict]:
    """Flatten any structured 'text' content lists into plain strings."""
    norm: list[dict] = []
    for m in raw_messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(
                part["text"] for part in content if part["type"] == "text"
            )
        norm.append({"role": m["role"], "content": content})
    return norm


def prompt_batch(dataset: str, chunk: int, qset: int) -> Path:
    """The prompt file every model reads for one dataset/chunk/checklist version."""
    return (
        INPUT_PATH
        / f"{EMBEDDING_MODEL.replace('/', '_')}_generic_{dataset}_{chunk}_{qset}.jsonl"
    )
