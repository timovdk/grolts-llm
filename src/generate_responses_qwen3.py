import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Configuration
# -------------------------
INPUT_PATH = "./batches"
OUTPUT_PATH = "../eval/batches_out"
SUBFOLDERS = ["ptsd"]  # , "achievement", "delinquency", "wellbeing"]

QUESTION_IDS = [0]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
GENERATOR_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHUNK_SIZES = [500]  # , 1000]
NEW_MAX_TOKENS = 1000
BATCH_MAX_TOKENS = 90000

# -------------------------
# Load model and tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    GENERATOR_MODEL,
    trust_remote_code=True,
    padding_side="left",
    cache_dir="/projects/prjs1302/hf_cache",
)

model = AutoModelForCausalLM.from_pretrained(
    GENERATOR_MODEL,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
    cache_dir="/projects/prjs1302/hf_cache",
)
model.eval()


def normalize_messages(raw_messages: List[Dict]) -> List[Dict]:
    """
    Normalize messages: flatten any 'text' content lists into strings.
    """
    norm: List[Dict] = []
    for m in raw_messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(
                part["text"] for part in content if part["type"] == "text"
            )
        norm.append({"role": m["role"], "content": content})
    return norm


def process_batch(batch_ids: List[str], batch_prompts: List[str], out_file) -> None:
    """
    Generate outputs for a batch of prompts and write them to file.
    """
    with torch.inference_mode():
        # Tokenize current batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Generate outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=NEW_MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    # Decode only generated part
    input_lengths = inputs["input_ids"].shape[1]
    completions = tokenizer.batch_decode(
        [o[input_lengths:] for o in outputs], skip_special_tokens=True
    )

    # Write directly to file
    for cid, comp in zip(batch_ids, completions):
        out_file.write(json.dumps({"custom_id": cid, "completion": comp}) + "\n")


def generate_responses(lines: List[Dict], out_file) -> None:
    """
    Process all items in lines, batching by token budget.
    """
    batch_ids, batch_prompts = [], []

    for item in tqdm(lines, desc="Processing"):
        messages = normalize_messages(item["body"]["messages"])
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Project batch size if we add this prompt
        projected_prompts = batch_prompts + [prompt]
        inputs = tokenizer(
            projected_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        effective_tokens = inputs["input_ids"].numel()  # batch_size × max_seq_len

        # If adding this item exceeds the max token budget, process current batch first
        if batch_prompts and effective_tokens > BATCH_MAX_TOKENS:
            process_batch(batch_ids, batch_prompts, out_file)
            batch_ids, batch_prompts = [], []

        # Add current sample
        batch_ids.append(item["custom_id"])
        batch_prompts.append(prompt)

    # Process any remaining items
    if batch_prompts:
        process_batch(batch_ids, batch_prompts, out_file)


def main() -> None:
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for subfolder in SUBFOLDERS:
        for chunk_size in CHUNK_SIZES:
            for q_id in QUESTION_IDS:
                # Open in and output files
                input_path = f"{INPUT_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_generic_{subfolder}_{chunk_size}_{q_id}.jsonl"
                output_path = f"{OUTPUT_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{GENERATOR_MODEL.replace('/', '_')}_{subfolder}_{chunk_size}_{q_id}.jsonl"

                print(f"[INFO] Processing {input_path}, saving to {output_path}")

                with (
                    open(input_path, "r", encoding="utf-8") as f_in,
                    open(output_path, "w", encoding="utf-8") as f_out,
                ):
                    lines = [json.loads(line) for line in f_in]
                    if not lines:
                        print(f"[WARN] No lines found in {input_path}, skipping.")
                        continue
                    generate_responses(lines, f_out)

                print(f"[INFO] Done processing {input_path}, saved to {output_path}")

    print("[INFO] Done processing all files.")


if __name__ == "__main__":
    main()
