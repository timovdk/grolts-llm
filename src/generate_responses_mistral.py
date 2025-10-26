import json
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, Mistral3ForConditionalGeneration

# -------------------------
# Configuration
# -------------------------
INPUT_PATH = "./batches"
OUTPUT_PATH = "../eval/batches_out"
SUBFOLDERS = ["ptsd"]  # , "achievement", "delinquency", "wellbeing"]

QUESTION_IDS = [0]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
GENERATOR_MODEL = "mistralai/Magistral-Small-2509"
CHUNK_SIZES = [500]  # , 1000]
NEW_MAX_TOKENS = 1000
BATCH_MAX_TOKENS = 80000
MAX_PROMPT_TOKENS = 131072 - NEW_MAX_TOKENS

# -------------------------
# Load model and tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(
    GENERATOR_MODEL,
    tokenizer_type="mistral",
    padding_side="left",
    cache_dir="/projects/prjs1302/hf_cache",
)
tokenizer.padding_side = "left"
tokenizer.model_max_length = MAX_PROMPT_TOKENS

# set pad_token to eos_token safely
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

# model = AutoModelForCausalLM.from_pretrained(
model = Mistral3ForConditionalGeneration.from_pretrained(
    GENERATOR_MODEL,
    dtype=torch.bfloat16,
    device_map="auto",
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


def process_batch(
    batch_ids: List[str], batch_messages: List[List[Dict]], out_file
) -> None:
    """
    Generate outputs for a batch of prompts and write them to file.
    """
    with torch.inference_mode():
        # Tokenize current batch
        inputs = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        # Ensure pad token id is set for safety
        pad_token_id = tokenizer.pad_token_id

        input_ids = inputs if isinstance(inputs, torch.Tensor) else inputs["input_ids"]

        # Create attention mask manually (1 = token, 0 = pad)
        attention_mask = (input_ids != pad_token_id).long()

        # Generate outputs
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=NEW_MAX_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

    # Decode only generated part
    input_lengths = inputs.shape[1]
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
    batch_ids, batch_messages = [], []

    for item in tqdm(lines, desc="Processing"):
        messages = normalize_messages(item["body"]["messages"])
        projected_batch = batch_messages + [messages]

        tokenized = tokenizer.apply_chat_template(
            projected_batch,
            tokenize=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        if isinstance(tokenized, torch.Tensor):
            input_ids = tokenized
        else:
            input_ids = tokenized["input_ids"]
        effective_tokens = input_ids.numel()  # batch_size × max_seq_len

        # If adding this item exceeds the max token budget, process current batch first
        if batch_messages and effective_tokens > BATCH_MAX_TOKENS:
            process_batch(batch_ids, batch_messages, out_file)
            batch_ids, batch_messages = [], []

        # Add current sample
        batch_ids.append(item["custom_id"])
        batch_messages.append(messages)

    # Process any remaining items
    if batch_messages:
        process_batch(batch_ids, batch_messages, out_file)


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
