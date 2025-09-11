import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# -------------------------
# Configuration
# -------------------------
model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
max_tokens_per_batch = 100000
max_new_tokens = 1000
input_path = Path("./batches/Qwen3_input.jsonl")
output_path = Path("./batches_out/Qwen3_responses.jsonl")

# -------------------------
# Load model and tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)
model.eval()


# -------------------------
# Helper to normalize messages
# -------------------------
def normalize_messages(raw_messages):
    norm = []
    for m in raw_messages:
        content = m["content"]
        if isinstance(content, list):
            content = "".join(
                part["text"] for part in content if part["type"] == "text"
            )
        norm.append({"role": m["role"], "content": content})
    return norm


# -------------------------
# Open in and output files
# -------------------------
with open(input_path, "r") as f:
    lines = [json.loads(line) for line in f]

output_path.parent.mkdir(parents=True, exist_ok=True)
out_file = open(output_path, "w")

# -------------------------
# Inference loop with batching
# -------------------------
batch_ids, batch_prompts = [], []

with torch.inference_mode():
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
        if batch_prompts and effective_tokens > max_tokens_per_batch:
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
                max_new_tokens=max_new_tokens,
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
                out_file.write(
                    json.dumps({"custom_id": cid, "completion": comp}) + "\n"
                )

            # Reset batch
            batch_ids, batch_prompts = [], []

        # Add current sample
        batch_ids.append(item["custom_id"])
        batch_prompts.append(prompt)

    # Process any remaining items
    if batch_prompts:
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
        )

        input_lengths = inputs["input_ids"].shape[1]
        completions = tokenizer.batch_decode(
            [o[input_lengths:] for o in outputs], skip_special_tokens=True
        )

        for cid, comp in zip(batch_ids, completions):
            out_file.write(json.dumps({"custom_id": cid, "completion": comp}) + "\n")

out_file.close()
print(f"✅ Done. Saved results to: {output_path}")
