import json
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
import torch
from tqdm import tqdm

# === Config ===
model_name = "microsoft/phi-4"  # or "Qwen/Qwen2-7B-Instruct"
use_pipeline = True  # Set to False for Qwen3-style manual generation
batch_size = 4
max_new_tokens = 1024
input_path = Path("text-embedding-3-large_gpt-4o-mini_1000_3.jsonl")
output_path = Path("batched_results.jsonl")

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

if use_pipeline:
    chat = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )


# === Helper: flatten OpenAI message content ===
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


# === Load input JSONL ===
with open(input_path, "r") as f:
    lines = [json.loads(line) for line in f]

results = []
batch = []

for item in tqdm(lines, desc="Processing"):
    messages = normalize_messages(item["body"]["messages"])
    batch.append((item["custom_id"], messages))

    if len(batch) == batch_size:
        if use_pipeline:
            # === PIPELINE MODE ===
            custom_ids, msg_batch = zip(*batch)
            outputs = chat(list(msg_batch), max_new_tokens=max_new_tokens)
            for cid, output in zip(custom_ids, outputs):
                results.append(
                    {"custom_id": cid, "completion": output["generated_text"]}
                )
        else:
            # === MANUAL TEMPLATE + GENERATE MODE ===
            custom_ids, msg_batch = zip(*batch)
            prompts = [
                tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in msg_batch
            ]
            inputs = tokenizer(
                prompts,
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

            for cid, input_ids, output_ids in zip(
                custom_ids, inputs["input_ids"], outputs
            ):
                gen_tokens = output_ids[len(input_ids) :]
                completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                results.append({"custom_id": cid, "completion": completion})

        batch = []

# === Process any remaining examples ===
if batch:
    custom_ids, msg_batch = zip(*batch)
    if use_pipeline:
        outputs = chat(list(msg_batch), max_new_tokens=max_new_tokens)
        for cid, output in zip(custom_ids, outputs):
            results.append({"custom_id": cid, "completion": output["generated_text"]})
    else:
        prompts = [
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            for messages in msg_batch
        ]
        inputs = tokenizer(
            prompts,
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

        for cid, input_ids, output_ids in zip(custom_ids, inputs["input_ids"], outputs):
            gen_tokens = output_ids[len(input_ids) :]
            completion = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            results.append({"custom_id": cid, "completion": completion})

# === Write results ===
with open(output_path, "w") as out_f:
    for r in results:
        out_f.write(json.dumps(r) + "\n")

print(f"✅ Done. Saved results to: {output_path}")
