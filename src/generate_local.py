import json
from pathlib import Path

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# === Config ===
model_name = "microsoft/phi-4"
use_pipeline = True  # Set to False for manual generation
batch_size = 6
max_new_tokens = 1000
input_path = Path("./batches/text-embedding-3-large_gpt-4o-mini_1000_3.jsonl")
output_path = Path("phi4_1000_3.jsonl")

# === Load model and tokenizer ===
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

if use_pipeline:
    chat = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        return_full_text=False,
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
            outputs = chat(
                list(msg_batch),
                batch_size=batch_size,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
            for cid, output in zip(custom_ids, outputs):
                if isinstance(output, list):
                    output = output[0]
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
        outputs = chat(
            list(msg_batch),
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        for cid, output in zip(custom_ids, outputs):
            if isinstance(output, list):
                output = output[0]
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
