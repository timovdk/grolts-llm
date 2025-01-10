import os

import dotenv
import pandas as pd
from tqdm import tqdm
from markitdown import MarkItDown

from grolts_prompts import get_prompt_template
from grolts_questions import get_questions

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

cache_dir = "/projects/2/managed_datasets/hf_cache_dir"

dotenv.load_dotenv()
tqdm.pandas()

# Batch size for processing
BATCH_SIZE = 8

EXP_ID = int(os.environ.get("EXP_ID"))
NUM_PAPERS = int(os.environ.get("NUM_PAPERS"))
DATA_DIR = os.environ.get("DATA_DIR")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL")
PROMPT_ID = int(os.environ.get("PROMPT_ID"))
OUT_DIR = os.environ.get("OUT_DIR")

prompt_template = get_prompt_template(PROMPT_ID)
questions = get_questions(EXP_ID)

# quantization_config = BitsAndBytesConfig(load_in_4bit=True, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    GENERATION_MODEL,
    # cache_dir=cache_dir,
    # quantization_config=quantization_config,
    device_map="auto",
    torch_dtype="auto",
)

# model = AutoModelForCausalLM.from_pretrained(GENERATION_MODEL, cache_dir=cache_dir, device_map='auto', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL, padding_side="left"
)  # , cache_dir=cache_dir)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_outputs(paper_id, question_ids):
    results = []
    for question_id in question_ids:
        md = MarkItDown()

        context_text = "\n\n---\n\n".join(
            md.convert(f"./data/{paper_id}.pdf").text_content
        )
        question = questions[question_id]
        sys_prompt = prompt_template["system"]
        user_prompt = prompt_template["user"].format(
            question=question, context=context_text
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]

        results.append((question_id, messages))

    messages_batch = [r[1] for r in results]
    outputs = pipeline(messages_batch, max_new_tokens=2048, batch_size=BATCH_SIZE)

    # Parse outputs
    responses = []
    for idx, generated_output in enumerate(outputs):
        question_id = results[idx][0]
        response_text = generated_output[0]["generated_text"][-1]["content"]
        response = {"paper_id": paper_id, "question_id": question_id}
        current_section = None

        for item in response_text.split("\n"):
            item = item.strip()
            if item.startswith("ANSWER"):
                current_section = "answer"
                response["answer"] = item.replace("ANSWER:", "").strip()
            elif item.startswith("REASONING"):
                current_section = "reasoning"
                response["reasoning"] = item.replace("REASONING:", "").strip()
            elif item.startswith("EVIDENCE"):
                current_section = "evidence"
                response["evidence"] = item.replace("EVIDENCE:", "").strip()
            elif current_section:
                response[current_section] = (
                    response.get(current_section, "") + " " + item
                )
        responses.append(response)
    return responses


output_file = os.path.join(
    OUT_DIR, f"{GENERATION_MODEL.replace('/', '-')}-p{EXP_ID + 1}.csv"
)

pd.DataFrame(
    columns=["paper_id", "question_id", "reasoning", "evidence", "answer"]
).to_csv(output_file, index=False)

# Loop through each paper
for paper_id in tqdm(range(NUM_PAPERS), desc="Processing Papers"):
    paper_results = []

    # Process questions in batches
    question_ids = list(questions.keys())
    for i in range(0, len(question_ids), BATCH_SIZE):
        batch_question_ids = question_ids[i : i + BATCH_SIZE]
        outputs = generate_outputs(paper_id, batch_question_ids)
        paper_results.extend(outputs)

    # Convert the results to a DataFrame
    if paper_results:
        paper_df = pd.DataFrame(paper_results)

        # Append the DataFrame to the CSV file
        paper_df.to_csv(output_file, mode="a", header=False, index=False)
