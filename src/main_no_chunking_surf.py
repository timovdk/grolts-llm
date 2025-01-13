import os

import dotenv
import pandas as pd
from tqdm import tqdm
from markitdown import MarkItDown

from grolts_prompts import get_prompt_template
from grolts_questions import get_questions

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()
tqdm.pandas()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))
CACHE_DIR = os.environ.get(("CACHE_DIR"), '~/.cache/huggingface/datasets')
EXP_ID = int(os.environ.get("EXP_ID"))
NUM_PAPERS = int(os.environ.get("NUM_PAPERS"))
DATA_DIR = os.environ.get("DATA_DIR")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL")
PROMPT_ID = int(os.environ.get("PROMPT_ID"))
OUT_DIR = os.environ.get("OUT_DIR")

QUANTIZATION = False

prompt_template = get_prompt_template(PROMPT_ID)
questions = get_questions(EXP_ID)

if QUANTIZATION:
    from transformers import BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        cache_dir=CACHE_DIR,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype="auto",
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype="auto",
    )

tokenizer = AutoTokenizer.from_pretrained(
    GENERATION_MODEL, padding_side="left", cache_dir=CACHE_DIR)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate_outputs(paper_ids, question_ids):
    all_results = []
    for paper_id in paper_ids:
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

            results.append((paper_id, question_id, messages))

        all_results.extend(results)

    messages_batch = [r[2] for r in results]
    outputs = pipeline(messages_batch, max_new_tokens=2048, batch_size=BATCH_SIZE)

    # Parse outputs
    responses = []
    for idx, generated_output in enumerate(outputs):
        paper_id, question_id, _ = all_results[idx]
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

# Loop through each paper, process questions in batches, and prepare output
paper_results = []
question_ids = list(questions.keys())
paper_ids = range(NUM_PAPERS)

# Process all papers and questions in batches
responses = generate_outputs(paper_ids, question_ids)

# Save all results to the output file
if responses:
    paper_df = pd.DataFrame(responses)
    paper_df.to_csv(output_file, mode="a", header=False, index=False)
