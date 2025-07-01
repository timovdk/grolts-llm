import base64
import glob
import os
import pickle
import re
import traceback

import chromadb
import pandas as pd
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from openai import OpenAI
from tqdm import tqdm

from grolts_prompts import get_prompt_template
from grolts_questions import get_questions

API_KEY = ""

DATA_PATH = "./data"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"
PROCESSED_PATH = "./processed_pdfs"
OUTPUT_PATH = "./outputs"

QUESTION_ID = 0
PROMPT_ID = 1
TOP_K = 5
#EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
GENERATOR_MODEL = "gpt-4o-mini"
MULTI_MODAL_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 512

USE_CHUNKING = True
MULTI_MODAL = True

os.makedirs(OUTPUT_PATH, exist_ok=True)

document_collection_name = f"{EMBEDDING_MODEL.replace('/', '_')}_{CHUNK_SIZE}"
question_embedding_file = (
    f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{QUESTION_ID}.pkl"
)
document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{document_collection_name}"
output_file = f"{OUTPUT_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{GENERATOR_MODEL}_{CHUNK_SIZE}_{QUESTION_ID}.csv"

prompt_template = get_prompt_template(PROMPT_ID)

chroma_client = chromadb.PersistentClient(path=document_embedding_file)
collection = chroma_client.get_or_create_collection(document_collection_name)

# generator = HuggingFaceLocalGenerator(model="google/flan-t5-base")
generator = OpenAIGenerator(api_key=Secret.from_token(API_KEY), model=GENERATOR_MODEL)

results = []


def generate_answer_with_gpt4(prompt: str, image_paths: list = None) -> str:
    messages = [{"role": "user", "content": []}]

    client = OpenAI(api_key=API_KEY)

    # Add text prompt
    messages[0]["content"].append({"type": "input_text", "text": prompt})

    # If images are included
    if image_paths:
        for path in image_paths:
            with open(path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                messages[0]["content"].append(
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{img_b64}",
                    }
                )

    # Send to OpenAI
    response = client.responses.create(
        model=MULTI_MODAL_MODEL,
        input=messages,
    )

    return response.output_text


def load_question_embeddings(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def retrieve_chunks_per_question_embedding(
    question_embeddings: dict, top_k: int, pdf_name: str
):
    relevant_chunks_per_question = {}
    metadata_per_question = {}

    for q_id, q_emb in question_embeddings.items():
        result = collection.query(
            query_embeddings=[q_emb], n_results=top_k, where={"pdf_name": pdf_name}
        )

        relevant_chunks_per_question[q_id] = result["documents"][0]
        metadata_per_question[q_id] = {
            "pdf_name": result["metadatas"][0][0]["pdf_name"],
            "image_dir": result["metadatas"][0][0]["image_dir"],
        }

    return relevant_chunks_per_question, metadata_per_question


def ask_questions_from_embeddings(pdf_file, top_k=3):
    question_embeddings = load_question_embeddings(question_embedding_file)
    question_texts = get_questions(QUESTION_ID)

    results = []

    pdf_name = os.path.basename(pdf_file).removesuffix(".pdf")
    if USE_CHUNKING:
        retrievals, metadatas = retrieve_chunks_per_question_embedding(
            question_embeddings, top_k, pdf_name
        )
    else:
        retrievals = [PROCESSED_PATH]  # md files

    for q_id, pdf_chunks in retrievals.items():
        q_text = question_texts[q_id]
        context = "\n\n".join([chunk for chunk in pdf_chunks])
        prompt = prompt_template.format(question=q_text, context=context)
        if MULTI_MODAL:
            filenames = re.findall(
                r"!\[\]\(([^)]+\.(?:jpe?g|png|gif))\)", context, re.IGNORECASE
            )
            image_paths = [
                os.path.join(metadatas[q_id]["image_dir"], filename)
                for filename in filenames
                if os.path.exists(os.path.join(metadatas[q_id]["image_dir"], filename))
            ]
        else:
            image_paths = []
        if len(image_paths) > 0:
            answer = generate_answer_with_gpt4(prompt, list(set(image_paths)))
        else:
            answer = generator.run(prompt=prompt)["replies"][0]

        parsed_response = {}
        current_section = None
        for item in answer.split("\n"):
            item = item.strip()
            if item.startswith("ANSWER"):
                current_section = "answer"
                if "YES" in item:
                    parsed_response["answer"] = "YES"
                elif "UNSURE" in item:
                    parsed_response["answer"] = "UNSURE"
                elif "NO" in item:
                    parsed_response["answer"] = "NO"
                else:
                    parsed_response["answer"] = item.replace("ANSWER:", "").strip()
            elif item.startswith("REASONING"):
                current_section = "reasoning"
                parsed_response["reasoning"] = item.replace("REASONING:", "").strip()
            elif item.startswith("EVIDENCE"):
                current_section = "evidence"
                parsed_response["evidence"] = item.replace("EVIDENCE:", "").strip()
            elif current_section:
                # Append to the current section if the item is a continuation of the previous line
                parsed_response[current_section] = (
                    parsed_response.get(current_section, "") + " " + item
                )

        results.append(
            {
                "paper_id": pdf_name,
                "question_id": q_id,
                "question": q_text,
                **parsed_response,
            }
        )

    return results


files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
pdf_files = [f for f in files if not f.endswith(".gitkeep")]
results = []
try:
    for pdf_file in tqdm(pdf_files):
        results.extend(ask_questions_from_embeddings(pdf_file=pdf_file, top_k=TOP_K))
except Exception as e:
    print(f"\n[ERROR] Processing failed at file: {pdf_file}")
    print(f"[DETAILS] {e}")
    traceback.print_exc()
finally:
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
