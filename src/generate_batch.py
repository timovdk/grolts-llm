import base64
import glob
import json
import os
import pickle
import re

import chromadb
from tqdm import tqdm

from grolts_questions import get_questions

SYSTEM_PROMPT = """You are an assistant evaluating the quality of academic papers. Answer the QUESTION using only the given CONTEXT. Follow the format below exactly. Do not write anything before or after.

REASONING: Step-by-step explanation based only on the context. Assume all references to supplementary material or URLs contain the information the authors refer to. Conclude with a YES or NO.

EVIDENCE: List direct quotes from the context that support the reasoning. Each quote must be on a new line with a dash. If no evidence is found, write [].

ANSWER: Write only YES or NO.
"""

USER_PROMPT = """
QUESTION: {question}

CONTEXT: {context}
"""

DATA_PATH = "./data"
PROCESSED_DATA_PATH = "./processed_pdfs"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"
PROCESSED_PATH = "./processed_pdfs"
OUTPUT_PATH = "./batches"

QUESTION_ID = 0
TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-large"
# EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
GENERATOR_MODEL = "gpt-4o-mini"
MULTI_MODAL_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 500
MAX_TABLE_ROWS = 5
MAX_TABLE_COLS = 5


USE_CHUNKING = True
MULTI_MODAL = True

os.makedirs(OUTPUT_PATH, exist_ok=True)

document_collection_name = f"{EMBEDDING_MODEL.replace('/', '_')}_{CHUNK_SIZE}"
question_embedding_file = (
    f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{QUESTION_ID}.pkl"
)
document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{document_collection_name}"
output_file = f"{OUTPUT_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{GENERATOR_MODEL}_{CHUNK_SIZE if USE_CHUNKING else 'NO_CHUNKING'}_{str(MULTI_MODAL)}.jsonl"

chroma_client = chromadb.PersistentClient(path=document_embedding_file)
collection = chroma_client.get_or_create_collection(document_collection_name)

batch = []


def load_question_embeddings(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_preprocessed_md(file_name: str):
    pdf_dir = os.path.join(PROCESSED_DATA_PATH, file_name)
    md_file = os.path.join(pdf_dir, f"{file_name}.md")

    if not os.path.exists(md_file):
        raise FileNotFoundError(f"Markdown file not found for {file_name}")

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    return text


def replace_urls(text):
    return re.sub(r"\b(?:https?://|www\.)\S+\b", "URL", text)


def preprocess_tables(text: str, max_rows: int = MAX_TABLE_ROWS) -> str:
    """
    Detects markdown tables and flattens them to plain text if they have more than `max_rows`.
    """

    def is_table_line(line):
        return bool(re.match(r"^\s*\|.*\|\s*$", line))

    lines = text.splitlines()
    output = []
    table = []
    in_table = False

    for line in lines + [""]:  # Add empty line to flush at end
        if is_table_line(line):
            table.append(line)
            in_table = True
        else:
            if in_table:
                # Process the collected table
                if len(table) >= 3:  # at least header, divider, and one row
                    header = [h.strip() for h in table[0].strip().strip("|").split("|")]
                    rows = table[2:]  # Skip header and divider
                    if len(rows) > max_rows:
                        for row in rows:
                            values = [
                                v.strip() for v in row.strip().strip("|").split("|")
                            ]
                            flattened = ", ".join(
                                f"{h}: {v}"
                                for h, v in zip(header, values)
                                if v and v.strip()
                            )
                            flattened = re.sub(r"^:\s*", "", flattened)
                            if flattened.strip():
                                output.append(flattened + ".")
                    else:
                        output.extend(table)
                else:
                    output.extend(table)
                table = []
                in_table = False
            output.append(line)

    return "\n".join(output)


def clean_document(text):
    text = re.sub(r"�+", "", text)  # Remove replacement characters
    text = re.sub(r"\(<br\s*/?>\)", "", text)  # Remove (<br>)
    text = re.sub(r"<br\s*/?>", " ", text)  # Replace <br> with space
    text = re.sub(r"\b(?:https?://|www\.)\S+\b", "URL", text)  # Plain URLs
    text = re.sub(
        r"\[(.*?)\]\((?:https?://|www\.)\S+\)", r"[\1](URL)", text
    )  # Markdown links
    text = re.sub(
        r"^\s*:\s*$", "", text, flags=re.MULTILINE
    )  # Remove standalone colons
    text = "\n".join(
        re.sub(r"\s+", " ", line).strip() for line in text.splitlines()
    )  # Collapse whitespcae within lines
    text = preprocess_tables(text, MAX_TABLE_ROWS)  # Finally, flatten large tables
    return text


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

    pdf_name = os.path.basename(pdf_file).removesuffix(".pdf")
    if USE_CHUNKING:
        retrievals, metadatas = retrieve_chunks_per_question_embedding(
            question_embeddings, top_k, pdf_name
        )
    else:
        md_text = load_preprocessed_md(pdf_name)
        md_text = clean_document(md_text)

        # Create a dict with question IDs as keys and the same md_text for each
        retrievals = {q_id: [md_text] for q_id in question_texts.keys()}

    for q_id, pdf_chunks in retrievals.items():
        q_text = question_texts[q_id]
        context = "\n\n".join([chunk for chunk in pdf_chunks])
        prompt = USER_PROMPT.format(question=q_text, context=context)
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
            image_paths = None

        add_to_batch(f"{pdf_name}_{q_id}", SYSTEM_PROMPT, prompt, image_paths)


def add_to_batch(
    custom_id: str, system_prompt: str, user_prompt: str, image_paths: list = None
):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    if image_paths:
        for path in image_paths:
            with open(path, "rb") as img_file:
                img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
                messages[1]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    }
                )

    batch_line = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MULTI_MODAL_MODEL,
            "messages": messages,
        },
    }

    batch.append(batch_line)


def write_batch_to_file(output_file: str = output_file):
    with open(output_file, "a") as f:
        for line in batch:
            json.dump(line, f)
            f.write("\n")


files = glob.glob(os.path.join(DATA_PATH, "*.pdf"))
pdf_files = [f for f in files if not f.endswith(".gitkeep")]

for pdf_file in tqdm(pdf_files):
    ask_questions_from_embeddings(pdf_file=pdf_file, top_k=TOP_K)

write_batch_to_file()
