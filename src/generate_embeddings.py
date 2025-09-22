import glob
import os
import pickle
import re

import chromadb
import torch
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document

from grolts_questions import get_questions

# -------------------------
# Config
# -------------------------
DATA_PATH = "./data"
PROCESSED_DATA_PATH = "./processed_pdfs"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"
PROCESSED_PATH = "./processed_pdfs"

QUESTION_ID = 3
EMBEDDING_MODEL = "Qwen3-Embedding-8B"
CHUNK_SIZE = 500
OVERLAP = 50
FORCE_NEW_EMBEDDINGS = False
MAX_TABLE_ROWS = 5

# -------------------------
# Folder and file bookkeeping
# -------------------------
os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(DOCUMENT_EMBEDDING_PATH, exist_ok=True)
os.makedirs(QUESTION_EMBEDDING_PATH, exist_ok=True)

document_collection_name = f"{EMBEDDING_MODEL.replace('/', '_')}_{CHUNK_SIZE}"
question_embedding_file = (
    f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{QUESTION_ID}.pkl"
)
document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{document_collection_name}"

# -------------------------
# Set up ChromaDB client
# -------------------------
chroma_client = chromadb.PersistentClient(path=document_embedding_file)
collection = chroma_client.get_or_create_collection(document_collection_name)

# -------------------------
# Set up Splitter and Embedder
# -------------------------
splitter = DocumentSplitter(
    split_by="word",
    split_length=CHUNK_SIZE,
    respect_sentence_boundary=True,
    split_overlap=OVERLAP,
)
splitter.warm_up()

embedder = SentenceTransformersDocumentEmbedder(
    model="Qwen/Qwen3-Embedding-8B",
    model_kwargs={
        "attn_implementation": "flash_attention_2",
        "device_map": "auto",
        "dtype": torch.float16,
    },
    tokenizer_kwargs={"padding_side": "left"},
)
embedder.warm_up()


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


def load_preprocessed_md(file_name: str):
    pdf_dir = os.path.join(PROCESSED_DATA_PATH, file_name)
    md_file = os.path.join(pdf_dir, f"{file_name}.md")

    if not os.path.exists(md_file):
        raise FileNotFoundError(f"Markdown file not found for {file_name}")

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()
        text = clean_document(text)

    metadata = {"pdf_name": file_name, "image_dir": pdf_dir}

    return Document(content=text, meta=metadata)


def store_document_in_chroma(doc: Document):
    chunks = splitter.run([doc])["documents"]

    embedded_chunks = []
    for i in range(0, len(chunks), 8):
        batch = chunks[i : i + 8]
        embedded_chunks.extend(embedder.run(batch)["documents"])

    for idx, embedded_chunk in enumerate(embedded_chunks):
        embedded_chunk.meta.pop("_split_overlap", None)
        collection.add(
            documents=[embedded_chunk.content],
            embeddings=[embedded_chunk.embedding],
            metadatas=[embedded_chunk.meta],
            ids=[f"{embedded_chunk.meta['pdf_name']}_chunk_{idx}"],
        )


def process_mds(pdf_path):
    files = glob.glob(os.path.join(pdf_path, "*.pdf"))
    pdf_files = [f for f in files if not f.endswith(".gitkeep")]

    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file)
        print(f"Embedding: {pdf_name.removesuffix('.pdf')}")
        doc = load_preprocessed_md(pdf_name.removesuffix(".pdf"))
        store_document_in_chroma(doc)


def embed_questions(questions: dict):
    question_embeddings = {}
    questions_to_embed = [
        Document(
            content=f"Instruct: Given a query about reporting practices in latent trajectory studies, retrieve passages from the paper that answer the query\nQuery:{question_text}",
            meta={"question_id": q_id},
        )
        for q_id, question_text in questions.items()
    ]
    embeded_questions = embedder.run(questions_to_embed)["documents"]
    for embedded_question in embeded_questions:
        question_embeddings[embedded_question.meta["question_id"]] = (
            embedded_question.embedding
        )
        print(
            f"Embedded question '{embedded_question.meta['question_id']}': {embedded_question.content[:30]}..."
        )
    return question_embeddings


def process_questions(questions: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(embed_questions(questions), f)


def main():
    process_mds(DATA_PATH)
    process_questions(get_questions(QUESTION_ID), question_embedding_file)

    print(
        f"Document embeddings ready at {document_embedding_file}\nQuestion embeddings ready at {question_embedding_file}"
    )


if __name__ == "__main__":
    main()
