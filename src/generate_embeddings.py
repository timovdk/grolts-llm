import glob
import os
import pickle
import re
from typing import Dict, List

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
SUBFOLDERS = ["ptsd", "achievement", "delinquency", "wellbeing"]
PROCESSED_DATA_PATH = "./processed_pdfs"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"

QUESTION_IDS = [0, 3, 4]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
CHUNK_SIZES = [500, 1000]
OVERLAP = 50
MAX_TABLE_ROWS = 5

# -------------------------
# Set up Embedder
# -------------------------

embedder = SentenceTransformersDocumentEmbedder(
    model=EMBEDDING_MODEL,
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


def clean_document(text: str) -> str:
    """
    Cleans raw markdown text by removing unwanted characters, URLs, and large tables.
    """
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
    )  # Collapse whitespace within lines
    text = preprocess_tables(text, MAX_TABLE_ROWS)  # Flatten large tables
    return text


def load_preprocessed_md(processed_pdf_folder: str, file_name: str) -> Document:
    """
    Loads a preprocessed markdown file corresponding to a PDF and returns a cleaned Document.
    """
    pdf_dir = os.path.join(processed_pdf_folder, file_name)
    md_file = os.path.join(pdf_dir, f"{file_name}.md")

    if not os.path.exists(md_file):
        msg = f"Markdown file not found for {file_name}"
        print(f"[ERROR] {msg}")
        raise FileNotFoundError(msg)

    with open(md_file, "r", encoding="utf-8") as f:
        text = clean_document(f.read())

    metadata = {"pdf_name": file_name, "image_dir": pdf_dir}
    return Document(content=text, meta=metadata)


def store_document_in_chroma(
    doc: Document, collection: chromadb.Collection, splitter: DocumentSplitter
) -> None:
    """
    Splits, embeds, and stores a Document into a ChromaDB collection.
    """
    chunks = splitter.run([doc])["documents"]

    embedded_chunks: List[Document] = []
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


def process_mds(
    pdf_path: str,
    processed_pdf_folder: str,
    collection: chromadb.Collection,
    splitter: DocumentSplitter,
) -> None:
    """
    Loads and embeds all markdown documents corresponding to PDFs in a given folder.
    """
    files = glob.glob(os.path.join(pdf_path, "*.pdf"))
    pdf_files = [f for f in files if not f.endswith(".gitkeep")]

    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file).removesuffix(".pdf")
        print(f"[INFO] Embedding document: {pdf_name}")
        doc = load_preprocessed_md(processed_pdf_folder, pdf_name)
        store_document_in_chroma(doc, collection, splitter)


def embed_questions(questions: Dict[int, str]) -> Dict[int, List[float]]:
    """
    Embeds a dictionary of questions and returns their embeddings.
    """
    question_embeddings: Dict[int, List[float]] = {}
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
            f"[INFO] Embedded question {embedded_question.meta['question_id']}: {embedded_question.content[:30]}..."
        )
    return question_embeddings


def main() -> None:
    for subfolder in SUBFOLDERS:
        for chunk_size in CHUNK_SIZES:
            # Folder and file bookkeeping
            os.makedirs(DOCUMENT_EMBEDDING_PATH, exist_ok=True)
            chromadb_name = (
                f"{EMBEDDING_MODEL.replace('/', '_')}_{subfolder}_{chunk_size}"
            )
            document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{chromadb_name}"

            # Set up ChromaDB client
            chroma_client = chromadb.PersistentClient(path=document_embedding_file)
            collection = chroma_client.get_or_create_collection(chromadb_name)

            splitter = DocumentSplitter(
                split_by="word",
                split_length=chunk_size,
                respect_sentence_boundary=True,
                split_overlap=OVERLAP,
            )
            splitter.warm_up()

            # Process Documents
            print(f"[INFO] Processing subfolder: {subfolder}")
            process_mds(
                f"{DATA_PATH}/{subfolder}",
                f"{PROCESSED_DATA_PATH}/{subfolder}",
                collection,
                splitter,
            )
            print(f"[INFO] Document embeddings stored at: {document_embedding_file}\n")

    for q_id in QUESTION_IDS:
        # Folder and file bookkeeping
        os.makedirs(QUESTION_EMBEDDING_PATH, exist_ok=True)
        question_embedding_file = (
            f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{q_id}.pkl"
        )

        # Process Questions
        embeddings = embed_questions(get_questions(q_id))
        with open(question_embedding_file, "wb") as f:
            pickle.dump(embeddings, f)
        print(f"[INFO] Question embeddings stored at: {question_embedding_file}\n")


if __name__ == "__main__":
    main()
