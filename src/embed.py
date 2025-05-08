import glob
import os
import pickle
import subprocess

import chromadb
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.dataclasses import Document
from haystack.utils import Secret

from grolts_questions import get_questions

API_KEY = ""

DATA_PATH = "./data"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"
PROCESSED_PATH = "./processed_pdfs"

QUESTION_ID = 3
EMBEDDING_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 512
OVERLAP = 50
FORCE_NEW_EMBEDDINGS = True

os.makedirs(PROCESSED_PATH, exist_ok=True)
os.makedirs(DOCUMENT_EMBEDDING_PATH, exist_ok=True)
os.makedirs(QUESTION_EMBEDDING_PATH, exist_ok=True)

document_collection_name = f"{EMBEDDING_MODEL}_{CHUNK_SIZE}"
question_embedding_file = f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL}.pkl"
document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{document_collection_name}"

chroma_client = chromadb.PersistentClient(path=document_embedding_file)
collection = chroma_client.get_or_create_collection(document_collection_name)

splitter = DocumentSplitter(
    split_by="word",
    split_length=CHUNK_SIZE,
    respect_sentence_boundary=True,
    split_overlap=OVERLAP,
)
splitter.warm_up()
#embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL_PATH+EMBEDDING_MODEL)
embedder = OpenAIDocumentEmbedder(api_key=Secret.from_token(API_KEY), model=EMBEDDING_MODEL)
#embedder.warm_up()

def get_embedding(text: str):
    return embedder.embed(text)


def load_preprocessed_pdf(pdf_name: str):
    pdf_dir = os.path.join(PROCESSED_PATH, pdf_name)
    md_file = os.path.join(pdf_dir, f"{pdf_name}.md")

    if not os.path.exists(md_file):
        raise FileNotFoundError(f"Markdown file not found for {pdf_name}")

    with open(md_file, "r", encoding="utf-8") as f:
        text = f.read()

    metadata = {"pdf_name": pdf_name, "image_dir": pdf_dir}

    return text, metadata



def store_document_in_chroma(text: str, metadata: dict):
    chunks = splitter.run([Document(content=text, meta=metadata)])["documents"]
    docs_to_embed = [Document(content=c.content, meta=c.meta) for c in chunks]

    embedded_chunks = embedder.run(docs_to_embed)["documents"]

    for idx, embedded_chunk in enumerate(embedded_chunks):
        embedded_chunk.meta.pop("_split_overlap", None)
        collection.add(
            documents=[embedded_chunk.content],
            embeddings=[embedded_chunk.embedding],
            metadatas=[embedded_chunk.meta],
            ids=[f"{embedded_chunk.meta["pdf_name"]}_chunk_{idx}"],
        )


def check_extracted_files(pdf_name):
    md_file_path = os.path.join(PROCESSED_PATH, pdf_name, f"{pdf_name}.md")
    return os.path.exists(md_file_path)


def pre_process_pdfs(pdf_path: str):
    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))
    if any(
        [
            not check_extracted_files(os.path.basename(f).strip(".pdf"))
            for f in pdf_files
        ]
    ):
        subprocess.run(
            ["marker", pdf_path, "--output_dir", f"{PROCESSED_PATH}", "--workers", "2"],
            check=True,
        )


def process_pdfs(pdf_path):
    pdf_files = glob.glob(os.path.join(pdf_path, "*.pdf"))

    for pdf_file in pdf_files:
        pdf_name = os.path.basename(pdf_file)
        print(f"Embedding PDF: {pdf_name}")

        text, metadata = load_preprocessed_pdf(pdf_name.strip(".pdf"))
        store_document_in_chroma(text, metadata)


def embed_questions(questions: dict):
    question_embeddings = {}
    questions_to_embed = [Document(content=question_text, meta={"question_id": q_id}) for q_id, question_text in questions.items()]
    embeded_questions = embedder.run(questions_to_embed)["documents"]
    for embedded_question in embeded_questions:
        question_embeddings[embedded_question.meta["question_id"]] = embedded_question.embedding
        print(f"Embedded question '{embedded_question.meta["question_id"]}': {embedded_question.content[:30]}...")
    return question_embeddings


def process_questions(questions: dict, path: str):
    with open(path, "wb") as f:
        pickle.dump(embed_questions(questions), f)


def main():
    pre_process_pdfs(DATA_PATH)

    if FORCE_NEW_EMBEDDINGS:
        process_pdfs(DATA_PATH)
        process_questions(get_questions(QUESTION_ID), question_embedding_file)

    else:
        if collection.count() == 0:
            process_pdfs(DATA_PATH)
        if not os.path.exists(question_embedding_file):
            process_questions(get_questions(QUESTION_ID), question_embedding_file)

    print(
        f"Document embeddings ready at {document_embedding_file}\nQuestion embeddings ready at {question_embedding_file}"
    )


if __name__ == "__main__":
    main()
