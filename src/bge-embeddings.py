import os
import dotenv
from pathlib import Path
import pickle
from FlagEmbedding import BGEM3FlagModel
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from grolts_questions import get_questions
from typing import List

dotenv.load_dotenv()

CACHE_DIR = os.environ.get(("CACHE_DIR"))
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP"))
CHROMA_DIR = os.environ.get("CHROMA_DIR")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EXP_ID = int(os.environ.get("EXP_ID"))
NUM_PAPERS = int(os.environ.get("NUM_PAPERS"))
DATA_DIR = os.environ.get("DATA_DIR")
QUESTION_EMBEDDING_DIR = os.environ.get("QUESTION_EMBEDDING_DIR")

chroma_path = CHROMA_DIR + EMBEDDING_MODEL + "-" + str(CHUNK_SIZE)
question_embedding_file = (
    QUESTION_EMBEDDING_DIR + EMBEDDING_MODEL + "_" + str(EXP_ID) + ".pkl"
)
questions = get_questions(EXP_ID)

folder_pickle_files = Path("synergy-dataset", "pickles")
folder_pickle_files.mkdir(parents=True, exist_ok=True)

class BGEModel():
    def __init__(self):
        self.model = BGEM3FlagModel(EMBEDDING_MODEL, devices=["cuda:0"])

    def embed_documents(
        self, texts: List[str], chunk_size: int | None = None
    ) -> List[List[float]]:
        chunk_size_ = chunk_size
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            response = self.model.encode(
                texts[i : i + chunk_size_],
                max_length=8192,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )['dense_vecs']

            if not isinstance(response, dict):
                response = response.dict()
            embeddings.extend(r["embedding"] for r in response["data"])
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

embedding_function = BGEModel()

def save_to_chroma(docs: list[Document]):
    Chroma.from_documents(
        docs,
        embedding=embedding_function,
        persist_directory=chroma_path,
        collection_metadata={"hnsw:space": "cosine"},
    )
    print(f"Saved {len(docs)} chunks to {chroma_path}.")


def embed_questions(questions: dict):
    """Precompute and store embeddings for all questions."""
    question_embeddings = {}
    for q_id, question_text in questions.items():
        # Embed the question once and store the result
        question_embedding = embedding_function.embed_query(question_text)
        question_embeddings[q_id] = question_embedding
        print(
            f"Embedded question '{q_id}': {question_text[:30]}..."
        )  # Print a snippet of the question for debugging
    return question_embeddings


def save_embeddings(embedding_file, embeddings):
    """Save precomputed embeddings to a file."""
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {embedding_file}.")


question_embeddings = embed_questions(questions)
save_embeddings(question_embedding_file, question_embeddings)

for index in range(NUM_PAPERS):
    loader = PyPDFLoader(str(DATA_DIR + str(index) + ".pdf"))
    pages = loader.load()
    splitted_text = split_text(pages)
    save_to_chroma(splitted_text)
