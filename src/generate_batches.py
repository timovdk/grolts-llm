import glob
import json
import os
import pickle
from typing import Dict, List

import chromadb
from tqdm import tqdm

from grolts_questions import get_questions

# -------------------------
# Configuration
# -------------------------
SYSTEM_PROMPT = """You are an academic expert on latent trajectory studies evaluating the quality of academic papers. Answer the QUESTION using only the given CONTEXT, which is markdown-formatted text from a single academic paper. Follow the format below exactly. Do not write anything before or after.
REASONING: Step-by-step explanation based only on the CONTEXT. Interpret markdown formatting as needed. Assume that any reference to supplementary materials or external URLs (e.g., OSF, GitHub) is accurate and complete. If the CONTEXT states that a dataset, figure, or detail exists in such a source or the paper itself, you may treat it as if it is available and correct. Conclude with a YES or NO.
EVIDENCE: List direct quotes from the CONTEXT that support the reasoning. Each quote must be on a new line with a dash. If no direct quotes are found but the reasoning is strongly supported by implied content in the CONTEXT, you may include indirect evidence, but only if it is clearly and unambiguously implied. If no such evidence exists, write nothing. Still provide REASONING and ANSWER.
ANSWER: Write only YES or NO.
"""

USER_PROMPT = """
QUESTION: {question}
CONTEXT: {context}
"""

DATA_PATH = "./data"
SUBFOLDERS = ["ptsd", "achievement", "delinquency", "wellbeing"]
PROCESSED_DATA_PATH = "./processed_pdfs"
DOCUMENT_EMBEDDING_PATH = "./document_embeddings"
QUESTION_EMBEDDING_PATH = "./question_embeddings"
OUTPUT_PATH = "./batches"

QUESTION_IDS = [0, 3]
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
GENERATOR_MODELS = ["generic", "gpt-5-mini", "gpt-5"] # Generic is for local LLMs
CHUNK_SIZES = [500, 1000]
TOP_K = 10


def retrieve_chunks_per_question_embedding(
    question_embeddings: Dict[int, List[float]],
    pdf_name: str,
    collection: chromadb.Collection,
) -> Dict[int, List[str]]:
    """
    Retrieve the top-k most relevant chunks for each question embedding from Chroma.
    """
    relevant_chunks_per_question: Dict[int, List[str]] = {}

    for q_id, q_emb in question_embeddings.items():
        result = collection.query(
            query_embeddings=[q_emb], n_results=TOP_K, where={"pdf_name": pdf_name}
        )

        relevant_chunks_per_question[q_id] = result["documents"][0]

    return relevant_chunks_per_question


def ask_questions_from_embeddings(
    pdf_file: str,
    question_embeddings: Dict[int, List[float]],
    question_texts: Dict[int, str],
    collection: chromadb.Collection,
) -> Dict[str, List[Dict]]:
    """
    Construct prompts and batch lines for a single PDF given its embeddings.
    """
    pdf_name = os.path.basename(pdf_file).removesuffix(".pdf")
    retrievals = retrieve_chunks_per_question_embedding(
        question_embeddings, pdf_name, collection
    )

    model_batches: Dict[str, List[Dict]] = {model: [] for model in GENERATOR_MODELS}

    for q_id, pdf_chunks in retrievals.items():
        q_text = question_texts[q_id]
        context = "\n\n".join(pdf_chunks)
        prompt = USER_PROMPT.format(question=q_text, context=context)
        for generator_model in GENERATOR_MODELS:
            model_batches[generator_model].append(
                generate_batch_line(
                    f"{pdf_name}_{q_id}", SYSTEM_PROMPT, prompt, generator_model
                )
            )

    return model_batches


def generate_batch_line(
    custom_id: str, system_prompt: str, user_prompt: str, generator_model: str
) -> Dict:
    """
    Format a single API request line for batch inference.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
    ]

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": generator_model,
            "messages": messages,
        },
    }


def main() -> None:
    """
    For each subfolder and question set, retrieve relevant chunks and
    prepare batch JSONL files for generation.
    """
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    for subfolder in SUBFOLDERS:
        for chunk_size in CHUNK_SIZES:
            # Set up ChromaDB client
            chromadb_name = (
                f"{EMBEDDING_MODEL.replace('/', '_')}_{subfolder}_{chunk_size}"
            )
            document_embedding_file = f"{DOCUMENT_EMBEDDING_PATH}/{chromadb_name}"

            chroma_client = chromadb.PersistentClient(path=document_embedding_file)
            collection = chroma_client.get_or_create_collection(chromadb_name)

            for q_id in QUESTION_IDS:
                # Load question embeddings
                question_embedding_file = f"{QUESTION_EMBEDDING_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{q_id}.pkl"
                if not os.path.exists(question_embedding_file):
                    msg = (
                        f"Question embedding file not found: {question_embedding_file}"
                    )
                    print(f"[ERROR] {msg}")
                    raise FileNotFoundError(msg)

                with open(question_embedding_file, "rb") as f:
                    question_embeddings = pickle.load(f)

                question_texts = get_questions(q_id)

                files = glob.glob(os.path.join(f"{DATA_PATH}/{subfolder}", "*.pdf"))
                pdf_files = [f for f in files if not f.endswith(".gitkeep")]

                if not pdf_files:
                    print(f"[WARN] No PDF files found in {subfolder}")
                    continue

                output_files = {}
                for generator_model in GENERATOR_MODELS:
                    output_file_path = f"{OUTPUT_PATH}/{EMBEDDING_MODEL.replace('/', '_')}_{generator_model}_{subfolder}_{chunk_size}_{q_id}.jsonl"
                    print(f"[INFO] Writing output batch to {output_file_path}")
                    output_files[generator_model] = open(
                        output_file_path, "w", encoding="utf-8"
                    )

                try:
                    for pdf_file in tqdm(pdf_files, desc=f"Processing {subfolder}"):
                        # Generate batch lines for all models in one pass
                        model_batches = ask_questions_from_embeddings(
                            pdf_file, question_embeddings, question_texts, collection
                        )

                        for model, lines in model_batches.items():
                            for line in lines:
                                output_files[model].write(json.dumps(line) + "\n")
                finally:
                    # Close all file handles
                    for f in output_files.values():
                        f.close()

    print("[INFO] All batches completed successfully.")


if __name__ == "__main__":
    main()
