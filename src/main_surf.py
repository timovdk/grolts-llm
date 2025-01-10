import os
import pickle

import dotenv
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from tqdm import tqdm

from grolts_prompts import get_prompt_template
from grolts_questions import get_questions

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer  # , BitsAndBytesConfig
# import torch

# cache_dir = "/projects/2/managed_datasets/hf_cache_dir"

dotenv.load_dotenv()
tqdm.pandas()

# Batch size for processing, 3 batches of 7 (21 questions)
BATCH_SIZE = 7

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP"))
RELEVANT_CHUNKS = int(os.environ.get("RELEVANT_CHUNKS"))
CHROMA_DIR = os.environ.get("CHROMA_DIR")
EXP_ID = int(os.environ.get("EXP_ID"))
NUM_PAPERS = int(os.environ.get("NUM_PAPERS"))
DATA_DIR = os.environ.get("DATA_DIR")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL")
PROMPT_ID = int(os.environ.get("PROMPT_ID"))
QUESTION_EMBEDDING_DIR = os.environ.get("QUESTION_EMBEDDING_DIR")
OUT_DIR = os.environ.get("OUT_DIR")

chroma_path = CHROMA_DIR + EMBEDDING_MODEL + "-" + str(CHUNK_SIZE)
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)
question_embedding_file = (
    QUESTION_EMBEDDING_DIR + EMBEDDING_MODEL + "_" + str(EXP_ID) + ".pkl"
)
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
tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)  # , cache_dir=cache_dir)

pipeline = transformers.pipeline("text-generation", model=model, tokenizer=tokenizer)


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


def load_embeddings(embedding_file):
    """Load precomputed embeddings from a file."""
    with open(embedding_file, "rb") as f:
        question_embeddings = pickle.load(f)
    print("Loaded precomputed question embeddings.")
    return question_embeddings


def generate_outputs(db, paper_id, question_ids, question_embeddings):
    """
    Generate responses for multiple questions for a given paper.
    """
    # Generate prompts for each question for paper_id
    results = []
    for question_id in question_ids:
        question_embedding = question_embeddings[question_id]
        search_results = db.similarity_search_by_vector(
            question_embedding,
            k=RELEVANT_CHUNKS,
            filter={"source": str("./data/" + str(paper_id) + ".pdf")},
        )
        if not search_results:
            print(
                f"No matching results found for paper {paper_id}, question {question_id}"
            )
            continue

        context_text = "\n\n---\n\n".join([doc.page_content for doc in search_results])
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

    # Use batch processing with the pipeline
    if results:
        messages_batch = [r[1] for r in results]
        outputs = pipeline(messages_batch, max_new_tokens=2048)

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
    return []


# Run question embeddings if they do not exist yet
if not os.path.exists(question_embedding_file):
    question_embeddings = embed_questions(questions)
    save_embeddings(question_embedding_file, question_embeddings)

# Run chunk embeddings if they do not exist yet
if not os.path.exists(chroma_path):
    for index in range(NUM_PAPERS):
        loader = PyPDFLoader(str(DATA_DIR + str(index) + ".pdf"))
        pages = loader.load()
        splitted_text = split_text(pages)
        save_to_chroma(splitted_text)
else:
    print("Using precomputed document embeddings.")

# Try to load existing embeddings
question_embeddings = load_embeddings(question_embedding_file)

output_file = os.path.join(
    OUT_DIR,
    f"{CHUNK_SIZE}-{GENERATION_MODEL.replace('/', '-')}-{EMBEDDING_MODEL}-p{EXP_ID + 1}.csv",
)

pd.DataFrame(
    columns=["paper_id", "question_id", "reasoning", "evidence", "answer"]
).to_csv(output_file, index=False)

db = Chroma(
    persist_directory=chroma_path,
    embedding_function=embedding_function,
    collection_metadata={"hnsw:space": "cosine"},
)

# Loop through each paper
for paper_id in tqdm(range(NUM_PAPERS), desc="Processing Papers"):
    paper_results = []

    # Process questions in batches
    question_ids = list(questions.keys())
    for i in range(0, len(question_ids), BATCH_SIZE):
        batch_question_ids = question_ids[i : i + BATCH_SIZE]
        outputs = generate_outputs(
            db, paper_id, batch_question_ids, question_embeddings
        )
        paper_results.extend(outputs)

    # Convert the results to a DataFrame
    if paper_results:
        paper_df = pd.DataFrame(paper_results)

        # Append the DataFrame to the CSV file
        paper_df.to_csv(output_file, mode="a", header=False, index=False)
