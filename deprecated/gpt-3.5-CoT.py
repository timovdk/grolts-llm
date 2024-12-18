from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()
from tqdm import tqdm
tqdm.pandas()

import pickle
import os
import grolts_prompts
import grolts_questions

RUN_DOC_EMBEDDINGS = True

#CHUNK_SIZE = 250
#CHUNK_OVERLAP = 75
#RELEVANT_CHUNKS = 3

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
RELEVANT_CHUNKS = 3

#CHUNK_SIZE = 750
#CHUNK_OVERLAP = 150
#RELEVANT_CHUNKS = 3

#CHUNK_SIZE = 1000
#CHUNK_OVERLAP = 200
#RELEVANT_CHUNKS = 3

GENERATION_MODEL = 'gpt-3.5-turbo'
EMBEDDING_MODEL = 'text-embedding-3-large'
embedding_function = OpenAIEmbeddings(model=EMBEDDING_MODEL)

NUM_PAPERS = 38
PROMPT_ID = 0
EXP_ID = 1

DATA_PATH = './data/'
CHROMA_PATH = './chroma/chroma_' + EMBEDDING_MODEL + '-' + str(CHUNK_SIZE)

if PROMPT_ID == 0:
    PROMPT_TEMPLATE = grolts_prompts.SHORT_YN_INFERRING
elif PROMPT_ID == 1:
    PROMPT_TEMPLATE = grolts_prompts.LONG_YN_INFERRING
elif PROMPT_ID == 2:
    PROMPT_TEMPLATE = grolts_prompts.LONG_YN_NO_INFERRING
elif PROMPT_ID == 3:
    PROMPT_TEMPLATE = grolts_prompts.LONG_YNU_INFERRING
elif PROMPT_ID == 4:
    PROMPT_TEMPLATE = grolts_prompts.LONG_YNU_NO_INFERRING
else:
    print("ERROR: No prompt defined")
    exit(1)

if EXP_ID == 0:
    questions = grolts_questions.p1
elif EXP_ID == 1:
    questions = grolts_questions.p2
elif EXP_ID == 2:
    questions = grolts_questions.p3
else:
    print("ERROR: No questions defined")
    exit(1)

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
    db = Chroma.from_documents(
        docs, embedding=embedding_function, persist_directory=CHROMA_PATH, collection_metadata={"hnsw:space": "cosine"}
    )
    print(f"Saved {len(docs)} chunks to {CHROMA_PATH}.")

def embed_questions(questions: dict):
    """Precompute and store embeddings for all questions."""
    question_embeddings = {}
    for q_id, question_text in questions.items():
        # Embed the question once and store the result
        question_embedding = embedding_function.embed_query(question_text)
        question_embeddings[q_id] = question_embedding
        print(f"Embedded question '{q_id}': {question_text[:30]}...")  # Print a snippet of the question for debugging
    return question_embeddings

def load_embeddings(embedding_file):
    """Load precomputed embeddings from a file."""
    if os.path.exists(embedding_file):
        with open(embedding_file, 'rb') as f:
            question_embeddings = pickle.load(f)
        print("Loaded precomputed question embeddings.")
        return question_embeddings
    else:
        print(f"No precomputed embeddings found at {embedding_file}.")
        return None

def save_embeddings(embedding_file, embeddings):
    """Save precomputed embeddings to a file."""
    with open(embedding_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"Saved embeddings to {embedding_file}.")

def generate_output(paper_id, question_id, question_embedding):      
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function, collection_metadata={"hnsw:space": "cosine"})

    # Search the DB.
    results = db.similarity_search_by_vector(question_embedding, k=RELEVANT_CHUNKS, filter={'source': str('./data/' + str(paper_id) + '.pdf')})
    if len(results) == 0:
        print(f"Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=questions[question_id])

    model = ChatOpenAI(model=GENERATION_MODEL)
    response_text = model.invoke(prompt)
    #for item in response_text.content.split("\n"):
    #    if "ANSWER" in item:
    #        if "YES" in item:
    #            return "YES"
    #        #elif "UNKNOWN" in item:
    #        #    return "UNKNOWN"
    #        elif "NO" in item:
    #            return "NO"

    return response_text.content

# Path to store/load the precomputed embeddings
embedding_file = './question_embeddings_new/' + EMBEDDING_MODEL + '_' + str(EXP_ID) + '.pkl'

# Try to load existing embeddings
question_embeddings = load_embeddings(embedding_file)

# If no embeddings exist, compute and save them
if question_embeddings is None:
    question_embeddings = embed_questions(questions)
    save_embeddings(embedding_file, question_embeddings)

# Create DataFrame
df = pd.DataFrame(columns=list(questions.keys()))
df['Paper'] = range(NUM_PAPERS)

docs = []
ids = []
# Fill other columns with empty values
for index in range(NUM_PAPERS):
    df.loc[index, questions.keys()] = ''

if RUN_DOC_EMBEDDINGS:
    for index in range(NUM_PAPERS):
        loader = PyPDFLoader(str(DATA_PATH + str(index) + ".pdf"))
        pages = loader.load()
        splitted_text = split_text(pages)
        save_to_chroma(splitted_text)

def fill_cells(row):
    paper_id = row['Paper']
    for q_id, _ in row.items():
        if q_id != 'Paper':
            row[q_id] = generate_output(paper_id, q_id, question_embeddings[q_id])
    return row


# Apply the function to each cell in the DataFrame
df = df.progress_apply(fill_cells, axis=1)
df.to_csv('./data_out/' + str(CHUNK_SIZE) + '-' + GENERATION_MODEL + '-' + EMBEDDING_MODEL + '-p' + str(EXP_ID+1) + '.csv', index=False)
