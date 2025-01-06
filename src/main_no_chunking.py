import os

import dotenv
import pandas as pd
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from tqdm import tqdm
from markitdown import MarkItDown


from grolts_prompts import get_prompt_template
from grolts_questions import get_questions

dotenv.load_dotenv()
tqdm.pandas()

EXP_ID = int(os.environ.get("EXP_ID"))
NUM_PAPERS = int(os.environ.get("NUM_PAPERS"))
DATA_DIR = os.environ.get("DATA_DIR")
GENERATION_MODEL = os.environ.get("GENERATION_MODEL")
PROMPT_ID = int(os.environ.get("PROMPT_ID"))
OUT_DIR = os.environ.get("OUT_DIR")

prompt_template = get_prompt_template(PROMPT_ID)
questions = get_questions(EXP_ID)


def generate_output(paper_id, question_id):
    md = MarkItDown()
    context_text = md.convert("./data/" + str(paper_id) + ".pdf")

    prompt = ChatPromptTemplate.from_template(prompt_template)
    prompt_formatted = prompt.format(
        context=context_text.text_content, question=questions[question_id]
    )

    if "gpt" in GENERATION_MODEL:
        model = ChatOpenAI(model=GENERATION_MODEL)
    else:
        model = ChatGroq(temperature=0, model_name=GENERATION_MODEL)

    response_text = model.invoke(prompt_formatted)

    response = {"paper_id": paper_id, "question_id": question_id}
    current_section = None

    for item in response_text.content.split("\n"):
        item = item.strip()
        if item.startswith("ANSWER"):
            current_section = "answer"
            if "YES" in item:
                response["answer"] = "YES"
            elif "UNSURE" in item:
                response["answer"] = "UNSURE"
            elif "NO" in item:
                response["answer"] = "NO"
            else:
                response["answer"] = item.replace("ANSWER:", "").strip()

        elif item.startswith("REASONING"):
            current_section = "reasoning"
            response["reasoning"] = item.replace("REASONING:", "").strip()

        elif item.startswith("EVIDENCE"):
            current_section = "evidence"
            response["evidence"] = item.replace("EVIDENCE:", "").strip()

        elif current_section:
            # Append to the current section if the item is a continuation of the previous line
            response[current_section] = response.get(current_section, "") + " " + item

    return response


output_file = os.path.join(OUT_DIR, f"{GENERATION_MODEL}-p{EXP_ID + 1}.csv")

pd.DataFrame(
    columns=["paper_id", "question_id", "reasoning", "evidence", "answer"]
).to_csv(output_file, index=False)

# Loop through each paper and each question
for paper_id in tqdm(range(NUM_PAPERS), desc="Processing Papers"):
    paper_results = []  # Collect results for the current paper
    for question_id in questions.keys():
        output = generate_output(paper_id, question_id)
        paper_results.append(output)

    # Convert the results to a DataFrame
    paper_df = pd.DataFrame(paper_results)

    # Append the DataFrame to the CSV file
    paper_df.to_csv(output_file, mode="a", header=False, index=False)
