from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
import shutil
import os
import pandas as pd
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import dotenv
dotenv.load_dotenv()

CHROMA_PATH = "chroma"
PROMPT_TEMPLATE = """
You are a helpful assistant assessing the quality of academic papers. Answer the question with YES or NO. Write nothing else before or after.
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""



def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    #document = chunks[10]
    #print(document.page_content)
    #print(document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    #for reference: This only works with chroma version 0.4.14. here is the issue page: https://github.com/langchain-ai/langchain/issues/14872

    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def generate_output(paperpath, check):
    loader = PyPDFLoader(paperpath)
    pages = loader.load()
    splitted_test = split_text(pages)
    save_to_chroma(splitted_test)
    
    embedding_function = OpenAIEmbeddings(disallowed_special=())
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(check, k=3)
    if len(results) == 0 or results[0][1] < 0.6:
        print(f"Unable to find matching results.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=check)
    #print(prompt)

    model = ChatOpenAI()
    response_text = model.predict(prompt)
    return response_text



# List of questions
questions = [
    "Is the metric of time used in the statistical model reported?",
    "Is information presented about the mean and variance of time within a wave?",
    "Is the missing data mechanism reported?",
    "Is a description provided of what variables are related to attrition/missing data?",
    "Is a description provided of how missing data in the analyses were dealt with?",
    "Is information about the distribution of the observed variables included?",
    "Is the software mentioned?",
    "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    "Are alternative specifications of the between-class differences in variancecovariance matrix structure considered and clearly documented?",
    "Are alternative shape/functional forms of the trajectories described?",
    "If covariates have been used, can analyses still be replicated?",
    "Is information reported about the number of random start values and final iterations included?",
    "Are the model comparison (and selection) tools described from a statistical perspective?",
    "Are the total number of fitted models reported, including a one-class solution?",
    "Are the number of cases per class reported for each model (absolute sample size, or proportion)?",
    "If classification of cases in a trajectory is the goal, is entropy reported?",
    #"Is a plot included with the estimated mean trajectories of the final solution?",
    #"Are plots included with the estimated mean trajectories for each model?",
    #"Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    #"Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?"
]

questionsp2 = [
    "Is the metric or unit of time used in the statistical model reported?",
    "Is information presented about the mean and variance of time within a wave?",
    "Is the missing data mechanism reported?",
    "Is a description provided of what variables are related to attrition/missing data?",
    "Is a description provided of how missing data in the analyses were dealt with?",
    "Is information about the distribution of the observed variables included?",
    "Is the software that was used for the statistical analysis mentioned?",
    "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    "Are alternative specifications of the between-class differences in variancecovariance matrix structure considered and clearly documented?",
    "Are alternative shape/functional forms of the trajectories described?",
    "If covariates or predictors have been used, is it done in such a way that the analyses could be replicated?",
    "Is information reported about the number of random start values and final iterations included?",
    "Are the model comparison (and selection) tools described from a statistical perspective?",
    "Are the total number of fitted models reported, including a one-class solution?",
    "Are the number of cases per class reported for each model (absolute sample size, or proportion)?",
    "If classification of cases in a trajectory is the goal, is entropy reported?",
    #"Is a plot included with the estimated mean trajectories of the final solution?",
    #"Are plots included with the estimated mean trajectories for each model?",
    #"Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    #"Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?"
]

#New since 1-5-2024
questionsp3 = [
    "Is the metric or unit of time used in the statistical model reported? (i.e., hours, days, weeks, months, years, etc.)",
    "Is information presented about the mean and variance of time within a wave?(mean and variance of: within measurement occasion, mean and variance of: within a period of time, etc.)",
    "Is the missing data mechanism reported? (i.e., missing at random (MAR), Missing not at random (MNAR), missing completely at random (MCAR), etc.) ",
    "Is a description provided of what variables are related to attrition/missing data? (i.e., a dropout effect, auxillary variables, skip patterns, etc.)",
    "Is a description provided of how missing data in the analyses were dealt with?(i.e., List wise deletion, multiple imputation, Full information maximum likelihood (FIML) etc.)",
    "Is information about the distribution of the observed variables included? (i.e., tests for normally distributed variables within classes, multivariarte normality, etc.) ",
    "Is the software that was used for the statistical analysis mentioned? (i.e., Mplus, R, etc.)",
    "Are alternative specifications of within-class heterogeneity considered (e.g., LGCA vs. LGMM) and clearly documented?",
    "Are alternative specifications of the between-class differences in variance covariance matrix structure considered and clearly documented? (i.e., constrained accros subgroups, fixed accross subgroups, etc.)",
    "Are alternative shape/functional forms of the trajectories described? (e.g., was it tested whether a quadratic trend or a non-linear form would fit the data better)",
    "If covariates or predictors have been used, is it done in such a way that the analyses could be replicated? (e.g., was it reported they used time-varying or time-invariant covariates at the level of the dependent or independent variables)",
    "Is information reported about the number of random start values and final iterations included? (e.g., If ML has been used to estimate the latent trajectory model, then it should be reported if the final class solution has converged to the maximum of the ML distribution and not on a local maxima.)",
    "Are the model comparison (and selection) tools described from a statistical perspective? (i.e., BIC, AIC, etc.)",
    "Are the total number of fitted models reported, including a one-class solution?",
    "Are the number of cases per class reported for each model (absolute sample size, sample size per class or proportion)?",
    "If classification of cases in a trajectory is the goal, is entropy reported? (i.e., the relative entropy value, the number of misclassifications per model)",
    #"Is a plot included with the estimated mean trajectories of the final solution?",
    #"Are plots included with the estimated mean trajectories for each model?",
    #"Is a plot included of the combination of estimated means of the final model and the observed individual trajectories split out for each latent class?",
    "Are characteristics of the final class solution numerically described (i.e., means, SD/SE, n, CI, etc.)?",
    #"Are the syntax files available (either in the appendix, supplementary materials, or from the authors)?"
]

papers = []
for i in range(1):
    papers.append("./data/" + str(i) + ".pdf")    

# Create DataFrame
df = pd.DataFrame(columns=questionsp3) #PUT IN THE RIGHT QUESTIONS HERE
df['Paper'] = papers
# Fill other columns with empty values
for index, paper in enumerate(papers):
    df.loc[index, questionsp3] = ''  #PUT IN THE RIGHT QUESTIONS HERE

def fill_cells(row):
    paper_value = row['Paper']
    for column_name, cell_value in row.items():
        if column_name != 'Paper':
            row[column_name] = generate_output(paper_value,column_name)
    return row


import time
start_time = time.time()

# Apply the function to each cell in the DataFrame
df = df.apply(fill_cells, axis=1)

# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

print(f"Elapsed time: {elapsed_time} seconds")


#This works however is not very efficient, for each question it regenerates the chromadb wich just costs moremoney. I need to make this a two step process.

df.to_excel('outputp3.xlsx', index=False)


#print(generate_output(the_paper,the_check))



