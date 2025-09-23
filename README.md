# GRoLTS-llm
A repository for generating the GRoLTS scores for the GRoLTS update project.

## Installation
Tested with `Python 3.13` and a HPC cluster with NVIDIA h100 GPUs.
1. Run `uv sync` to create a virtual environment and install all dependencies in `pyproject.toml` through [uv](https://docs.astral.sh/uv/).

2. Place your paper pdfs in the folder `./src/data`
    - For this project, the PDFs should be in subfolders, corresponding to the different case studies: `achievement`, `delinquency`, `ptsd`, and `wellbeing`

3. Run `sbatch ./src/generate_markdown.sh` to transform PDFs in each subfolder to Markdown. The result is stored in `./src/processed_pdfs` in their respective subfolders.

4. Run `sbatch ./src/run_generate_embeddings.sh` to turn the Markdown files into passages of `500` and `1000` words, and store them with their embeddings in a ChromaDB in `./src/document_embeddings`.
    - This script also embeds the questions and stores them in `./src/question_embeddings`.

5. Run `./src/generate_batches.py` to create OpenAI JSONL batch request files using the document and question embeddings created in the previous step.
    - A batch file is created for each combination of `subfolder`, `chunk_size`, and `question_id`. They are stored in `./src/batches`.

6. Run `sbatch ./src/run_generate_responses.sh` to use the batch files from the previous step to generate the responses from the LLM. The responses are stored in JSONL format, with one input file corresponding to one output file, stored in `./eval/batches_out`

7. Run `./eval/process_batch_result.py` to create the output `.csv` files containing the answers to each question for each initially uploaded PDF, and a column called `score` with the final GRoLTS score. One CSV is created for each output batch file.

8. Run all cells in the notebook `./eval/eval.ipynb` to run the comparisons.
