# GRoLTS-llm
[![DOI](https://zenodo.org/badge/887834859.svg)](https://doi.org/10.5281/zenodo.15582825)

This repository supports the testing and improvement of the GRoLTS checklist using open-source large language models (LLMs).

## Purpose

The scripts provided here allow you to calculate GRoLTS scores — based on the checklist described in [this publication](https://doi.org/10.1080/10705511.2016.1247646) — for the PTSD datasets:

- [PTSD Dataset 1](https://doi.org/10.34894/YXR1X3)  
- [PTSD Dataset 2](https://doi.org/10.34894/CRE6ZC)

## Installation

Tested with **Python 3.13** and an HPC cluster with NVIDIA H100 GPUs.

1. **Create virtual environment and install dependencies**
```
uv sync
```
- Installs all dependencies listed in `pyproject.toml` via [uv](https://docs.astral.sh/uv/).


2. **Prepare PDFs**  
Place your PDFs in `./src/data`. Organize PDFs in subfolders corresponding to case studies:
```
achievement/
delinquency/
ptsd/
wellbeing/
```

3. **Convert PDFs to Markdown**
```
sbatch ./src/generate_markdown.sh
```
- Markdown files are stored in `./src/processed_pdfs` in the corresponding subfolders.

4. **Generate embeddings for documents and questions**
```
sbatch ./src/run_generate_embeddings.sh
```
- Creates passage chunks of `500` and `1000` words.
- Stores embeddings in ChromaDB: `./src/document_embeddings`.
- Embeds questions and stores them in `./src/question_embeddings`.

5. **Create batch files for LLM inference**
```
./src/generate_batches.py
```
- Uses document and question embeddings.
- Creates JSONL batch request files for each combination of `subfolder`, `chunk_size`, and `question_id`.
- Files are stored in `./src/batches`.

6. **Generate LLM responses**
```
sbatch ./src/run_generate_responses.sh
```
- Processes batch files from the previous step.
- Responses are stored in JSONL format in `./eval/batches_out`. Each input file has a corresponding output file.

7. **Process batch results to CSV**
```
./eval/process_batch_result.py
```
- Creates `.csv` files containing answers to each question for each PDF.
- Adds a column `score` with the final GRoLTS score.
- One CSV per output batch file.

8. **Run evaluation**  
Open and run all cells and compare results across case studies and question sets in:
```
./eval/eval.ipynb
```

## Pipeline Overview
```
PDFs → Markdown
(sbatch generate_markdown.sh)
    │
    ▼
Split & Embed Documents and Questions
(sbatch run_generate_embeddings.sh)
    │
    ▼
Generate Batch JSONL
(./src/generate_batches.py)
    │
    ▼
LLM Responses
(sbatch run_generate_responses.sh)
    │
    ▼
Process Batch Results
(./eval/process_batch_result.py)
    │
    ▼
CSV Outputs & Evaluation Notebook
(./outputs, ./eval/eval.ipynb)
```

## Notes

- **Batching & Memory:** The scripts are optimized for HPC environments with large GPU memory (e.g., NVIDIA H100).
- **Tokenization:** Prompts are tokenized per batch to respect GPU memory limits.
- **Outputs:** The `.csv` files contain one row per PDF and one column per question, plus the aggregated GRoLTS score.

## Funding 
The research is supported by the Dutch Research Council under grant number 406.22.GO.048
