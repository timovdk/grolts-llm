# GRoLTS-llm
[![DOI](https://zenodo.org/badge/887834859.svg)](https://doi.org/10.5281/zenodo.15582825)

This repository supports the testing and improvement of the GRoLTS checklist using open-source large language models (LLMs).

## Purpose

The scripts provided here allow you to calculate GRoLTS scores — based on the checklist described in [this publication](https://doi.org/10.1080/10705511.2016.1247646) — across three case studies:

- **PTSD**, from the original GRoLTS study ([dataset 1](https://doi.org/10.34894/YXR1X3), [dataset 2](https://doi.org/10.34894/CRE6ZC))
- **Educational achievement** and **adolescent delinquency**, built for this study (<https://osf.io/kw8a7>); see `eval/human_labels/achievement.md` and `delinquency.md` for their search and screening protocols

Two checklist versions are used: the original (`--qset 0`, 21 items) and the revised version
(`--qset 4`, 19 items), both defined in `src/grolts_questions.py`.

## Installation

Tested with **Python 3.13** and an HPC cluster with NVIDIA H100 GPUs.

1. **Create virtual environment and install dependencies**
```
uv sync                      # full pipeline (torch, marker, chromadb, ...)
uv sync --only-group eval    # analysis only — enough to reproduce every table and figure
```
- Installs dependencies listed in `pyproject.toml` via [uv](https://docs.astral.sh/uv/).
- Reproducing the *results* needs only the `eval` group; the generation pipeline needs GPUs.


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
- Creates passage chunks; set `CHUNK_SIZES` in the script for the sizes you want (`500`, `1000`, or both).
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

For locally hosted models — one script for all four, selected by argument:
```
sbatch          ./src/run_generate_responses.sh qwen3-30b       ptsd 0 1000
sbatch --gpus=2 ./src/run_generate_responses.sh llama-3.3-70b   ptsd 0 1000
```
Models: `qwen3-30b`, `qwen3-next-80b`, `llama-3.3-70b`, `magistral-small`. The 70B and 80B
models need two H100s. Completed runs are skipped, so an interrupted job can be resubmitted.

For `gpt-5-mini`, via the OpenAI Batch API (needs `OPENAI_API_KEY`):
```
python ./src/submit_openai_batch.py --dataset ptsd --qset 0 --chunk 1000
```
Or, to hand the requests to someone else's account, `--export` writes ready-to-submit JSONL files
and a set of instructions into `packages/openai_batches/` without needing a key.

To quantify run-to-run variability, add `--runs N --temperature 0.7`; outputs then carry a
`_run{i}` suffix and are analysed by `ge.response_stability` / `ge.majority_vote`.

- Responses are stored in JSONL format in `./eval/batches_out`, one output file per input file.

7. **Process batch results to CSV**
```
./eval/process_batch_result.py
```
- Creates `.csv` files with one row per (PDF, question), holding the model's reasoning, quoted
  evidence and binary answer.
- One CSV per output batch file.

8. **Run evaluation**

Shared analysis code lives in `./eval/grolts_eval.py`; the notebooks are thin drivers over it.
```
./eval/results_figures.ipynb      # manuscript tables and figures
./eval/revision_analyses.ipynb    # analyses added in response to reviewers
```
Set `CHUNK` at the top of `results_figures.ipynb` to switch between the 1000- and 500-word
retrieval configurations — there is no separate notebook per chunk size. Figures are written to
`./eval/viz/chunk{CHUNK}/`.

See [`REVISION.md`](REVISION.md) for the status of the reviewer response and for what still
requires manual work.

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
(sbatch run_generate_responses.sh <model> <dataset> <qset> <chunk>)
(python submit_openai_batch.py  — for gpt-5-mini)
    │
    ▼
Process Batch Results
(./eval/process_batch_result.py)
    │
    ▼
CSV Outputs & Evaluation Notebooks
(./eval/outputs, ./eval/results_figures.ipynb, ./eval/revision_analyses.ipynb)
```

## Notes

- **Batching & Memory:** The scripts are optimized for HPC environments with large GPU memory (e.g., NVIDIA H100).
- **Tokenization:** Prompts are tokenized per batch to respect GPU memory limits.
- **Outputs:** The `.csv` files contain one row per (PDF, question) with `reasoning`, `evidence`
  and `answer` columns; totals are computed at analysis time in `eval/grolts_eval.py`.
- **Human labels:** `eval/human_labels/*.csv` are `;`-delimited, one row per study and one column
  per checklist item, with 0-based item ids (so `question_id` 12 is item 13 in the manuscript).
