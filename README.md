# GRoLTS-llm
[![DOI](https://zenodo.org/badge/887834859.svg)](https://doi.org/10.5281/zenodo.15582825)

This repository supports the testing and improvement of the GRoLTS checklist using open-source large language models (LLMs).

## Purpose

The scripts provided here allow you to calculate GRoLTS scores — based on the checklist described in [this publication](https://doi.org/10.1080/10705511.2016.1247646) — across three case studies:

- **PTSD**, from the original GRoLTS study ([dataset 1](https://doi.org/10.34894/YXR1X3), [dataset 2](https://doi.org/10.34894/CRE6ZC))
- **Educational achievement** and **adolescent delinquency**, built for this study (<https://osf.io/kw8a7>); see `eval/human_labels/achievement.md` and `delinquency.md` for their search and screening protocols

Two checklist versions are used: the original (`--qset 0`, 21 items) and the revised version
(`--qset 4`, 19 items), both defined in `src/grolts_questions.py`.

## Repository layout

```
src/                     generation pipeline (needs GPUs)
  run_pdf_to_markdown.sh        PDFs -> markdown (marker)
  generate_embeddings.py        chunk + embed documents and questions
  run_generate_embeddings.sh
  generate_batches.py           retrieval -> prompt JSONL in src/batches
  batches/                      prompts; gpt-5-mini reruns in <dataset>_<chunk>_<qset>/
  generate_responses_vllm.py    local generation, vLLM engine
  generate_responses.py         local generation, transformers engine
  run_generate_responses_1gpu.sh  SLURM job array: qwen3-30b, magistral-small
  run_generate_responses_2gpu.sh  SLURM job array: llama-3.3-70b, qwen3-next-80b
  submit_openai_batches.py      gpt-5-mini via the OpenAI Batch API
  grolts_questions.py           the checklist item text, per version
  pipeline_config.py            model registry, paths, .env loading

eval/                    analysis (CPU only)
  process_batch_result.py       raw completions -> one CSV per run
  grolts_eval.py                all shared analysis code
  results_figures.ipynb         manuscript tables and figures
  revision_analyses.ipynb       analyses added for the reviewer response
  make_rater_packages.py        build the double-coding packages for raters
  check_rater_agreement.py      ingest returned workbooks, report agreement
  human_labels/                 reference standards + rating protocols
  outputs/, viz/, tables/       generated artefacts (gitignored)
```

Everything large or copyrighted is gitignored: PDFs, converted markdown, the vector store,
prompt batches, raw completions, parsed CSVs, figures, and the human ratings themselves
(those live on OSF). Only the `.md` provenance documents and `rater_assignments.json` are
kept inside `eval/human_labels`.

## Installation

Tested with **Python 3.13** and an HPC cluster with NVIDIA H100 GPUs.

```
uv sync                      # full pipeline (torch, marker, chromadb, ...)
uv sync --only-group eval    # analysis only — enough to reproduce every table and figure
```
- Installs dependencies listed in `pyproject.toml` via [uv](https://docs.astral.sh/uv/).
- Reproducing the *results* needs only the `eval` group; the generation pipeline needs GPUs.

Keys come from a gitignored `.env` at the repo root — copy `.env.example` and fill in.
`src/pipeline_config.py` loads it with python-dotenv on import, so every entry point sees it.
`HF_TOKEN` is what gets the gated repos (`meta-llama/Llama-3.3-70B-Instruct`) to download;
that token's account must also have accepted the model licence on huggingface.co.
`HF_CACHE_DIR` says where weights are cached, and defaults to the cluster path.

## Running the pipeline

All `src/` scripts use paths relative to `src/`, so run them from there.

### 1. Prepare PDFs

Place your PDFs in `./src/data`, in subfolders per case study:
```
achievement/
delinquency/
ptsd/
wellbeing/
```

### 2. Convert PDFs to Markdown
```
cd src && sbatch run_pdf_to_markdown.sh
```
- Runs `marker` over all four subfolders; markdown lands in `./src/processed_pdfs/<subfolder>`.

### 3. Generate embeddings for documents and questions
```
cd src && sbatch run_generate_embeddings.sh
```
- Cleans the markdown, splits it into logical blocks and chunks them with overlap.
- `CHUNK_SIZES` at the top of `generate_embeddings.py` selects the retrieval configuration
  (`1000` by default; the published work also used `500`).
- Document embeddings go to ChromaDB under `./src/document_embeddings`; question embeddings
  are pickled into `./src/question_embeddings`.

### 4. Create batch files for LLM inference
```
cd src && python generate_batches.py
```
- Retrieves the top-10 chunks per question and writes JSONL prompt files to `./src/batches`,
  one per (`subfolder`, `chunk_size`, `question_id`) and per generator target — `gpt-5-mini`
  for the API path and `generic` for every locally hosted model.
- Existing batch files are **skipped**: they are the prompts behind the published runs, and
  the embeddings needed to rebuild them live only on the cluster. Pass `--overwrite` to
  replace them anyway.

### 5. Generate LLM responses

**Locally hosted models.** The whole grid is queued as two SLURM job arrays, split by how
many GPUs a model needs:
```
cd src
sbatch run_generate_responses_1gpu.sh   # qwen3-30b, magistral-small
sbatch run_generate_responses_2gpu.sh   # llama-3.3-70b, qwen3-next-80b
```
Each script holds eight `model dataset qset` tasks, one per array index. Run either **without**
`sbatch` to list its tasks, and use `--array` to submit a subset:
```
./run_generate_responses_1gpu.sh        # list the tasks
sbatch --array=0,3 run_generate_responses_1gpu.sh
```
Submit from `src/`: the SBATCH log paths are relative to the submit directory and `logs/`
lives there. Behaviour is tuned with environment variables rather than by editing the script:

| variable | default | meaning |
| --- | --- | --- |
| `RUNS` | `5` | sampled runs per task, for run-to-run variability |
| `TEMPERATURE` | `0.7` | sampling temperature for those runs |
| `CHUNK` | `1000` | retrieval configuration |
| `ENGINE` | `vllm` | `vllm`, or anything else for the transformers path |
| `MAX_NUM_SEQS` | `64` | vLLM's engine-side batch limit |

vLLM is the faster path and additionally emits a greedy pass (`--also-greedy`); the
transformers path produced the published runs and is kept for comparison. Completed runs
are skipped, so a job killed by the wall clock can simply be resubmitted.

A single ad-hoc run bypasses the arrays:
```
cd src
python generate_responses_vllm.py --model qwen3-30b --dataset ptsd --qset 4 --chunk 1000 \
    --runs 5 --temperature 0.7 --also-greedy --n-gpus 1
python generate_responses.py      --model qwen3-30b --dataset ptsd --qset 0 --chunk 1000
```
Models: `qwen3-30b`, `qwen3-next-80b`, `llama-3.3-70b`, `magistral-small` — registered in
`src/pipeline_config.py`. The 70B and 80B models need two H100s. vLLM outputs carry a `_vllm`
tag so they never collide with the published transformers runs, and sampled runs carry a
`_run{i}` suffix; `ge.response_stability` and `ge.majority_vote` analyse them.

**gpt-5-mini** goes through the OpenAI Batch API:
```
cd src && python submit_openai_batches.py --submit    # reads OPENAI_API_KEY from .env
```
It reads the rerun request files from `./src/batches/<dataset>_<chunk>_<qset>/` (override with
`--export-dir`). Only that one level down is read, so the greedy prompt files sitting directly
in `batches/` — already spent on the published runs — are never resubmitted. It keeps four
batches in flight at a time (`--max-in-flight`): the Batch API caps enqueued tokens per model
and these are ~10M input tokens each. Each result is written into `./eval/batches_out/` under
the request file's own name, which is how the analysis identifies runs. Run it without
`--submit` for a dry run, or with `--collect` to resume polling. A manifest is written as each
batch is created, so it can be stopped and restarted without resubmitting anything.

All responses end up as JSONL in `./eval/batches_out`, one output file per input file.

### 6. Process batch results to CSV
```
cd eval && python process_batch_result.py
```
- Creates `.csv` files with one row per (PDF, question), holding the model's reasoning, quoted
  evidence and binary answer.
- One CSV per output batch file, written to `./eval/outputs`.

### 7. Run evaluation

Shared analysis code lives in `./eval/grolts_eval.py`; the notebooks are thin drivers over it.
```
./eval/results_figures.ipynb      # manuscript tables and figures
./eval/revision_analyses.ipynb    # analyses added in response to reviewers
```
Set `CHUNK` at the top of either notebook to switch between the 1000- and 500-word retrieval
configurations — there is no separate notebook per chunk size. Figures are written to
`./eval/viz/chunk{CHUNK}/`. (`results_figures_500.ipynb` is the superseded standalone version
of the chunk-500 analysis, kept only for provenance.)

See [`REVISION.md`](REVISION.md) for the status of the reviewer response and for what still
requires manual work.

## Human double-coding

The revision adds a second human rating so that human–LLM agreement can be read against a
human–human benchmark.

```
uv run --group eval python eval/make_rater_packages.py       # build packages/rater{1,2,3}.zip
uv run --group eval python eval/check_rater_agreement.py --dataset ptsd
```
- `make_rater_packages.py` draws the double-coding subsets with a fixed seed, records them in
  `eval/human_labels/rater_assignments.json`, and writes one zip per rater containing the PDFs,
  a pre-formatted Excel workbook and the codebook.
- Rater 1 rates all 38 PTSD studies and becomes the PTSD v2 reference standard; raters 2 and 3
  rate random halves to estimate human–human agreement. Achievement and Delinquency keep their
  existing labels as the reference.
- Put returned workbooks in `eval/returned/` under the name they were issued
  (`rater1_grolts_v2.xlsx` and so on). `check_rater_agreement.py` validates them, writes
  normalised label and notes CSVs into `eval/human_labels/`, and reports the agreement.

## Pipeline Overview
```
PDFs → Markdown
(sbatch run_pdf_to_markdown.sh)
    │
    ▼
Split & Embed Documents and Questions
(sbatch run_generate_embeddings.sh)
    │
    ▼
Generate Batch JSONL
(python generate_batches.py)
    │
    ▼
LLM Responses
(sbatch run_generate_responses_{1,2}gpu.sh — local models)
(python submit_openai_batches.py --submit — gpt-5-mini)
    │
    ▼
Process Batch Results
(python eval/process_batch_result.py)
    │
    ▼
CSV Outputs & Evaluation Notebooks
(eval/outputs, eval/results_figures.ipynb, eval/revision_analyses.ipynb)
```

## Notes

- **Batching & Memory:** The scripts are optimized for HPC environments with large GPU memory (e.g., NVIDIA H100).
- **Context window:** vLLM runs with a 32k context, which holds the largest prompt (~21.6k
  tokens) plus generation. vLLM rejects anything longer rather than truncating silently.
- **Outputs:** The `.csv` files contain one row per (PDF, question) with `reasoning`, `evidence`
  and `answer` columns; totals are computed at analysis time in `eval/grolts_eval.py`.
- **Human labels:** `eval/human_labels/*.csv` are `;`-delimited, one row per study and one column
  per checklist item, with 0-based item ids (so `question_id` 12 is item 13 in the manuscript).
- **Checklist mapping:** PTSD was rated by a human against v1 only, so scoring v2 model answers
  against it goes through `ge.ID_MAP` / `ge.INVERSE_ID_MAP`. v1 items 13 and 14 were each split
  in two, so the mapped labels contain duplicated values (`ge.DUPLICATED_BY_MAPPING`).
