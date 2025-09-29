# GRoLTS-llm
A repository for generating the GRoLTS scores used in the meta-analysis of the FORAS project on PTSD trajectories following traumatic events (Pre-print: https://doi.org/10.31219/osf.io/fkjb2_v1).

It is part of the **Hunt for the Last Relevant Paper** project, pre-registered  as "[Trajectories of PTSD Following Traumatic Events: A Systematic and Multi-database Review](https://www.crd.york.ac.uk/prospero/display_record.php?RecordID=494027)".

## Installation
Tested with `Python 3.13` and a HPC cluster with NVIDIA H100 GPUs.
1. Install all packages in `requirements.txt`:
```
pip install -r ./requirements.txt
```

2. Place your paper pdfs in the folder `./src/data`

3. Run `generate_markdown.sh` to transform PDFs to Markdown.

4. Run `generate_embeddings.py` to turn the Markdown files into passages and store them with their embeddings in a ChromaDB.

5. Run `generate_batch.py` to create the tasks that are to be send to the LLM, so a combination of the prompt, the question, and the most relevant passages from a paper.

6. Run `generate_responses.py` to use the batch file to generate the responses from the LLM. (note that there is also a SLURM script for when you have access to a HPC cluster.)

7. Run `process_batch_result.py` to create the output `.csv` file containing the answers to each question for each initially uploaded PDF, and a column called `score` with the final GRoLTS score.

## Funding 
The research is supported by the Dutch Research Council under grant number 406.22.GO.048
