# GRoLTS-llm
This repository supports the testing and improvement of the GRoLTS checklist using open-source large language models (LLMs).

## Purpose

The scripts provided here allow you to calculate GRoLTS scores — based on the checklist described in [this publication](https://doi.org/10.1080/10705511.2016.1247646) — for the PTSS datasets:

- [PTSS Dataset 1](https://doi.org/10.34894/YXR1X3)  
- [PTSS Dataset 2](https://doi.org/10.34894/CRE6ZC)

This work is part of the FORAS project, which is pre-registered in PROSPERO under ID [CRD42023494027](https://www.crd.york.ac.uk/PROSPERO/view/494027).

## Installation
Tested with `Python 3.13`
1. Install all packages in `requirements.txt`:
```
pip install -r ./requirements.txt
```

2. Place your paper pdfs in the folder `./src/data`

3. Copy the OpenAI API key in the corresponding fields in the files: `./src/embed.py` and `./src/generate.py`

4. Update the questions in `p4` in the file `./src/grolts_questions.py`

5. Move into the `./src` folder: `cd ./src`

6. Run `./embed.py` to create chunks of the papers and embed these chunks and questions

7. After `./embed.py` completes, run `./generate.py`. This will take a while!

8. After this completes, ensure to update `./human_labels.csv` with the ground truth labels (Pay extra attention to the question ids, they should match with the ids in `p4` in `./grolts_questions.py`. Also, make sure that the paper_ids correspond with the names of the pdfs in the `./data` folder)

9. Run the first 4 cells in `./eval.ipynb` and see the agreement proportion between the LLM and yourself!

## Funding

This project is funded by the Dutch Research Council (NWO), grant number 406.22.GO.048.
