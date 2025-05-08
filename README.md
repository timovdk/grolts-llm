# GRoLTS-llm
A repository for testing and improving the GRoLTS checklist on open-source llms.

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