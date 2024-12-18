# GRoLTS-llm
A repository for testing and improving the GRoLTS checklist on open-source llms.

## Installation
Tested with `Python 3.13`
1. Install all packages in `requirements.txt`:
```
pip install -r ./requirements.txt
```

2. Create the file `/src/.env` and copy the contents of `./src/env.example` to it. Make sure to also fill in your API keys in the fields `OPENAI_API_KEY` and `GROQ_API_KEY`

3. Populate the data folder `/src/data/`

4. Run the python file using `python3 main.py` and find the output files in `/src/data_out/`