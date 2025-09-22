import glob
import os
import re

import pandas as pd

files = glob.glob("./batches_out/*.jsonl")

for f in files:
    df = pd.read_json(f, lines=True, dtype={"custom_id": str})
    filename = os.path.basename(f).strip(".jsonl")
    if "completion" in df:
        responses = df["completion"]
        pdf_qid = df["custom_id"].apply(
            lambda x: {"pdf": x.split("_")[0], "q_id": x.split("_")[1]}
        )
    else:
        df1 = pd.json_normalize(df["response"].to_numpy(), max_level=1)
        responses = df1["body.choices"].apply(
            lambda x: x[0]["message"]["content"]
            if isinstance(x, list) and len(x) > 0
            else None
        )
        pdf_qid = df.custom_id.apply(
            lambda x: {"pdf": x.split("_")[0], "q_id": x.split("_")[1]}
        )

    results = []
    for i in range(len(pdf_qid)):
        pdf_id = pdf_qid[i]["pdf"]
        q_id = pdf_qid[i]["q_id"]
        response = responses[i]

        parsed_response = {}
        current_section = None
        for item in response.split("\n"):
            item = item.strip()
            if "ANSWER" in item:
                current_section = "answer"
                if "YES" in item:
                    parsed_response["answer"] = "YES"
                elif "UNSURE" in item:
                    parsed_response["answer"] = "UNSURE"
                elif "NO" in item:
                    parsed_response["answer"] = "NO"
                else:
                    parsed_response["answer"] = item.replace("ANSWER:", "").strip()
            elif "REASONING" in item:
                current_section = "reasoning"
                parsed_response["reasoning"] = item.replace("REASONING:", "").strip()
            elif "EVIDENCE" in item:
                current_section = "evidence"
                parsed_response["evidence"] = item.replace("EVIDENCE:", "").strip()
            elif current_section:
                # Append to the current section if the item is a continuation of the previous line
                parsed_response[current_section] = (
                    parsed_response.get(current_section, "") + " " + item
                )
        results.append(
            {
                "paper_id": pdf_id,
                "question_id": q_id,
                **parsed_response,
            }
        )
    df_res = pd.DataFrame(results)

    # Function to replace sentences containing 'yes' or 'no' with 1 or 0
    def replace_yes_no(sentence):
        if isinstance(sentence, str):  # Check if the value is a string
            if re.search(r"\b(yes)\b", sentence, re.IGNORECASE):
                return 1
            elif re.search(r"\b(no)\b", sentence, re.IGNORECASE):
                return 0
            else:
                print(f"Unexpected value: {sentence}")
        return sentence

    df_res["answer"] = df_res["answer"].fillna("no")
    df_res["answer"] = df_res["answer"].apply(replace_yes_no)
    
    df_res["answer"] = df_res["answer"].astype(int)
    df_res['question_id'] = df_res['question_id'].astype(int)

    # pivot to wide format
    df_wide = df_res.pivot_table(
        index="paper_id",
        columns="question_id",
        values="answer",  # use 'answer' values
        aggfunc="first",  # or sum/mean if there are duplicates
    )
    
    # reindex columns to ensure all 0–17 exist (fill missing with 0)
    df_wide = df_wide.reindex(columns=range(18), fill_value=0)

    # compute score as row sum
    df_wide["score"] = df_wide.sum(axis=1)

    # reset index to make paper_id a column
    df_wide = df_wide.reset_index()

    df_wide.to_csv(f"./outputs/{filename}.csv", index=False)
