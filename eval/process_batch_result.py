import glob
import os
import re
from typing import Any, Dict, List

import pandas as pd

INPUT_PATH = "./batches_out"
OUTPUT_PATH = "./outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def parse_response(response: str) -> Dict[str, str]:
    """
    Parse a single model response into structured fields: answer, reasoning, evidence.
    """
    parsed = {}
    current_section = None
    for line in response.split("\n"):
        line = line.strip()
        if "ANSWER" in line:
            current_section = "answer"
            ans = line.replace("ANSWER:", "").strip().strip("*").upper()
            parsed["answer"] = ans
        elif "REASONING" in line:
            current_section = "reasoning"
            parsed["reasoning"] = line.replace("REASONING:", "").strip().strip("*")
        elif "EVIDENCE" in line:
            current_section = "evidence"
            parsed["evidence"] = line.replace("EVIDENCE:", "").strip().strip("*")
        elif current_section:
            # Append continuation lines
            parsed[current_section] = (
                parsed.get(current_section, "") + " " + line
            ).strip()
    return parsed


def replace_yes_no(value: str) -> int:
    """
    Convert 'YES' -> 1, 'NO' -> 0, leave other values unchanged but print warning.
    """
    if isinstance(value, str):
        if re.search(r"\bYES\b", value, re.IGNORECASE):
            return 1
        elif re.search(r"\bNO\b", value, re.IGNORECASE):
            return 0
        else:
            print(f"[WARN] Unexpected string answer value: {value}")
    else:
        print(f"[WARN] Unexpected non-string answer value: {value}")
    return 0


def process_file(filepath: str) -> None:
    """
    Process a single JSONL batch file, parse responses, pivot to wide format, and save CSV.
    """
    df = pd.read_json(filepath, lines=True, dtype={"custom_id": str})
    filename = os.path.basename(filepath).replace(".jsonl", "")

    # Extract responses and paper/question IDs
    if "completion" in df.columns:
        responses = df["completion"]
        pdf_qid = df["custom_id"].apply(
            lambda x: {"paper": x.split("_")[0], "question": x.split("_")[1]}
        )
    else:
        df_norm = pd.json_normalize(df["response"].to_numpy(), max_level=1)
        responses = df_norm["body.choices"].apply(
            lambda x: x[0]["message"]["content"]
            if isinstance(x, list) and len(x) > 0
            else ""
        )
        pdf_qid = df["custom_id"].apply(
            lambda x: {"paper": x.split("_")[0], "question": x.split("_")[1]}
        )

    # Parse responses
    results: List[Dict[str, Any]] = []
    for i, resp in enumerate(responses):
        parsed = parse_response(resp)
        results.append(
            {
                "paper_id": pdf_qid[i]["paper"],
                "question_id": int(pdf_qid[i]["question"]),
                **parsed,
            }
        )

    df_res = pd.DataFrame(results)
    df_res["answer"] = df_res["answer"].apply(replace_yes_no)

    output_file = os.path.join(OUTPUT_PATH, f"{filename}.csv")
    df_res.to_csv(output_file, index=False)
    print(f"[INFO] Saved CSV: {output_file}")


def main() -> None:
    files = glob.glob(os.path.join(INPUT_PATH, "*.jsonl"))
    if not files:
        print(f"[WARN] No JSONL files found in {INPUT_PATH}")
        return
    for f in files:
        process_file(f)


if __name__ == "__main__":
    main()
