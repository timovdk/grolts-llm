import glob
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd

INPUT_PATH = "./batches_out"
OUTPUT_PATH = "./outputs"
os.makedirs(OUTPUT_PATH, exist_ok=True)


def parse_response(response: str) -> Dict[str, str]:
    """
    Parse a single model response into structured fields: answer, reasoning, evidence.
    """
    response = response.strip().lstrip("<s>").rstrip("</s>").strip()
    parsed = {}
    current_section = None
    error = False

    for line in response.split("\n"):
        # Normalize line
        line_clean = line.strip().strip("<>/").strip()
        if not line_clean:
            continue

        # Detect headers
        if (
            "ANSWER" in line_clean
            or re.match(r"^\s*A\s*:(?!\w)", line_clean)
            or "Answer:" in line_clean
            or "answer:" in line_clean
        ):
            current_section = "answer"
            # If YES/NO is on the same line
            if re.search(r"\bYES\b", line_clean):
                parsed["answer"] = "YES"
                current_section = "skip"
                error = False
            elif re.search(r"\bNO\b", line_clean):
                parsed["answer"] = "NO"
                current_section = "skip"
                error = False
            else:
                parsed["answer"] = ""  # will be filled by next line

        elif (
            "REASONING" in line_clean
            or re.match(r"^\s*R:", line_clean)
            or "Reasoning:" in line_clean
            or "reasoning:" in line_clean
        ):
            current_section = "reasoning"
            parsed["reasoning"] = (
                line_clean.replace("REASONING", "").strip().strip("*:")
            )

        elif (
            "EVIDENCE" in line_clean
            or re.match(r"^\s*E:", line_clean)
            or "Evidence:" in line_clean
            or "evidence:" in line_clean
        ):
            current_section = "evidence"
            parsed["evidence"] = line_clean.replace("EVIDENCE", "").strip().strip("*:")

        elif current_section:
            # continuation lines
            if current_section == "answer" and parsed.get("answer", "") == "":
                # answer is on this line
                if re.search(r"\bYES\b", line_clean):
                    parsed["answer"] = "YES"
                    current_section = "skip"
                    error = False
                elif re.search(r"\bNO\b", line_clean):
                    parsed["answer"] = "NO"
                    current_section = "skip"
                    error = False
                else:
                    parsed["answer"] = line_clean
                    error = True
            elif current_section in {"reasoning", "evidence"}:
                parsed[current_section] = (
                    parsed.get(current_section, "") + " " + line_clean
                ).strip()

    if error:
        print(f"[WARN] Could not parse response properly:\n{response}")
        print(f"[WARN] Parsed fields: {parsed}")
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
    return np.nan


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
