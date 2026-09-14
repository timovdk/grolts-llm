"""Ingest returned rating workbooks and report human-human agreement.

Put the workbooks the raters send back in ``eval/returned/`` under the name they were issued
(``rater1_grolts_v2.xlsx`` and so on), then::

    uv run --group eval python eval/check_rater_agreement.py --dataset ptsd

The script reads the workbook, validates it, writes normalised label and notes CSVs into
``eval/human_labels/``, and reports agreement between the two ratings of that dataset.

Which two ratings depends on the dataset. PTSD has no pre-existing v2 labels, so rater 1 (who
rates all 38) is the reference standard and rater 2's half supplies the reliability estimate.
Achievement and Delinquency were already rated natively on v2 by the original expert, so those
labels stay the reference and rater 3's half is compared against them.

Answers reviewer 1 major 2 and reviewer 2 major 1: without a second rating there is no benchmark
to read human-LLM agreement against, and "accuracy" against a single rater overstates what the
comparison can support.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grolts_eval as ge  # noqa: E402

RETURNED_DIR = Path(__file__).resolve().parent / "returned"
MANIFEST = ge.LABEL_DIR / "rater_assignments.json"
LEAD_COLUMNS = ["paper_id", "Ambiguous items", "Notes"]


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------


def read_sheet(path: Path, dataset: str) -> pd.DataFrame:
    """Read one dataset's sheet from a returned workbook.

    Everything downstream joins on the ``paper_id`` column, never on row position: raters sort
    and filter, and row order does not survive the round trip.
    """
    workbook = load_workbook(path, data_only=True)
    title = dataset.capitalize()
    if title not in workbook.sheetnames:
        raise SystemExit(f"{path.name}: no sheet named {title!r} (has {workbook.sheetnames})")

    rows = list(workbook[title].values)
    frame = pd.DataFrame(rows[1:], columns=[str(c) for c in rows[0]])
    frame = frame.dropna(subset=["paper_id"])
    frame["paper_id"] = frame["paper_id"].astype(int)
    return frame


def item_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in LEAD_COLUMNS]


def validate(name: str, frame: pd.DataFrame, expected: list[int]) -> tuple[int, list[str]]:
    """Return (cells filled, problems). Reports partial progress rather than failing."""
    problems: list[str] = []
    items = item_columns(frame)

    if len(items) != 19:
        problems.append(f"expected 19 item columns, found {len(items)}")

    seen = sorted(frame["paper_id"].tolist())
    if seen != sorted(expected):
        missing = sorted(set(expected) - set(seen))
        extra = sorted(set(seen) - set(expected))
        if missing:
            problems.append(f"{len(missing)} assigned papers absent: {missing[:8]}")
        if extra:
            problems.append(f"{len(extra)} papers not in the assignment: {extra[:8]}")

    values = frame[items]
    stacked = values.stack(future_stack=True).dropna()
    bad = stacked[~stacked.isin([0, 1, 0.0, 1.0])]
    if len(bad):
        problems.append(f"{len(bad)} cells are neither 0 nor 1 (e.g. {list(bad.unique()[:5])})")

    blank = int(values.isna().sum().sum())
    if blank:
        worst = values.isna().sum()
        worst = worst[worst > 0].sort_values(ascending=False).head(5)
        problems.append(
            f"{blank} cells still blank; most incomplete items: "
            + ", ".join(f"item {k}={v}" for k, v in worst.items())
        )
    return int(values.notna().sum().sum()), problems


def to_long(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    """Wide sheet -> long ``paper_id, question_id, <column>``, 0-based question ids."""
    items = item_columns(frame)
    long = frame.melt(
        id_vars=["paper_id"], value_vars=items, var_name="question_id", value_name=column
    )
    long["question_id"] = long["question_id"].astype(int) - 1  # sheets are 1-based
    return long.dropna(subset=[column]).astype({column: int})


def write_outputs(dataset: str, rater: str, frame: pd.DataFrame) -> None:
    """Write the label CSV in the existing format, and notes to a separate file."""
    items = item_columns(frame)
    wide = frame.set_index("paper_id")[items].sort_index()
    wide.columns = [int(c) - 1 for c in items]
    labels_path = ge.LABEL_DIR / f"{dataset}_v2_{rater}.csv"
    wide.to_csv(labels_path, sep=";", index_label="paper_id")

    notes = frame[["paper_id", "Ambiguous items", "Notes"]].rename(
        columns={"Ambiguous items": "ambiguous_items", "Notes": "notes"}
    )
    notes_path = ge.LABEL_DIR / f"{dataset}_v2_{rater}_notes.csv"
    notes.sort_values("paper_id").to_csv(notes_path, index=False)
    print(f"    labels -> {labels_path.name}   notes -> {notes_path.name}")


# --------------------------------------------------------------------------------------
# Agreement
# --------------------------------------------------------------------------------------


def pooled_kappa(frame: pd.DataFrame) -> float:
    if frame["a"].nunique() < 2 and frame["b"].nunique() < 2:
        return float("nan")
    return float(cohen_kappa_score(frame["a"], frame["b"]))


def report_agreement(merged: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    print(f"\n{'=' * 70}\n{name_a} vs {name_b}: {len(merged)} co-rated cells\n{'=' * 70}")

    raw = float((merged["a"] == merged["b"]).mean())
    kappa = pooled_kappa(merged)
    low, high = ge.bootstrap_paper_ci(merged, pooled_kappa)
    print(f"raw agreement : {raw:.3f}")
    print(f"Cohen's kappa : {kappa:.3f}  (95% CI {low:.3f} to {high:.3f}, studies resampled)")

    rows = []
    for qid, group in merged.groupby("question_id"):
        agree = float((group["a"] == group["b"]).mean())
        rows.append({
            "item": qid + 1,
            "n": len(group),
            "raw agreement": agree,
            # Prevalence- and bias-adjusted: defined even where kappa is not.
            "PABAK": 2 * agree - 1,
            "kappa": pooled_kappa(group),
            f"prev {name_a}": group["a"].mean(),
            f"prev {name_b}": group["b"].mean(),
        })
    per_item = pd.DataFrame(rows).sort_values("raw agreement")
    print("\nPer item (lowest agreement first):")
    print(per_item.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(
        "\nkappa is NaN where an item has no variance in one rater. At these sample sizes roughly"
        "\nthree items per dataset fall out that way, which is why raw agreement and PABAK are"
        "\nreported alongside rather than kappa alone."
    )
    return per_item


def compare_to_models(dataset: str, per_item: pd.DataFrame) -> None:
    """Do the humans stumble on the same items the models do?

    The human analogue of the cross-domain stability result: if per-item human-human agreement
    tracks per-item human-LLM agreement, the difficulty is a property of the item rather than a
    limitation of the models.
    """
    from scipy.stats import spearmanr

    mapped = dataset == "ptsd"
    table = ge.build_rater_table(
        ge.load_human_labels(dataset),
        ge.output_files(dataset, chunk=1000, qset=ge.QSET_V2),
        ge.INVERSE_ID_MAP if mapped else None,
    )
    model_agreement = (
        ge.item_accuracy_from_raters(table)
        .set_index("question_id")[ge.MODEL_LABELS]
        .mean(axis=1)
    )
    joined = per_item.assign(question_id=per_item["item"] - 1).set_index("question_id")
    joined["human-LLM"] = model_agreement
    joined = joined.dropna(subset=["human-LLM"])

    rho, p = spearmanr(joined["raw agreement"], joined["human-LLM"])
    print(f"\n{'=' * 70}\nDo humans stumble on the same items as the models?\n{'=' * 70}")
    print(f"Spearman rho between per-item human-human and human-LLM agreement: {rho:.2f} (p={p:.4f})")
    if mapped:
        print("NOTE: the human-LLM side still uses the mapped v1 labels for PTSD; rerun once")
        print("      rater 1's ratings are the reference standard.")


def ambiguity_tally(dataset: str, raters: list[str]) -> None:
    """Which items did the raters themselves flag as unclear?"""
    counts: dict[int, int] = {}
    total = 0
    for rater in raters:
        path = ge.LABEL_DIR / f"{dataset}_v2_{rater}_notes.csv"
        if not path.exists():
            continue
        notes = pd.read_csv(path)
        for entry in notes["ambiguous_items"].dropna():
            for token in str(entry).replace(";", ",").split(","):
                token = token.strip()
                if token.isdigit():
                    counts[int(token)] = counts.get(int(token), 0) + 1
                    total += 1
    if not total:
        return
    print(f"\n{'=' * 70}\nItems the raters flagged as ambiguous\n{'=' * 70}")
    flagged = pd.Series(counts).sort_values(ascending=False)
    print(flagged.to_string(header=False))
    print("\nCompare with the low-agreement items in revision_analyses.ipynb: convergence between")
    print("what a rater calls unclear and where the models disagree is the diagnostic claim.")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def load_reference(dataset: str) -> pd.DataFrame:
    """The existing expert labels, as long ``paper_id, question_id, a``."""
    return ge.load_human_labels(dataset).rename(columns={"answer": "a"})


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dataset", required=True, choices=sorted(ge.DATASETS))
    args = parser.parse_args(argv)

    if not MANIFEST.exists():
        raise SystemExit(f"no manifest at {MANIFEST}; run make_rater_packages.py first")
    manifest = json.loads(MANIFEST.read_text())

    assigned = {
        rater: work[args.dataset]["paper_ids"]
        for rater, work in manifest["raters"].items()
        if args.dataset in work
    }
    if not assigned:
        raise SystemExit(f"no rater is assigned to {args.dataset}")

    print(f"{args.dataset}: {', '.join(f'{r} ({len(v)} papers)' for r, v in assigned.items())}")

    total_cells = 19
    ratings: dict[str, pd.DataFrame] = {}
    for rater, expected in assigned.items():
        path = RETURNED_DIR / f"{rater}_grolts_v2.xlsx"
        if not path.exists():
            print(f"\n  {rater}: not returned yet ({path} missing)")
            continue
        frame = read_sheet(path, args.dataset)
        filled, problems = validate(rater, frame, expected)
        print(f"\n  {rater}: {filled}/{len(expected) * total_cells} cells filled")
        for problem in problems:
            print(f"    - {problem}")
        write_outputs(args.dataset, rater, frame)
        ratings[rater] = frame

    if not ratings:
        print("\nNothing returned yet; nothing to compare.")
        return 0

    # PTSD's reference is whoever rated the whole set; elsewhere it is the existing expert labels.
    # Only raters actually assigned to this dataset count -- a missing assignment also has no
    # "fraction", and must not be mistaken for one covering the whole set.
    full_raters = [
        r for r in assigned if manifest["raters"][r][args.dataset]["fraction"] is None
    ]
    if full_raters and full_raters[0] in ratings:
        name_a = full_raters[0]
        side_a = to_long(ratings[name_a], "a")
    elif full_raters:
        print(f"\n{full_raters[0]} rates the reference set but has not returned it; "
              "cannot compute agreement yet.")
        return 0
    else:
        name_a = "original expert"
        side_a = load_reference(args.dataset)

    partners = [r for r in ratings if r != name_a]
    if not partners:
        print("\nOnly the reference rating is in; no second rating to compare against yet.")
        return 0

    for name_b in partners:
        merged = side_a.merge(to_long(ratings[name_b], "b"), on=["paper_id", "question_id"])
        if merged.empty:
            print(f"\nNo overlapping cells between {name_a} and {name_b}.")
            continue
        per_item = report_agreement(merged, name_a, name_b)

        disagreements = merged[merged["a"] != merged["b"]].copy()
        disagreements["item"] = disagreements["question_id"] + 1
        disagreements["pdf"] = disagreements["paper_id"].map(
            lambda p: f"src/data/{args.dataset}/{p}.pdf"
        )
        out = ge.LABEL_DIR / f"{args.dataset}_v2_disagreements_{name_b}.csv"
        disagreements.sort_values(["question_id", "paper_id"])[
            ["paper_id", "pdf", "item", "a", "b"]
        ].rename(columns={"a": name_a, "b": name_b}).to_csv(out, index=False)
        print(f"\n{len(disagreements)} disagreements -> {out.name}")

        compare_to_models(args.dataset, per_item)

    ambiguity_tally(args.dataset, list(ratings))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
