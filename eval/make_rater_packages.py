"""Build the rating packages for the GRoLTS v2 double-coding (task A1 in REVISION.md).

Draws the double-coding subsets with a fixed seed, records them in a manifest, and writes one
zip per rater containing the PDFs, a pre-formatted Excel workbook and the codebook::

    uv run --group eval python eval/make_rater_packages.py

Allocation
----------
* **Rater 1** rates all 38 PTSD studies. These become the reference standard for PTSD v2,
  replacing the v1 labels that are currently mapped onto the revised items.
* **Rater 2** rates a random half of PTSD, and **rater 3** a random half of Achievement and of
  Delinquency. These exist only to estimate human-human agreement -- the benchmark both reviewers
  asked for. The existing Achievement and Delinquency labels stay the reference standard.

Sampling is simple random with a fixed seed. Stratifying by total reporting score was tested and
did not reduce the number of items left with no variance in the subset (2.7 vs 2.8 for
Achievement, 3.9 vs 3.6 for Delinquency): that is driven by individual items sitting at extreme
prevalence, which stratifying on totals cannot fix. The manifest makes the draw reproducible, and
the representativeness of each subset is reported rather than engineered.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import grolts_eval as ge  # noqa: E402
import grolts_questions as gq  # noqa: E402

SEED = 20260914
PDF_ROOT = Path(__file__).resolve().parents[1] / "src" / "data"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "packages"
MANIFEST = ge.LABEL_DIR / "rater_assignments.json"

QUESTIONS = gq.p5  # the revised checklist, 0-based ids, as prompted to the models
N_ITEMS = len(QUESTIONS)

#: Columns before the item block. They sit inside the frozen pane so a rater can still see the
#: paper id and their own notes while scrolling right through the item columns.
LEAD_COLUMNS = ["paper_id", "Ambiguous items", "Notes"]
FIRST_ITEM_COL = len(LEAD_COLUMNS) + 1  # 1-based; column D

#: Item columns carry the full question text, so they are wide and the header row is tall. A
#: smaller header font fits more words per line and keeps the row to a workable height.
ITEM_COL_WIDTH = 24
HEADER_FONT_SIZE = 10
#: Rendered line height is roughly a third more than the point size.
HEADER_LINE_RATIO = 1.33

HEADER_FILL = PatternFill("solid", fgColor="DDE5F0")
BLANK_FILL = PatternFill("solid", fgColor="FFF2CC")


def header_row_height() -> float:
    """Tall enough for the longest question at the chosen width, so no header is clipped."""
    chars_per_line = ITEM_COL_WIDTH * 11 / HEADER_FONT_SIZE
    longest = max(len(f"{i + 1}. {text}") for i, text in QUESTIONS.items())
    lines = math.ceil(longest / chars_per_line)
    return round(lines * HEADER_FONT_SIZE * HEADER_LINE_RATIO + 6, 1)


@dataclass(frozen=True)
class Assignment:
    """One rater's work on one dataset."""

    rater: str
    dataset: str
    fraction: float | None  # None = the whole dataset


ASSIGNMENTS = [
    Assignment("rater1", "ptsd", None),
    Assignment("rater2", "ptsd", 0.5),
    Assignment("rater3", "achievement", 0.5),
    Assignment("rater3", "delinquency", 0.5),
]


# --------------------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------------------


def paper_ids(dataset: str) -> list[int]:
    """Every paper id in a dataset, as defined by its human label file."""
    return sorted(ge.load_human_labels(dataset)["paper_id"].unique().tolist())


def total_scores(dataset: str) -> pd.Series:
    """Total checklist score per paper, used only to report subset representativeness."""
    labels = ge.load_human_labels(dataset)
    return labels.groupby("paper_id")["answer"].sum()


def draw(dataset: str, fraction: float | None, rng: np.random.Generator) -> list[int]:
    ids = paper_ids(dataset)
    if fraction is None:
        return ids
    k = int(len(ids) * fraction)
    return sorted(rng.choice(ids, size=k, replace=False).tolist())


def report_representativeness(dataset: str, subset: list[int]) -> dict:
    """Compare the drawn subset's reporting-quality distribution against the full set."""
    scores = total_scores(dataset)
    full, part = scores, scores.loc[subset]
    summary = {
        "n_full": int(len(full)),
        "n_subset": int(len(part)),
        "mean_total_full": round(float(full.mean()), 2),
        "mean_total_subset": round(float(part.mean()), 2),
        "sd_total_full": round(float(full.std()), 2),
        "sd_total_subset": round(float(part.std()), 2),
        "range_subset": [int(part.min()), int(part.max())],
        "range_full": [int(full.min()), int(full.max())],
    }
    print(
        f"    total score: full mean {summary['mean_total_full']} "
        f"(SD {summary['sd_total_full']}, range {summary['range_full']}) | "
        f"subset mean {summary['mean_total_subset']} "
        f"(SD {summary['sd_total_subset']}, range {summary['range_subset']})"
    )
    return summary


# --------------------------------------------------------------------------------------
# Codebook
# --------------------------------------------------------------------------------------


#: The codebook as structured blocks, so the Word file the raters receive and the Markdown copy
#: archived in the repository cannot drift apart. Inline ``**bold**`` and ``\`code\``` are honoured
#: by both renderers.
def codebook_blocks() -> list[tuple[str, str]]:
    """The rating instructions, with item text generated from the prompted questions."""
    blocks: list[tuple[str, str]] = [
        ("title", "Codebook — rating studies against the revised GRoLTS checklist"),
        ("h1", "What to do"),
        ("number", "Open the workbook in your package. One row per study; `paper_id` matches the "
                   "file `papers/<paper_id>.pdf` (or `papers/<dataset>/<paper_id>.pdf` if you "
                   "have two datasets)."),
        ("number", "Score every item **1** if the criterion is reported, **0** if it is not. "
                   "The cells only accept 0 or 1. Leave nothing blank."),
        ("number", "Use the two columns before the items as you go — they stay visible while you "
                   "scroll. **Ambiguous items**: the item numbers you found genuinely unclear for "
                   "this study, e.g. `7, 13`. **Notes**: why, in your own words. These are data, "
                   "not admin — which items trained raters find ambiguous is one of the results."),
        ("number", "Work **blind**: do not look at another rater's sheet, at any existing ratings "
                   "of these studies, or at any model output, until your sheet is complete. The "
                   "whole point is an independent estimate of how far two people agree, and it is "
                   "void if the ratings are anchored on an existing set."),
        ("number", "Send back the workbook. You do not need to rename it."),
        ("h1", "Scoring conventions"),
        ("p", "These follow the instructions the language models were given, so that the human "
              "and model judgements answer the same question:"),
        ("bullet", "Judge **reporting, not quality**. The question is whether the information is "
                   "present, not whether the choice it describes was a good one."),
        ("bullet", "Treat references to supplements, appendices, data repositories or URLs "
                   "mentioned in the text as valid and available — score 1 without chasing the "
                   "link."),
        ("bullet", "Score 1 only on **explicit evidence**. If the information is missing, "
                   "unclear, or merely implied by common practice in the field, score 0."),
        ("bullet", "Where an item says **“all the models tested”** it means every model "
                   "that was fitted, not just the selected one; where it says **“the final "
                   "model”**, only the selected solution matters. Items 11 and 12, and items "
                   "13 and 14, are deliberately separated on exactly this distinction. Read those "
                   "four carefully — it was the largest single source of disagreement in the "
                   "original checklist."),
        ("bullet", "If covariates or predictors were not used at all, item 8 is scored 0."),
        ("h1", "Items"),
    ]
    for i, text in QUESTIONS.items():
        column = get_column_letter(FIRST_ITEM_COL + i)
        blocks.append(("h2", f"Item {i + 1}  ·  column {column}"))
        blocks.append(("quote", text))
    return blocks


def _inline_runs(text: str) -> list[tuple[str, str]]:
    r"""Split a string into (style, chunk) runs on ``**bold**`` and ``\`code\```."""
    runs, buffer, index = [], "", 0
    while index < len(text):
        if text.startswith("**", index) and "**" in text[index + 2:]:
            close = text.index("**", index + 2)
            if buffer:
                runs.append(("plain", buffer))
                buffer = ""
            runs.append(("bold", text[index + 2:close]))
            index = close + 2
        elif text[index] == "`" and "`" in text[index + 1:]:
            close = text.index("`", index + 1)
            if buffer:
                runs.append(("plain", buffer))
                buffer = ""
            runs.append(("code", text[index + 1:close]))
            index = close + 1
        else:
            buffer += text[index]
            index += 1
    if buffer:
        runs.append(("plain", buffer))
    return runs


def codebook_markdown() -> str:
    """Render the codebook as Markdown, for the archived copy in the repository."""
    prefixes = {"title": "# ", "h1": "## ", "h2": "### ", "p": "", "bullet": "- ",
                "number": "1. ", "quote": "> "}
    return "\n\n".join(prefixes[kind] + text for kind, text in codebook_blocks()) + "\n"


def codebook_docx(path: Path) -> None:
    """Render the codebook as a Word document — what the raters actually receive.

    Methodologists should not need a Markdown viewer to read their instructions.
    """
    document = Document()
    styles = {"title": "Title", "h1": "Heading 1", "h2": "Heading 3", "p": "Body Text",
              "bullet": "List Bullet", "number": "List Number", "quote": "Intense Quote"}

    for kind, text in codebook_blocks():
        paragraph = document.add_paragraph(style=styles[kind])
        for run_style, chunk in _inline_runs(text):
            run = paragraph.add_run(chunk)
            if run_style == "bold":
                run.bold = True
            elif run_style == "code":
                run.font.name = "Consolas"
                run.font.size = Pt(10)

    document.save(path)


# --------------------------------------------------------------------------------------
# Workbook
# --------------------------------------------------------------------------------------


def add_rating_sheet(wb: Workbook, dataset: str, subset: list[int], first: bool) -> None:
    """Add one dataset's rating grid to the workbook."""
    title = dataset.capitalize()
    ws = wb.active if first else wb.create_sheet()
    ws.title = title

    # The question itself goes in the header cell rather than in a hover comment: fewer items
    # fit on screen, but the rater can read the criterion they are scoring without leaving the
    # keyboard. Row 1 is frozen, so the height is paid for once.
    ws.append(LEAD_COLUMNS + [f"{i + 1}. {text}" for i, text in QUESTIONS.items()])
    for cell in ws[1][: len(LEAD_COLUMNS)]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for cell in ws[1][len(LEAD_COLUMNS) :]:
        cell.font = Font(bold=True, size=HEADER_FONT_SIZE)
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)

    for paper_id in subset:
        ws.append([paper_id, None, None] + [None] * N_ITEMS)

    last_row = ws.max_row
    last_col = FIRST_ITEM_COL + N_ITEMS - 1
    item_range = (
        f"{get_column_letter(FIRST_ITEM_COL)}2:{get_column_letter(last_col)}{last_row}"
    )

    # Only 0 and 1 are accepted. This is what makes ingestion trustworthy.
    validation = DataValidation(
        type="whole",
        operator="between",
        formula1=0,
        formula2=1,
        allow_blank=True,
        showErrorMessage=True,
        errorTitle="Score must be 0 or 1",
        error="Enter 1 if the criterion is reported, 0 if it is not.",
    )
    ws.add_data_validation(validation)
    validation.add(item_range)

    # Unfilled cells stand out, so progress is visible at a glance.
    ws.conditional_formatting.add(
        item_range,
        FormulaRule(
            formula=[f"ISBLANK({get_column_letter(FIRST_ITEM_COL)}2)"], fill=BLANK_FILL
        ),
    )

    ws.freeze_panes = f"{get_column_letter(FIRST_ITEM_COL)}2"
    ws.column_dimensions["A"].width = 9
    ws.column_dimensions["B"].width = 18
    ws.column_dimensions["C"].width = 60
    for i in range(N_ITEMS):
        ws.column_dimensions[get_column_letter(FIRST_ITEM_COL + i)].width = ITEM_COL_WIDTH

    for row in ws.iter_rows(min_row=2, min_col=2, max_col=3):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2, min_col=FIRST_ITEM_COL, max_col=last_col):
        for cell in row:
            cell.alignment = Alignment(horizontal="center")

    ws.row_dimensions[1].height = header_row_height()
    ws.auto_filter.ref = f"A1:{get_column_letter(last_col)}{last_row}"


def add_codebook_sheet(wb: Workbook) -> None:
    ws = wb.create_sheet("Codebook")
    ws.append(["Item", "Question"])
    for cell in ws[1]:
        cell.font = Font(bold=True)
        cell.fill = HEADER_FILL
    for i, text in QUESTIONS.items():
        ws.append([i + 1, text])
    ws.column_dimensions["A"].width = 7
    ws.column_dimensions["B"].width = 120
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def build_workbook(path: Path, sheets: list[tuple[str, list[int]]]) -> None:
    wb = Workbook()
    for index, (dataset, subset) in enumerate(sheets):
        add_rating_sheet(wb, dataset, subset, first=index == 0)
    add_codebook_sheet(wb)
    wb.save(path)


# --------------------------------------------------------------------------------------
# Packaging
# --------------------------------------------------------------------------------------


def build_package(rater: str, sheets: list[tuple[str, list[int]]], staging: Path) -> Path:
    """Assemble one rater's folder and zip it."""
    root = staging / rater
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    multi = len(sheets) > 1
    for dataset, subset in sheets:
        target = root / "papers" / dataset if multi else root / "papers"
        target.mkdir(parents=True, exist_ok=True)
        for paper_id in subset:
            source = PDF_ROOT / dataset / f"{paper_id}.pdf"
            if not source.exists():
                raise FileNotFoundError(source)
            shutil.copy2(source, target / f"{paper_id}.pdf")

    build_workbook(root / f"{rater}_grolts_v2.xlsx", sheets)
    codebook_docx(root / "Codebook.docx")

    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    archive = PACKAGE_DIR / f"{rater}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(root))
    return archive


def main() -> int:
    rng = np.random.default_rng(SEED)

    manifest: dict = {"seed": SEED, "raters": {}}
    by_rater: dict[str, list[tuple[str, list[int]]]] = {}

    print("Drawing subsets\n" + "=" * 70)
    for assignment in ASSIGNMENTS:
        subset = draw(assignment.dataset, assignment.fraction, rng)
        share = "all" if assignment.fraction is None else f"{assignment.fraction:.0%}"
        print(f"  {assignment.rater} · {assignment.dataset}: {len(subset)} papers ({share})")
        stats = report_representativeness(assignment.dataset, subset)

        manifest["raters"].setdefault(assignment.rater, {})[assignment.dataset] = {
            "paper_ids": subset,
            "fraction": assignment.fraction,
            "representativeness": stats,
        }
        by_rater.setdefault(assignment.rater, []).append((assignment.dataset, subset))

    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"\nmanifest -> {MANIFEST}")

    # Keep one canonical copy in the repo as well as the copy inside each package, so the
    # instructions the raters worked from stay citable after the zips are gone.
    codebook = ge.LABEL_DIR / "codebook_v2.md"
    codebook.write_text(codebook_markdown())
    print(f"codebook -> {codebook}")

    print("\nBuilding packages\n" + "=" * 70)
    staging = PACKAGE_DIR / "_staging"
    for rater, sheets in by_rater.items():
        archive = build_package(rater, sheets, staging)
        size_mb = archive.stat().st_size / 1e6
        datasets = ", ".join(f"{d} ({len(s)})" for d, s in sheets)
        print(f"  {archive.name:<14} {datasets:<36} {size_mb:6.1f} MB")
    shutil.rmtree(staging, ignore_errors=True)

    print(f"\nDone. Send the zips in {PACKAGE_DIR}; they are gitignored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
