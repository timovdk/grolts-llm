"""Shared analysis utilities for the GRoLTS human-LLM evaluation.

Extracted from ``results_figures.ipynb`` so that the chunk-1000 and chunk-500 analyses
are driven by the same code instead of two notebooks that differ only in filename
literals. Every entry point takes ``chunk`` as an argument.

Install the analysis-only dependencies with::

    uv sync --only-group eval

Checklist versions
------------------
``qset=0`` is the original GRoLTS checklist (v1, 21 items) and ``qset=4`` is the revised
checklist (v2, 19 items); both are defined in ``src/grolts_questions.py``. Human labels
are stored per dataset in ``eval/human_labels`` with 0-based ``question_id`` columns, so a
``question_id`` of 12 is item 13 as printed in the manuscript appendices.

The PTSD studies were rated by a human against v1 only, so comparing them against v2 model
answers requires mapping between the two item sets:

* ``ID_MAP`` maps a v2 item onto the v1 item it came from. Used to score v2 model answers
  against v1 human labels while keeping v2 item ids for reporting.
* ``INVERSE_ID_MAP`` maps a v1 item onto the v2 item(s) it was split into. Used to expand
  v1 human labels into v2 item space.

Because v1 items 13 and 14 were each split into two v2 items, the mapped PTSD human labels
contain *duplicated* values: v2 items 10/11 both carry the v1 item 13 label, and v2 items
12/13 both carry the v1 item 14 label. See ``DUPLICATED_BY_MAPPING``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = EVAL_DIR / "outputs"
LABEL_DIR = EVAL_DIR / "human_labels"
VIZ_DIR = EVAL_DIR / "viz"

# --------------------------------------------------------------------------------------
# Experiment grid
# --------------------------------------------------------------------------------------

EMBEDDER = "Qwen_Qwen3-Embedding-8B"

#: Model file slugs, in the canonical reporting order used by ``MODEL_LABELS``.
MODEL_SLUGS: tuple[str, ...] = (
    "gpt-5-mini",
    "Qwen_Qwen3-30B-A3B-Instruct-2507",
    "meta-llama_Llama-3.3-70B-Instruct",
    "mistralai_Magistral-Small-2509",
    "Qwen_Qwen3-Next-80B-A3B-Instruct",
)

#: Display names, index-aligned with ``MODEL_SLUGS``.
MODEL_LABELS: list[str] = [
    "gpt-5-mini",
    "Qwen-3-30B",
    "Llama-3.3-70B",
    "Magistral-Small",
    "Qwen-Next-80B",
]

DATASETS: tuple[str, ...] = ("achievement", "delinquency", "ptsd")

QSET_V1 = 0
QSET_V2 = 4

# --------------------------------------------------------------------------------------
# Checklist item bookkeeping
# --------------------------------------------------------------------------------------

#: v2 item -> the v1 item it was derived from.
ID_MAP: dict[int, int] = {
    0: 0, 1: 1, 2: 4, 3: 5, 4: 6, 5: 7, 6: 9, 7: 10, 8: 11, 9: 12,
    10: 13, 11: 13, 12: 14, 13: 14, 14: 15, 15: 16, 16: 18, 17: 19, 18: 20,
}

#: v1 item -> the v2 item(s) it was split into. v1 items 2, 3, 8 and 17 were dropped in v2.
INVERSE_ID_MAP: dict[int, list[int]] = {
    0: [0], 1: [1], 4: [2], 5: [3], 6: [4], 7: [5], 9: [6], 10: [7], 11: [8],
    12: [9], 13: [10, 11], 14: [12, 13], 15: [14], 16: [15], 18: [16], 19: [17], 20: [18],
}

#: v2 item pairs that share a single v1 human label under ``INVERSE_ID_MAP``.
DUPLICATED_BY_MAPPING: tuple[tuple[int, int], ...] = ((10, 11), (12, 13))

#: v1 items with no v2 counterpart (missing-data mechanism, attrition variables,
#: between-class var-cov structure, per-model trajectory plots).
V1_ONLY_ITEMS: tuple[int, ...] = (2, 3, 8, 17)

GROLTS_LABELS_OLD: list[str] = [
    "1", "2", "3a", "3b", "3c", "4", "5", "6a", "6b", "7", "8",
    "9", "10", "11", "12", "13", "14a", "14b", "14c", "15", "16",
]
GROLTS_LABELS_NEW: list[str] = [str(i) for i in range(1, 20)]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def output_files(
    dataset: str,
    *,
    chunk: int = 1000,
    qset: int = QSET_V2,
    run: int | None = None,
    path: Path | str | None = None,
) -> list[Path]:
    """Return the five parsed model output CSVs for one dataset/chunk/checklist version.

    Ordered to match :data:`MODEL_LABELS`. ``run`` selects one of the sampled reruns
    written by ``generate_responses.py --runs``; omit it for the original greedy run.
    """
    base = Path(path) if path is not None else OUTPUT_DIR
    suffix = "" if run is None else f"_run{run}"
    return [
        base / f"{EMBEDDER}_{slug}_{dataset}_{chunk}_{qset}{suffix}.csv"
        for slug in MODEL_SLUGS
    ]


def load_human_labels(dataset: str, *, path: Path | str | None = None) -> pd.DataFrame:
    """Load one dataset's human labels as long ``[paper_id, question_id, answer]``."""
    base = Path(path) if path is not None else LABEL_DIR
    df = pd.read_csv(base / f"{dataset}.csv", delimiter=";", dtype=int)
    df = df.melt(id_vars=["paper_id"], var_name="question_id", value_name="answer")
    return df.astype({"paper_id": int, "question_id": int, "answer": int})


def load_llm_answers(file: Path | str) -> pd.DataFrame:
    """Load one parsed model output CSV.

    A small number of responses could not be parsed into YES/NO and carry ``NaN``
    answers; they are dropped here rather than silently later, and the count is logged.
    """
    df = pd.read_csv(file)
    n_missing = int(df["answer"].isna().sum())
    if n_missing:
        logger.warning(
            "%s: dropping %d/%d unparsed answers", Path(file).name, n_missing, len(df)
        )
        df = df.dropna(subset=["answer"])
    df["answer"] = df["answer"].astype(int)
    return df


def exclude_questions(df: pd.DataFrame, items: Iterable[int] | None) -> pd.DataFrame:
    """Drop rows whose ``question_id`` is in ``items``. No-op when ``items`` is falsy."""
    if not items:
        return df
    return df[~df["question_id"].isin(list(items))]


# --------------------------------------------------------------------------------------
# Per-item agreement
# --------------------------------------------------------------------------------------


def load_llm_accuracies(
    df_labels: pd.DataFrame,
    files: Sequence[Path | str],
    mapping: Mapping[int, int] | None = None,
    *,
    labels: Sequence[str] = MODEL_LABELS,
) -> pd.DataFrame:
    """Per-item agreement between each model and the human labels.

    ``mapping`` is the forward :data:`ID_MAP` (v2 item -> v1 item), used when the model
    answered v2 items but the human labels are v1. Reporting stays in v2 item ids.

    Returns a frame with a ``question_id`` column and one column per model.
    """
    data = {}
    for label, file in zip(labels, files):
        df = load_llm_answers(file)

        if mapping:
            df["question_id_file"] = df["question_id"]
            df["question_id"] = df["question_id"].map(mapping)
            merged = df.merge(
                df_labels, on=["paper_id", "question_id"], suffixes=("_pred", "_true")
            )
            merged["question_id"] = merged["question_id_file"]
            merged = merged.drop(columns=["question_id_file"])
        else:
            merged = df.merge(
                df_labels, on=["paper_id", "question_id"], suffixes=("_pred", "_true")
            )

        merged["correct"] = (merged["answer_pred"] == merged["answer_true"]).astype(int)
        data[label] = merged.groupby("question_id")["correct"].mean()

    return pd.DataFrame(data).sort_index().reset_index()


def human_prevalence(
    df_labels: pd.DataFrame, mapping: Mapping[int, int] | None = None
) -> pd.Series:
    """Proportion of "yes" human labels per item, in v2 item space when ``mapping`` given."""
    prevalence = df_labels.groupby("question_id")["answer"].mean()
    if mapping:
        return pd.Series({new: prevalence[old] for new, old in mapping.items()})
    return prevalence


def add_human_prevalence(
    acc_df: pd.DataFrame,
    df_labels: pd.DataFrame,
    mapping: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Attach a ``Human`` column of per-item human "yes" prevalence to an accuracy frame."""
    acc_df = acc_df.copy()
    acc_df["Human"] = acc_df["question_id"].map(human_prevalence(df_labels, mapping))
    return acc_df


# --------------------------------------------------------------------------------------
# Rater tables and chance-corrected agreement
# --------------------------------------------------------------------------------------


def build_rater_table(
    df_labels: pd.DataFrame,
    files: Sequence[Path | str],
    mapping: Mapping[int, list[int]] | None = None,
    *,
    labels: Sequence[str] = MODEL_LABELS,
) -> pd.DataFrame:
    """One row per (paper, item) with a ``human`` column and one column per model.

    ``mapping`` is the :data:`INVERSE_ID_MAP` (v1 item -> v2 item(s)), used to expand v1
    human labels into v2 item space. Items produced by a 1-to-2 split share the same human
    label; see :data:`DUPLICATED_BY_MAPPING`.
    """
    df = df_labels.rename(columns={"answer": "human"})

    if mapping:
        # NB: assign() per target id -- mutating a shared frame in place would give every
        # copy the last id, leaving the other split item without a human label.
        df = pd.concat(
            [
                df[df["question_id"] == old_id].assign(question_id=new_id)
                for old_id, new_ids in mapping.items()
                for new_id in new_ids
            ],
            ignore_index=True,
        )

    merged_labels = []
    for label, file in zip(labels, files):
        df2 = load_llm_answers(file).rename(columns={"answer": label})
        df = df.merge(
            df2[["paper_id", "question_id", label]],
            on=["paper_id", "question_id"],
            how="left",
        )
        merged_labels.append(label)

    # `files` may be empty, to get the human labels alone in v2 item space.
    if merged_labels:
        n_missing = int(df[merged_labels].isna().any(axis=1).sum())
        if n_missing:
            logger.warning(
                "rater table: %d/%d rows lack a model answer", n_missing, len(df)
            )

    return df


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Fleiss' kappa from an ``n_items x n_categories`` matrix of rater counts."""
    N, _ = ratings_matrix.shape
    n_per_item = ratings_matrix.sum(axis=1)
    if not np.all(n_per_item == n_per_item[0]):
        raise ValueError("All items must have the same number of raters")
    n = n_per_item[0]

    p = ratings_matrix.sum(axis=0) / (N * n)
    P = (ratings_matrix * (ratings_matrix - 1)).sum(axis=1) / (n * (n - 1))

    P_e = float(np.sum(p**2))
    if P_e == 1:
        return float("nan")
    return float((P.mean() - P_e) / (1 - P_e))


def compute_fleiss_kappa(
    df_raters: pd.DataFrame, raters: Sequence[str] | None = None
) -> float:
    """Fleiss' kappa over the given rater columns, pooled across all (paper, item) rows.

    Defaults to the human plus every model. Pass ``raters=MODEL_LABELS`` for the
    LLM-only figure, which separates model consensus from human-model agreement.
    """
    if raters is None:
        raters = (["human"] if "human" in df_raters.columns else []) + list(MODEL_LABELS)
    values = df_raters[list(raters)].to_numpy()
    counts_1 = values.sum(axis=1)
    return fleiss_kappa(np.column_stack([len(raters) - counts_1, counts_1]))


def cohen_kappa_vs_human(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> dict[str, float]:
    """Cohen's kappa between the human and each model, pooled over all (paper, item) rows.

    Pairwise deletion: each model is compared on the rows where it and the human both
    answered, so one model's unparsed answers do not shrink another model's sample.
    """
    results: dict[str, float] = {}
    for label in labels:
        pair = df_raters[["human", label]].dropna()
        results[label] = float(cohen_kappa_score(pair["human"], pair[label]))
    return results


# --------------------------------------------------------------------------------------
# Total scores and rank order
# --------------------------------------------------------------------------------------


def total_scores(
    df_raters: pd.DataFrame, raters: Sequence[str] | None = None
) -> pd.DataFrame:
    """Sum each rater's "yes" answers per paper, over rows where those raters answered.

    Summing straight through would score an unparsed model answer as 0 (pandas sums with
    ``skipna=True``), so rows missing an answer for any requested rater are dropped first.
    Pass a single model in ``raters`` to get that model's score without letting another
    model's unparsed answers remove items from it.
    """
    if raters is None:
        raters = [c for c in df_raters.columns if c not in ("paper_id", "question_id")]
    raters = list(raters)
    return df_raters.dropna(subset=raters).groupby("paper_id")[raters].sum()


def spearman_vs_human(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> dict[str, float]:
    """Spearman rho between human and model total checklist scores, per model.

    Uses pairwise deletion: each model is scored over the (paper, item) cells where both
    that model and the human answered, and the human total is recomputed over the same
    cells. Listwise deletion would let one model's unparsed answers shift every other
    model's totals.
    """
    results: dict[str, float] = {}
    for label in labels:
        scores = total_scores(df_raters, ["human", label])
        results[label] = float(spearmanr(scores["human"], scores[label]).statistic)
    return results


def tercile_map(
    acc_df: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> dict[int, str]:
    """Split items into high/mid/low agreement terciles.

    Items are ranked by accuracy averaged over the models and split ``n // 3`` /
    ``n // 3`` / remainder -- a rank-based split with no fixed threshold. Note this
    assigns terciles using the same responses later used to compute rank correlations;
    use a held-out split for an out-of-sample estimate.
    """
    ranked = acc_df.assign(_mean=acc_df[list(labels)].mean(axis=1)).sort_values(
        "_mean", ascending=False
    )["question_id"]
    size = len(ranked) // 3
    return {
        **{q: "high" for q in ranked.iloc[:size]},
        **{q: "mid" for q in ranked.iloc[size : 2 * size]},
        **{q: "low" for q in ranked.iloc[2 * size :]},
    }


def tercile_rank_correlations(
    df_raters: pd.DataFrame, tercile_mapping: Mapping[int, str]
) -> dict[str, dict[str, float]]:
    """Spearman rho per agreement tercile, scoring each tercile's items separately."""
    results: dict[str, dict[str, float]] = {}
    for tercile in ("high", "mid", "low"):
        items = [q for q, t in tercile_mapping.items() if t == tercile]
        if not items:
            continue
        subset = df_raters[df_raters["question_id"].isin(items)]
        results[tercile] = spearman_vs_human(subset)
    return results


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------


def plot_accuracy_with_human(
    acc_df: pd.DataFrame,
    *,
    title: str,
    outfile: Path | str | None = None,
    labels: Sequence[str] = MODEL_LABELS,
    show: bool = True,
) -> plt.Figure:
    """Item x model accuracy heatmap with row/column mean strips and a human prevalence strip.

    ``acc_df`` must have a ``question_id`` column, one column per model, and ``Human``.
    """
    indexed = acc_df.set_index("question_id")
    model_acc = indexed[list(labels)]
    human = indexed["Human"]

    row_means = model_acc.mean(axis=1)
    col_means = model_acc.mean(axis=0)
    overall_mean = model_acc.to_numpy().mean()

    n_items = model_acc.shape[0]
    if n_items == len(GROLTS_LABELS_NEW):
        y_labels = GROLTS_LABELS_NEW
    elif n_items == len(GROLTS_LABELS_OLD):
        y_labels = [str(i + 1) for i in range(len(GROLTS_LABELS_OLD))]
    else:  # after an item exclusion the ids no longer line up with a full checklist
        y_labels = [str(q + 1) for q in model_acc.index]

    fig = plt.figure(figsize=(11, 7))
    gs = gridspec.GridSpec(
        2,
        6,
        width_ratios=[4, 0.25, 0.1, 0.35, 0.5, 0.1],
        height_ratios=[4, 0.25],
        wspace=0.05,
        hspace=0.05,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    ax_row = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_spacer = fig.add_subplot(gs[0, 3])
    ax_human = fig.add_subplot(gs[0, 4])
    ax_cbar_human = fig.add_subplot(gs[0, 5])
    ax_col = fig.add_subplot(gs[1, 0], sharex=ax_main)
    ax_corner = fig.add_subplot(gs[1, 1])
    corners = [fig.add_subplot(gs[1, i]) for i in (2, 3, 4, 5)]

    sns.heatmap(
        model_acc,
        ax=ax_main,
        vmin=0,
        vmax=1,
        cmap="rocket",
        cbar_ax=ax_cbar,
        cbar_kws={"label": "Accuracy"},
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        xticklabels=False,
    )
    ax_main.set_yticklabels(y_labels, rotation=0)
    ax_main.set_title(title)
    ax_main.set_xlabel("")
    ax_main.set_ylabel("Question ID")
    ax_main.tick_params(bottom=False, top=False, labelbottom=False, left=True, labelleft=True)

    sns.heatmap(
        row_means.to_frame(name="Mean"),
        ax=ax_row,
        cmap="rocket",
        vmin=0,
        vmax=1,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        yticklabels=False,
        xticklabels=False,
        linewidths=0.5,
    )
    ax_row.set_facecolor("white")
    ax_row.set_ylabel("")
    ax_row.set_xlabel("")
    ax_row.tick_params(left=False, right=False, labelleft=False)

    sns.heatmap(
        pd.DataFrame([col_means.values], columns=col_means.index, index=["Mean"]),
        ax=ax_col,
        cmap="rocket",
        vmin=0,
        vmax=1,
        cbar=False,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        yticklabels=False,
        linewidths=0.5,
    )
    ax_col.set_facecolor("white")
    ax_col.set_xticklabels(list(labels), rotation=30, ha="center")
    ax_col.set_ylabel("")
    ax_col.set_xlabel("")

    ax_corner.text(
        0.5, 0.5, f"{overall_mean:.2f}", ha="center", va="center",
        fontsize=11, fontweight="bold",
    )
    ax_corner.set_xticks([])
    ax_corner.set_yticks([])
    ax_corner.set_facecolor("#f0f0f0")
    for ax in (*corners, ax_spacer):
        ax.axis("off")

    sns.heatmap(
        human.to_frame(name="Human"),
        ax=ax_human,
        vmin=0,
        vmax=1,
        cmap=sns.light_palette("gray", as_cmap=True),
        annot=True,
        fmt=".2f",
        annot_kws={"size": 12},
        yticklabels=False,
        linewidths=0.5,
        cbar_kws={"label": "Proportion of 1's"},
        cbar_ax=ax_cbar_human,
    )
    ax_human.set_facecolor("white")
    ax_human.set_ylabel("")
    ax_human.set_xlabel("")
    ax_human.set_xticklabels([""], rotation=30, ha="center")

    if outfile is not None:
        os.makedirs(Path(outfile).parent, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def item_agreement_long(
    acc_dfs: Mapping[str, pd.DataFrame], *, labels: Sequence[str] = MODEL_LABELS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reshape per-dataset accuracy frames for the ranked-agreement figures.

    Returns ``(summary, long)``. ``summary`` has one row per (dataset, item) with the mean
    across models, its SD and a 95% CI; ``long`` has one row per (dataset, item, model).
    Both carry a ``rank`` column giving a single item ordering shared by all datasets,
    lowest mean agreement first.
    """
    long = pd.concat(
        [
            acc_df.melt(id_vars="question_id", value_vars=list(labels),
                        var_name="model", value_name="acc").assign(dataset=name)
            for name, acc_df in acc_dfs.items()
        ],
        ignore_index=True,
    )

    summary = (
        long.groupby(["dataset", "question_id"])["acc"]
        .agg(mean="mean", sd="std", n="count")
        .reset_index()
    )
    summary["ci"] = 1.96 * summary["sd"] / np.sqrt(summary["n"])

    order = summary.groupby("question_id")["mean"].mean().sort_values().index
    ranks = {q: i for i, q in enumerate(order)}
    summary["rank"] = summary["question_id"].map(ranks)
    long["rank"] = long["question_id"].map(ranks)

    return summary.sort_values(["dataset", "rank"]), long


def plot_ranked_agreement(
    summary: pd.DataFrame,
    long: pd.DataFrame,
    *,
    title: str,
    outfile: Path | str | None = None,
    labels: Sequence[str] = MODEL_LABELS,
    show: bool = True,
) -> plt.Figure:
    """Items ranked by mean human-model agreement, with per-model points.

    Individual model scores are colour- and marker-coded rather than drawn in a single
    grey, so a reader can tell which model sits where (reviewer 2, Figure 2).
    """
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    fig, ax = plt.subplots(figsize=(9, 4.5))

    order = summary.sort_values("rank")["question_id"].unique()
    datasets = list(summary["dataset"].unique())

    markers = ["o", "s", "^", "D", "v"]
    palette = dict(zip(labels, sns.color_palette("colorblind", len(labels))))
    for offset, (model, marker) in enumerate(zip(labels, markers)):
        sub = long[long["model"] == model]
        jitter = (offset - (len(labels) - 1) / 2) * 0.11
        ax.scatter(
            sub["rank"] + jitter, sub["acc"], s=22, marker=marker,
            color=palette[model], alpha=0.75, linewidths=0, label=model, zorder=2,
        )

    dataset_offsets = np.linspace(-0.28, 0.28, len(datasets)) if len(datasets) > 1 else [0.0]
    for offset, dataset in zip(dataset_offsets, datasets):
        sub = summary[summary["dataset"] == dataset].sort_values("rank")
        x = sub["rank"].to_numpy() + offset
        ax.errorbar(
            x, sub["mean"], yerr=sub["ci"], fmt="_", markersize=14, markeredgewidth=2.5,
            color="black" if len(datasets) == 1 else None, ecolor="gray",
            elinewidth=1.2, capsize=2, zorder=3, label=f"Mean — {dataset}",
        )

    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Question rank (low → high agreement)")
    ax.set_ylabel("Mean human–LLM agreement")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([q + 1 for q in order])
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="lower right")
    sns.despine()
    plt.tight_layout()

    if outfile is not None:
        os.makedirs(Path(outfile).parent, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_tercile_rank_correlations(
    all_tercile: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    outfile: Path | str | None = None,
    labels: Sequence[str] = MODEL_LABELS,
    show: bool = True,
) -> plt.Figure:
    """Grouped bars of Spearman rho per model, split by item-agreement tercile."""
    sns.set_theme(style="white", context="paper", font_scale=1.2)
    datasets = list(all_tercile)
    terciles = ("high", "mid", "low")
    colors = {"high": "#1f77b4", "mid": "#ff7f0e", "low": "#2ca02c"}

    nrows = int(np.ceil(len(datasets) / 2))
    fig, axes = plt.subplots(nrows, 2, figsize=(9, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    width = 0.22
    x = np.arange(len(labels))

    for ax, dataset in zip(axes, datasets):
        for i, tercile in enumerate(terciles):
            values = [all_tercile[dataset].get(tercile, {}).get(m, np.nan) for m in labels]
            ax.bar(x + i * width - width, values, width=width,
                   color=colors[tercile], alpha=0.85, label=tercile.capitalize())
        ax.set_xticks(x)
        ax.set_xticklabels(list(labels), rotation=45, ha="right")
        ax.set_title(dataset)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(-0.3, 1)
    for ax in axes[len(datasets):]:
        ax.axis("off")
    for ax in axes[::2]:
        ax.set_ylabel("Spearman ρ")

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles[:3], legend_labels[:3], title="Agreement level",
               frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    sns.despine()
    fig.suptitle(
        "Spearman rank correlations between human and LLM scores by item agreement level",
        y=1.02,
    )
    plt.tight_layout()

    if outfile is not None:
        os.makedirs(Path(outfile).parent, exist_ok=True)
        plt.savefig(outfile, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def bootstrap_paper_ci(
    df: pd.DataFrame,
    statistic,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``statistic(df)``, resampling whole studies.

    Studies are the independent unit here, not (study, item) cells: the items within one
    paper are rated by the same person off the same text and are not independent draws.
    Returns ``(low, high)``, or ``(nan, nan)`` if the statistic is undefined throughout
    (e.g. an item every rater scored the same way).
    """
    rng = np.random.default_rng(seed)
    papers = df["paper_id"].unique()
    by_paper = {p: g for p, g in df.groupby("paper_id")}

    values = []
    for _ in range(n_boot):
        draw = rng.choice(papers, size=len(papers), replace=True)
        sample = pd.concat([by_paper[p] for p in draw], ignore_index=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            value = statistic(sample)
        if value is not None and np.isfinite(value):
            values.append(value)

    if not values:
        return (float("nan"), float("nan"))
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


# --------------------------------------------------------------------------------------
# Item-level agreement (reviewer 1, majors 5 and 6; minor 2)
# --------------------------------------------------------------------------------------


def item_accuracy_from_raters(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> pd.DataFrame:
    """Per-item agreement computed straight off a rater table.

    Equivalent to :func:`load_llm_accuracies` but usable on any subset of studies, which
    is what the split-half tercile analysis needs.
    """
    data = {}
    for label in labels:
        pair = df_raters[["question_id", "human", label]].dropna()
        data[label] = (
            pair.assign(correct=(pair["human"] == pair[label]).astype(int))
            .groupby("question_id")["correct"]
            .mean()
        )
    return pd.DataFrame(data).sort_index().reset_index()


def _kappa_2x2(human: np.ndarray, model: np.ndarray) -> np.ndarray:
    """Cohen's kappa for binary ratings, row-wise over a 2-D array of resamples.

    Closed form from the 2x2 table, so a whole bootstrap evaluates in one pass. Returns
    ``nan`` where expected agreement is 1 (both raters constant and identical).
    """
    n = human.shape[1]
    a = (human & model).sum(axis=1)
    b = (human & ~model).sum(axis=1)
    c = (~human & model).sum(axis=1)
    d = (~human & ~model).sum(axis=1)

    p_o = (a + d) / n
    p_e = ((a + b) * (a + c) + (c + d) * (b + d)) / n**2
    with np.errstate(invalid="ignore", divide="ignore"):
        kappa = np.where(p_e == 1, np.nan, (p_o - p_e) / (1 - p_e))
    return kappa


def per_item_kappa(
    df_raters: pd.DataFrame,
    *,
    labels: Sequence[str] = MODEL_LABELS,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Cohen's kappa between the human and each model, computed *within* each item.

    The pooled kappa reported in the manuscript mixes items of very different prevalence
    into one expected-agreement term. This reports one estimate per (item, model) with a
    bootstrap CI over studies, so item difficulty and rater agreement stay separable.

    ``kappa`` is undefined for an item where one rater never varies; ``accuracy`` and
    ``prevalence`` are still meaningful there and are reported alongside.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for label in labels:
        for qid, group in df_raters.groupby("question_id"):
            pair = group[["paper_id", "human", label]].dropna()
            if pair.empty:
                continue

            human = pair["human"].to_numpy(dtype=bool)
            model = pair[label].to_numpy(dtype=bool)
            kappa = _kappa_2x2(human[None, :], model[None, :])[0]

            if np.isfinite(kappa):
                # One item means one row per study, so the bootstrap is a resample of the
                # array itself -- done in one vectorised pass rather than n_boot regroups.
                draws = rng.integers(0, len(human), size=(n_boot, len(human)))
                boot = _kappa_2x2(human[draws], model[draws])
                boot = boot[np.isfinite(boot)]
                low, high = (
                    (float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975)))
                    if boot.size else (float("nan"), float("nan"))
                )
            else:
                low, high = float("nan"), float("nan")

            rows.append({
                "question_id": qid,
                "model": label,
                "n": len(pair),
                "accuracy": float((pair["human"] == pair[label]).mean()),
                "kappa": kappa,
                "ci_low": low,
                "ci_high": high,
                "human_prevalence": float(pair["human"].mean()),
                "model_prevalence": float(pair[label].mean()),
            })
    return pd.DataFrame(rows)


def pairwise_llm_kappa(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> pd.DataFrame:
    """Cohen's kappa between every pair of models, ignoring the human.

    Reviewer 1's point on Fleiss' kappa: consensus among the five models can carry the
    combined figure even where they all disagree with the human rater. This makes the
    model-model half of that visible on its own.
    """
    matrix = pd.DataFrame(index=list(labels), columns=list(labels), dtype=float)
    for a in labels:
        for b in labels:
            if a == b:
                matrix.loc[a, b] = 1.0
                continue
            pair = df_raters[[a, b]].dropna()
            matrix.loc[a, b] = float(cohen_kappa_score(pair[a], pair[b]))
    return matrix


def mean_pairwise_llm_kappa(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> float:
    """Mean Cohen's kappa over all distinct model pairs."""
    matrix = pairwise_llm_kappa(df_raters, labels=labels).to_numpy()
    return float(matrix[np.triu_indices_from(matrix, k=1)].mean())


def confusion_metrics(
    df_raters: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS, by_item: bool = True
) -> pd.DataFrame:
    """Confusion counts and prevalence-robust metrics, per item and model.

    Reviewer 1 minor 2. Raw accuracy and kappa are both hard to read on items where the
    human said "yes" to 3% or 97% of studies, so this reports the counts themselves plus
    sensitivity, specificity, balanced accuracy, MCC and PABAK. Treats a human "yes" as
    the positive class.
    """
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef

    rows = []
    groups = df_raters.groupby("question_id") if by_item else [(None, df_raters)]
    for label in labels:
        for qid, group in groups:
            pair = group[["human", label]].dropna()
            if pair.empty:
                continue
            human, model = pair["human"].to_numpy(), pair[label].to_numpy()
            tp = int(((human == 1) & (model == 1)).sum())
            fp = int(((human == 0) & (model == 1)).sum())
            fn = int(((human == 1) & (model == 0)).sum())
            tn = int(((human == 0) & (model == 0)).sum())
            accuracy = (tp + tn) / len(pair)
            rows.append({
                "question_id": qid,
                "model": label,
                "n": len(pair),
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
                "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
                "accuracy": accuracy,
                "balanced_accuracy": (
                    float(balanced_accuracy_score(human, model))
                    if len(np.unique(human)) > 1 else float("nan")
                ),
                "MCC": (
                    float(matthews_corrcoef(human, model))
                    if len(np.unique(human)) > 1 and len(np.unique(model)) > 1
                    else float("nan")
                ),
                # Prevalence- and bias-adjusted kappa: defined even at extreme prevalence.
                "PABAK": 2 * accuracy - 1,
                "human_prevalence": float(human.mean()),
            })
    out = pd.DataFrame(rows)
    return out.drop(columns=["question_id"]) if not by_item else out


# --------------------------------------------------------------------------------------
# Out-of-sample tercile analysis (reviewer 1, major 6)
# --------------------------------------------------------------------------------------


def tercile_transfer(
    rater_tables: Mapping[str, pd.DataFrame],
    *,
    labels: Sequence[str] = MODEL_LABELS,
) -> pd.DataFrame:
    """Define terciles on one dataset, evaluate rank correlations on another.

    Reviewer 1 is right that assigning terciles by human-model agreement and then
    rank-correlating the same responses is circular. Datasets sharing a checklist version
    share item ids, so a tercile assignment learned on one can be applied to another --
    the split is then independent of the responses it is evaluated on.
    """
    rows = []
    for source, source_table in rater_tables.items():
        mapping = tercile_map(item_accuracy_from_raters(source_table, labels=labels),
                              labels=labels)
        for target, target_table in rater_tables.items():
            shared = set(target_table["question_id"]) & set(mapping)
            if not shared:
                continue
            applied = {q: t for q, t in mapping.items() if q in shared}
            for tercile, per_model in tercile_rank_correlations(target_table, applied).items():
                for model, rho in per_model.items():
                    rows.append({
                        "terciles from": source,
                        "evaluated on": target,
                        "in sample": source == target,
                        "tercile": tercile,
                        "model": model,
                        "rho": rho,
                    })
    return pd.DataFrame(rows)


def split_half_tercile(
    df_raters: pd.DataFrame,
    *,
    labels: Sequence[str] = MODEL_LABELS,
    n_repeats: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign terciles on half the studies, evaluate rank correlations on the other half.

    A within-dataset version of :func:`tercile_transfer`. Averaged over ``n_repeats``
    random splits. The gap against the in-sample figure is the optimism introduced by
    choosing items and scoring them on the same data.
    """
    rng = np.random.default_rng(seed)
    papers = df_raters["paper_id"].unique()
    rows = []
    for _ in range(n_repeats):
        shuffled = rng.permutation(papers)
        fit, test = shuffled[: len(shuffled) // 2], shuffled[len(shuffled) // 2 :]
        mapping = tercile_map(
            item_accuracy_from_raters(
                df_raters[df_raters["paper_id"].isin(fit)], labels=labels
            ),
            labels=labels,
        )
        held_out = df_raters[df_raters["paper_id"].isin(test)]
        for tercile, per_model in tercile_rank_correlations(held_out, mapping).items():
            for model, rho in per_model.items():
                rows.append({"tercile": tercile, "model": model, "rho": rho})

    return (
        pd.DataFrame(rows)
        .groupby(["tercile", "model"])["rho"]
        .agg(mean="mean", sd="std")
        .reset_index()
    )


# --------------------------------------------------------------------------------------
# Is the total score a sensible scale? (reviewer 1, major 6)
# --------------------------------------------------------------------------------------


def scale_diagnostics(df_labels: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """KR-20 and corrected item-total correlations for one rater's labels.

    The manuscript sums 19-21 binary items into a "reporting quality" score. Reviewer 1
    asks what licenses treating that as one dimension. Returns ``(kr20, per_item)`` where
    ``per_item`` holds each item's prevalence and its correlation with the total of the
    *other* items.
    """
    wide = df_labels.pivot(index="paper_id", columns="question_id", values="answer")
    n_items = wide.shape[1]
    p = wide.mean()
    variance = wide.sum(axis=1).var(ddof=1)
    kr20 = (
        float((n_items / (n_items - 1)) * (1 - (p * (1 - p)).sum() / variance))
        if variance > 0 else float("nan")
    )

    rows = []
    for item in wide.columns:
        rest = wide.drop(columns=[item]).sum(axis=1)
        # An item every study answers the same way has no variance to correlate.
        constant = wide[item].nunique() < 2
        rows.append({
            "question_id": item,
            "prevalence": float(wide[item].mean()),
            "item_total_r": float("nan") if constant else float(wide[item].corr(rest)),
        })
    return kr20, pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Run-to-run stability (reviewer 2, major 2)
# --------------------------------------------------------------------------------------


def available_runs(
    dataset: str, *, chunk: int = 1000, qset: int = QSET_V2, path: Path | str | None = None
) -> list[int]:
    """Which sampled reruns are on disk for every model in this configuration."""
    base = Path(path) if path is not None else OUTPUT_DIR
    complete = []
    for run in range(1, 100):
        files = output_files(dataset, chunk=chunk, qset=qset, run=run, path=base)
        if all(f.exists() for f in files):
            complete.append(run)
        elif any(f.exists() for f in files):
            logger.warning("run %d is present for only some models; skipping", run)
    return complete


def load_runs(
    dataset: str,
    runs: Sequence[int],
    *,
    chunk: int = 1000,
    qset: int = QSET_V2,
    labels: Sequence[str] = MODEL_LABELS,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Long frame of every sampled answer: paper, item, model, run, answer."""
    frames = []
    for run in runs:
        files = output_files(dataset, chunk=chunk, qset=qset, run=run, path=path)
        for label, file in zip(labels, files):
            frames.append(
                load_llm_answers(file)[["paper_id", "question_id", "answer"]]
                .assign(model=label, run=run)
            )
    return pd.concat(frames, ignore_index=True)


def response_stability(
    df_runs: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> pd.DataFrame:
    """How often does the same prompt get a different answer across runs?

    Per model: the share of (study, item) cells whose answer is not unanimous across
    runs, and the mean agreement with that cell's own majority vote. This is what
    replaces the manuscript's unsupported "repeated runs confirmed stable outputs".
    """
    rows = []
    for label in labels:
        subset = df_runs[df_runs["model"] == label]
        by_cell = subset.groupby(["paper_id", "question_id"])["answer"]
        counts = by_cell.agg(n="size", yes="sum")
        counts = counts[counts["n"] > 1]
        if counts.empty:
            continue
        unanimous = (counts["yes"] == 0) | (counts["yes"] == counts["n"])
        majority_share = np.maximum(counts["yes"], counts["n"] - counts["yes"]) / counts["n"]
        rows.append({
            "model": label,
            "cells": len(counts),
            "runs": int(counts["n"].max()),
            "flip rate": float(1 - unanimous.mean()),
            "mean agreement with own majority": float(majority_share.mean()),
        })
    return pd.DataFrame(rows)


def majority_vote(
    df_runs: pd.DataFrame, *, labels: Sequence[str] = MODEL_LABELS
) -> pd.DataFrame:
    """Collapse the runs to one answer per (study, item, model) by majority vote.

    Ties -- possible with an even number of runs -- resolve to 0, matching the prompt's
    instruction to answer NO unless the evidence is explicit. Returns a frame shaped like
    a rater table so it drops straight into the existing agreement functions.
    """
    votes = (
        df_runs.groupby(["paper_id", "question_id", "model"])["answer"]
        .agg(["sum", "size"])
        .reset_index()
    )
    votes["answer"] = (votes["sum"] * 2 > votes["size"]).astype(int)
    return votes.pivot(
        index=["paper_id", "question_id"], columns="model", values="answer"
    ).reset_index()[["paper_id", "question_id"] + list(labels)]
