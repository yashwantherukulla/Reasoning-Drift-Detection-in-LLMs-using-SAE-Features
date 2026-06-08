"""
src/analysis/stability.py
=========================
Correlates per-problem RDS (and other divergence metrics) with answer
sensitivity — does higher divergence predict an answer change?

This module implements task 4.4 from phases.md:

    "Correlate RDS with answer sensitivity (does high RDS predict answer
    change?).  Stratify: (same-answer, high-RDS) vs (diff-answer, high-RDS)
    to distinguish brittleness from robustness."

Key design decisions
--------------------
- ``answer_changed`` is True when the model gave *different* answers under
  the two conditions being compared (case-insensitive, stripped).
- ``answer_correct_{a,b}`` are True when the model answer matches gold_answer
  for the respective condition.
- Stratification uses the median RDS across all rows as the high/low boundary
  (configurable via ``rds_threshold``).

Output
------
``compute_stability`` returns two DataFrames:
  1. ``detail_df``  — the full metrics DataFrame enriched with answer metadata.
  2. ``summary_df`` — aggregated statistics per (layer, cond_a, cond_b,
                      answer_changed) stratum.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_answers(outputs_path: str | Path) -> dict[str, dict[str, dict]]:
    """
    Return ``{problem_id: {condition: {"model_answer": ..., "gold_answer": ...}}}``
    from the inference outputs JSON.
    """
    path = Path(outputs_path)
    if not path.exists():
        raise FileNotFoundError(f"Outputs file not found: {path}")

    with open(path) as f:
        records = json.load(f)

    index: dict[str, dict[str, dict]] = {}
    for rec in records:
        index.setdefault(rec["problem_id"], {})[rec["condition"]] = rec
    return index


from src.dataset.answer_parser import extract_answer, has_repetition_loop


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_stability(
    metrics_df: pd.DataFrame,
    outputs_path: str | Path,
    rds_threshold: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Enrich metrics DataFrame with answer sensitivity columns and produce a
    stratified summary.

    Args:
        metrics_df:     Output of ``layerwise.compute_layerwise_metrics``.
        outputs_path:   Path to inference outputs JSON.
        rds_threshold:  RDS value dividing "high" from "low" divergence.
                        Defaults to the median RDS across all rows.

    Returns:
        (detail_df, summary_df)
        detail_df  — full metrics with extra columns:
                     model_answer_a, model_answer_b, gold_answer,
                     answer_changed (bool), answer_correct_a (bool),
                     answer_correct_b (bool), rds_above_threshold (bool),
                     has_repetition_loop_a (bool), has_repetition_loop_b (bool).
        summary_df — group statistics (mean/std of all metric columns,
                     count, % answer changed) per
                     (layer, cond_a, cond_b, answer_changed).
    """
    if metrics_df.empty:
        logger.warning("metrics_df is empty — returning empty stability DataFrames.")
        return pd.DataFrame(), pd.DataFrame()

    answers = _load_answers(outputs_path)

    # --- Enrich detail rows ---
    answer_changed_col: list[bool] = []
    model_ans_a_col: list[str] = []
    model_ans_b_col: list[str] = []
    gold_ans_col: list[str] = []
    correct_a_col: list[bool] = []
    correct_b_col: list[bool] = []
    loop_a_col: list[bool] = []
    loop_b_col: list[bool] = []

    for _, row in metrics_df.iterrows():
        pid = row["problem_id"]
        ca = row["cond_a"]
        cb = row["cond_b"]

        rec_a = answers.get(pid, {}).get(ca, {})
        rec_b = answers.get(pid, {}).get(cb, {})

        raw_a = rec_a.get("model_answer", "")
        raw_b = rec_b.get("model_answer", "")
        
        ma = extract_answer(raw_a, pid)
        mb = extract_answer(raw_b, pid)
        gold = extract_answer(rec_a.get("gold_answer") or rec_b.get("gold_answer"), pid)

        model_ans_a_col.append(ma)
        model_ans_b_col.append(mb)
        gold_ans_col.append(gold)
        answer_changed_col.append(ma != mb)
        correct_a_col.append(bool(ma) and ma == gold)
        correct_b_col.append(bool(mb) and mb == gold)
        loop_a_col.append(has_repetition_loop(raw_a))
        loop_b_col.append(has_repetition_loop(raw_b))

    detail = metrics_df.copy()
    detail["model_answer_a"] = model_ans_a_col
    detail["model_answer_b"] = model_ans_b_col
    detail["gold_answer"] = gold_ans_col
    detail["answer_changed"] = answer_changed_col
    detail["answer_correct_a"] = correct_a_col
    detail["answer_correct_b"] = correct_b_col
    detail["has_repetition_loop_a"] = loop_a_col
    detail["has_repetition_loop_b"] = loop_b_col

    # --- Threshold for high/low RDS ---
    threshold = rds_threshold if rds_threshold is not None else detail["rds"].median()
    detail["rds_above_threshold"] = detail["rds"] > threshold
    logger.info(
        f"Stability analysis: RDS threshold={threshold:.4f} "
        f"(median), {detail['answer_changed'].sum()} answer changes "
        f"out of {len(detail)} rows."
    )

    # --- Summary statistics ---
    metric_cols = [
        "jaccard",
        "weighted_jaccard",
        "cosine_sim",
        "l1_distance",
        "rds",
        "weighted_rds",
        "exclusive_mass_asymmetry",
        "drift_dir_cosine",
    ]
    # Only keep columns that exist in the DataFrame
    metric_cols = [c for c in metric_cols if c in detail.columns]

    group_keys = ["layer", "cond_a", "cond_b", "answer_changed"]
    summary = (
        detail.groupby(group_keys)[metric_cols]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    # Flatten multi-level columns
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]

    # Add % answer changed per (layer, cond_a, cond_b)
    pct_changed = (
        detail.groupby(["layer", "cond_a", "cond_b"])["answer_changed"]
        .mean()
        .rename("pct_answer_changed")
        .reset_index()
    )
    summary = summary.merge(pct_changed, on=["layer", "cond_a", "cond_b"], how="left")

    # --- Log high-RDS / answer-changed stratum ---
    _log_stratum_comparison(detail, threshold)

    return detail, summary


def _log_stratum_comparison(detail: pd.DataFrame, threshold: float) -> None:
    """Log the 2×2 stratum breakdown: high/low RDS × answer changed/same."""
    for pair in detail[["cond_a", "cond_b"]].drop_duplicates().itertuples():
        sub = detail[
            (detail["cond_a"] == pair.cond_a) & (detail["cond_b"] == pair.cond_b)
        ]
        if sub.empty:
            continue
        hi_rds = sub["rds"] > threshold
        changed = sub["answer_changed"]
        n_total = len(sub)

        counts = {
            "(high-RDS, changed)": int((hi_rds & changed).sum()),
            "(high-RDS, same)": int((hi_rds & ~changed).sum()),
            "(low-RDS,  changed)": int((~hi_rds & changed).sum()),
            "(low-RDS,  same)": int((~hi_rds & ~changed).sum()),
        }
        lines = "\n".join(f"  {k}: {v}/{n_total}" for k, v in counts.items())
        logger.info(
            f"Stratum breakdown for {pair.cond_a!r} vs {pair.cond_b!r}:\n{lines}"
        )
