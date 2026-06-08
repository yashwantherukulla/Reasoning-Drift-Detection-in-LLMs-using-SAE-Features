"""
src/analysis/layerwise.py
=========================
Orchestrates layerwise divergence analysis across all problems, condition
pairs, and SAE layers.

For each combination of (problem_id, condition_pair, layer) this module:
  1. Loads the two SAE feature tensors from ``activations/sae_features/``.
  2. Resolves the comparison token position (last_generated or last_prefix).
  3. Computes all pairwise metrics via ``metrics.compare_pair``.
  4. Computes the 3-condition drift-direction-cosine when all three conditions
     are available for a problem.

Output: a ``pandas.DataFrame`` with one row per (problem × pair × layer).
The caller (``scripts/run_analysis.py``) saves this to CSV.

Column schema
-------------
problem_id          str   e.g. "arith_5"
layer               int   e.g. 6, 12, 18, 20, 24, 27
cond_a              str   e.g. "clean"
cond_b              str   e.g. "helpful_hint"
token_position      str   "last_generated" | "last_prefix"
pos_a               int   resolved integer index into feat_a
pos_b               int   resolved integer index into feat_b
jaccard             float binary Jaccard similarity
weighted_jaccard    float soft Jaccard using activation magnitudes
cosine_sim          float cosine similarity
l1_distance         float L1 (Manhattan) distance
rds                 float 1 − Jaccard
weighted_rds        float 1 − weighted_Jaccard
exclusive_mass_asymmetry  float directional drift asymmetry
drift_dir_cosine    float cosine of (helpful−clean) vs (misleading−clean) delta vectors;
                          NaN unless the pair is (clean, helpful_hint) or
                          (clean, misleading_hint) and all three conds are available
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm

from src.analysis.metrics import (
    compare_pair,
    drift_direction_cosine,
    resolve_position,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CONDS = ("clean", "helpful_hint", "misleading_hint")


def _load_outputs(outputs_path: str | Path) -> dict[str, dict[str, dict]]:
    """
    Load inference outputs and index as outputs[problem_id][condition].

    Returns a nested dict:
        { problem_id: { condition: {model_answer, gold_answer, prompt_len?} } }
    """
    path = Path(outputs_path)
    if not path.exists():
        raise FileNotFoundError(f"Outputs file not found: {path}")

    with open(path) as f:
        records = json.load(f)

    index: dict[str, dict[str, dict]] = {}
    for rec in records:
        pid = rec["problem_id"]
        cond = rec["condition"]
        index.setdefault(pid, {})[cond] = rec
    return index


def _load_feat(
    sae_dir: Path, problem_id: str, condition: str, layer: int
) -> torch.Tensor | None:
    """
    Load a single SAE feature tensor from disk.

    Expected filename: ``{problem_id}_{condition}_{layer}.pt``
    Returns ``None`` if the file does not exist.
    """
    path = sae_dir / f"{problem_id}_{condition}_{layer}.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True)


def _discover_problems(sae_dir: Path) -> list[str]:
    """
    Scan *sae_dir* and return a sorted list of unique problem_id strings.
    Parses filenames of the form ``{problem_id}_{condition}_{layer}.pt``.
    """
    problem_ids: set[str] = set()
    for p in sae_dir.glob("*.pt"):
        # stem = e.g. "arith_5_clean_6"  or  "gsm8k_10_helpful_hint_27"
        parts = p.stem.rsplit("_", 1)  # split off layer
        if len(parts) != 2:
            continue
        prefix = parts[0]  # e.g. "arith_5_clean"
        # strip condition suffix
        for cond in _CONDS:
            suffix = f"_{cond}"
            if prefix.endswith(suffix):
                problem_ids.add(prefix[: -len(suffix)])
                break
    return sorted(problem_ids)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_layerwise_metrics(
    sae_dir: str | Path,
    outputs_path: str | Path,
    layers: list[int],
    pairwise_conditions: list[list[str]],
    token_position: str = "last_generated",
) -> pd.DataFrame:
    """
    Compute pairwise divergence metrics for all problems × layers × condition pairs.

    Args:
        sae_dir:              Directory containing SAE feature ``.pt`` files.
        outputs_path:         Path to inference outputs JSON (with optional
                              ``prompt_len`` field per record).
        layers:               List of layer indices to analyze.
        pairwise_conditions:  List of [cond_a, cond_b] pairs to compare.
        token_position:       Named token position for metric computation.
                              ``"last_generated"`` (default) uses the last
                              token in each sequence.  ``"last_prefix"``
                              requires ``prompt_len`` in the outputs JSON.

    Returns:
        DataFrame with one row per (problem_id, layer, cond_a, cond_b).
    """
    sae_dir = Path(sae_dir)
    outputs = _load_outputs(outputs_path)

    problem_ids = _discover_problems(sae_dir)
    if not problem_ids:
        raise RuntimeError(f"No SAE feature files found in {sae_dir}")

    logger.info(
        f"Analysing {len(problem_ids)} problem(s) × {len(layers)} layer(s) "
        f"× {len(pairwise_conditions)} pair(s) at position '{token_position}'"
    )

    rows: list[dict[str, Any]] = []

    for problem_id in tqdm(problem_ids, desc="Problems"):
        # Pre-load all feature tensors for this problem (all conds × all layers)
        feats: dict[str, dict[int, torch.Tensor | None]] = {cond: {} for cond in _CONDS}
        for cond in _CONDS:
            for layer in layers:
                feats[cond][layer] = _load_feat(sae_dir, problem_id, cond, layer)

        # Retrieve prompt_len per condition (may be missing)
        prompt_lens: dict[str, int | None] = {
            cond: outputs.get(problem_id, {}).get(cond, {}).get("prompt_len")
            for cond in _CONDS
        }

        for layer in layers:
            # Compute drift_direction_cosine across the 3-condition triangle
            # at this layer (requires all three feature tensors to be present)
            feat_clean = feats["clean"][layer]
            feat_helpful = feats["helpful_hint"][layer]
            feat_mislead = feats["misleading_hint"][layer]

            ddc_val: float | None = None
            if (
                feat_clean is not None
                and feat_helpful is not None
                and feat_mislead is not None
            ):
                try:
                    pos_clean = resolve_position(
                        feat_clean, token_position, prompt_lens["clean"]
                    )
                    pos_helpful = resolve_position(
                        feat_helpful, token_position, prompt_lens["helpful_hint"]
                    )
                    pos_mislead = resolve_position(
                        feat_mislead, token_position, prompt_lens["misleading_hint"]
                    )
                    ddc_val = drift_direction_cosine(
                        feat_clean[pos_clean],
                        feat_helpful[pos_helpful],
                        feat_mislead[pos_mislead],
                    )
                except Exception as exc:
                    logger.warning(
                        f"drift_direction_cosine failed for {problem_id} layer {layer}: {exc}"
                    )

            for pair in pairwise_conditions:
                cond_a, cond_b = pair[0], pair[1]

                feat_a = feats[cond_a][layer]
                feat_b = feats[cond_b][layer]

                if feat_a is None or feat_b is None:
                    logger.debug(
                        f"Missing feature file: {problem_id} {cond_a}/{cond_b} layer {layer}"
                    )
                    continue

                try:
                    pos_a = resolve_position(
                        feat_a, token_position, prompt_lens[cond_a]
                    )
                    pos_b = resolve_position(
                        feat_b, token_position, prompt_lens[cond_b]
                    )
                except ValueError as exc:
                    logger.warning(f"Cannot resolve position for {problem_id}: {exc}")
                    continue

                metrics = compare_pair(feat_a, feat_b, pos_a, pos_b)

                row: dict[str, Any] = {
                    "problem_id": problem_id,
                    "layer": layer,
                    "cond_a": cond_a,
                    "cond_b": cond_b,
                    "token_position": token_position,
                    "pos_a": pos_a,
                    "pos_b": pos_b,
                    **metrics,
                    "drift_dir_cosine": ddc_val,
                }
                rows.append(row)

    if not rows:
        logger.warning("No metric rows computed — check that feature files exist.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    logger.info(f"Computed {len(df)} metric rows.")
    return df
