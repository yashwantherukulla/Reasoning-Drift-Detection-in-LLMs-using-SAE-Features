"""
scripts/run_analysis.py
=======================
CLI entry point for Phase 4 — Feature Analysis & Metrics.

Computes Jaccard, weighted Jaccard, cosine similarity, L1 distance, RDS,
weighted RDS, exclusive mass asymmetry, and drift-direction cosine for all
problems × SAE layers × condition pairs, then correlates divergence metrics
with answer sensitivity.

Output files (in ``cfg.analysis.results_dir``):
    metrics_detail.csv    — one row per (problem × layer × pair)
    metrics_summary.csv   — mean/std/count grouped by (layer, cond_a, cond_b)
    stability_detail.csv  — metrics enriched with answer-change columns
    stability_summary.csv — stratified stats (high/low RDS × answer changed)

Usage:
    # Run with default config (reads activations/sae_features/, outputs_10_sanity.json)
    uv run scripts/run_analysis.py

    # Override token position
    uv run scripts/run_analysis.py analysis.token_position=last_prefix

    # Override outputs file
    uv run scripts/run_analysis.py +analysis.outputs_path=data/processed/outputs.json

    # Custom results directory
    uv run scripts/run_analysis.py analysis.results_dir=results/custom/

    # Inspect effective config without running
    uv run scripts/run_analysis.py --cfg job

Hydra overrides:
    +analysis.outputs_path=<path>  Path to outputs JSON (overrides default)
    +analysis.sae_dir=<path>       Override sae features directory
"""

from __future__ import annotations

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.config import register_configs

register_configs()

# Default outputs file — outputs_10_sanity.json has prompt_len; fall back
# to outputs.json for the full run.
_DEFAULT_OUTPUTS_PRIORITY = [
    "data/processed/outputs_10_sanity.json",
    "data/processed/outputs.json",
]


def _resolve_outputs_path(cfg: DictConfig) -> str:
    """Pick the best available outputs JSON to use as answer/prompt metadata."""
    # Allow explicit override via +analysis.outputs_path=...
    override = OmegaConf.select(cfg, "analysis.outputs_path", default=None)
    if override:
        return override

    for candidate in _DEFAULT_OUTPUTS_PRIORITY:
        if Path(candidate).exists():
            logger.info(f"Using outputs file: {candidate}")
            return candidate

    raise FileNotFoundError(
        "No outputs JSON found. Expected one of: "
        + ", ".join(_DEFAULT_OUTPUTS_PRIORITY)
    )


def _resolve_sae_dir(cfg: DictConfig) -> str:
    """Allow +analysis.sae_dir override, else use cfg.activations.sae_dir."""
    override = OmegaConf.select(cfg, "analysis.sae_dir", default=None)
    return override or cfg.activations.sae_dir


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logger.info(f"Phase 4 — Feature Analysis & Metrics\n{OmegaConf.to_yaml(cfg)}")

    # --- Resolve paths ---
    sae_dir = _resolve_sae_dir(cfg)
    outputs_path = _resolve_outputs_path(cfg)
    results_dir = Path(cfg.analysis.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    layers: list[int] = list(cfg.sae.layers_to_analyze)
    pairwise_conditions: list[list[str]] = [
        list(pair) for pair in cfg.analysis.pairwise_conditions
    ]
    token_position: str = cfg.analysis.token_position

    # --- Phase 4 core: compute layerwise metrics ---
    from src.analysis.layerwise import compute_layerwise_metrics

    metrics_df = compute_layerwise_metrics(
        sae_dir=sae_dir,
        outputs_path=outputs_path,
        layers=layers,
        pairwise_conditions=pairwise_conditions,
        token_position=token_position,
    )

    if metrics_df.empty:
        logger.error("No metrics computed — aborting.")
        return

    # Save core metrics
    metrics_path = results_dir / "metrics_detail.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved {len(metrics_df)} rows → {metrics_path}")

    # Save per-layer summary
    metric_cols = [
        c
        for c in [
            "jaccard",
            "weighted_jaccard",
            "cosine_sim",
            "l1_distance",
            "rds",
            "weighted_rds",
            "exclusive_mass_asymmetry",
            "drift_dir_cosine",
        ]
        if c in metrics_df.columns
    ]
    summary_df = (
        metrics_df.groupby(["layer", "cond_a", "cond_b"])[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary_df.columns = ["_".join(c).rstrip("_") for c in summary_df.columns]
    summary_path = results_dir / "metrics_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary → {summary_path}")

    # --- Sanity check: RDS between clean and clean should be ~0 ---
    _sanity_check_clean_vs_clean(sae_dir, layers, token_position)

    # --- Phase 4.4: stability / answer-sensitivity analysis ---
    from src.analysis.stability import compute_stability

    detail_df, stab_summary_df = compute_stability(
        metrics_df=metrics_df,
        outputs_path=outputs_path,
    )

    if not detail_df.empty:
        stab_detail_path = results_dir / "stability_detail.csv"
        detail_df.to_csv(stab_detail_path, index=False)
        logger.info(f"Saved stability detail → {stab_detail_path}")

    if not stab_summary_df.empty:
        stab_summary_path = results_dir / "stability_summary.csv"
        stab_summary_df.to_csv(stab_summary_path, index=False)
        logger.info(f"Saved stability summary → {stab_summary_path}")

    logger.info("Phase 4 complete.")


def _sanity_check_clean_vs_clean(
    sae_dir: str,
    layers: list[int],
    token_position: str,
) -> None:
    """
    Task 4.7: verify RDS between clean and clean is ~0.

    Loads the first available clean feature file and computes metrics
    against itself.  Any non-zero RDS indicates a bug in the metric code.
    """
    from src.analysis.metrics import compare_pair, resolve_position

    sae_path = Path(sae_dir)
    import torch

    for pt_file in sorted(sae_path.glob("*_clean_*.pt")):
        try:
            feat = torch.load(pt_file, map_location="cpu", weights_only=True)
            pos = resolve_position(feat, token_position)
            result = compare_pair(feat, feat, pos, pos)
            j = result["jaccard"]
            r = result["rds"]
            wj = result["weighted_jaccard"]
            wrds = 1.0 - wj
            if r > 1e-6 or wrds > 1e-6 or j < 1.0 - 1e-6:
                logger.error(
                    f"SANITY FAIL: clean vs. clean for {pt_file.name}: "
                    f"RDS={r:.6f}, weighted_RDS={wrds:.6f} (expected 0.0)"
                )
            else:
                logger.info(
                    f"Sanity check PASS: clean vs. clean RDS=0.0, "
                    f"Jaccard=1.0 for {pt_file.name}"
                )
            return  # one check is sufficient
        except Exception as exc:
            logger.warning(f"Sanity check skipped ({pt_file.name}): {exc}")
            continue


if __name__ == "__main__":
    main()
