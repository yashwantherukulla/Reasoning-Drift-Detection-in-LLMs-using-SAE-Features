"""
scripts/run_sae_projection.py
==============================
CLI entry point for Phase 3 — SAE Projection.

Projects every raw residual stream activation file through the official
Qwen-Scope Top-K SAEs and writes sparse feature tensors to:
    activations/sae_features/{problem_id}_{condition}_{layer}.pt

Usage:
    # Full run (reads from activations.raw_dir in config)
    uv run scripts/run_sae_projection.py

    # Run against the 10-problem sanity set
    uv run scripts/run_sae_projection.py +raw_dir=activations/raw_10_sanity

    # Inspect config without running
    uv run scripts/run_sae_projection.py --cfg job

Notes:
    - SAEs are downloaded from HuggingFace on first run, then cached locally
      by huggingface_hub.
    - Each layer's SAE is loaded once and reused across all problems.
    - Existing output files are skipped automatically (resume-safe).
"""

from __future__ import annotations

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.config import register_configs

register_configs()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir_override: str | None = OmegaConf.select(cfg, "raw_dir", default=None)

    logger.info(f"Run config:\n{OmegaConf.to_yaml(cfg)}")
    if raw_dir_override:
        logger.info(f"raw_dir override: {raw_dir_override}")

    from src.sae.feature_extractor import extract_and_save_features

    n_written = extract_and_save_features(
        cfg,
        raw_dir_override=raw_dir_override,
    )
    logger.info(f"Phase 3 complete — {n_written} feature file(s) written.")


if __name__ == "__main__":
    main()
