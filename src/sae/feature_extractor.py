"""
src/sae/feature_extractor.py
=============================
Iterates over raw residual stream activation files, projects each through the
corresponding layer's SAE, and saves the resulting sparse feature tensors.

Input format  (from Phase 2 / runner.py):
    activations/raw/{problem_id}_{condition}_{layer}.pt
    Shape: [1, seq_len, d_model]  (float16)

Output format (consumed by Phase 4 / analysis):
    activations/sae_features/{problem_id}_{condition}_{layer}.pt
    Shape: [seq_len, d_sae]  (float16)
    Property: exactly cfg.sae.top_k_features non-zero values per row.

Key design decisions:
    - SAEs are loaded once per layer and reused across all problems.
    - Files are skipped if the output already exists (resume support).
    - Output is saved as float16 to halve disk usage.
    - Works with both the main raw/ directory and the raw_10_sanity/ sanity
      directory; pass raw_dir_override to switch.
"""

from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from src.sae.projector import project_topk
from src.sae.sae_loader import load_saes_for_layers


def _parse_stem(stem: str) -> tuple[str, int] | None:
    """
    Parse a raw activation filename stem into (prefix, layer).

    Filename pattern: ``{problem_id}_{condition}_{layer}``
    Examples:
        ``arith_5_clean_6``         → ("arith_5_clean", 6)
        ``gsm8k_10_helpful_hint_27`` → ("gsm8k_10_helpful_hint", 27)

    Returns None if the last component is not an integer.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    prefix, layer_str = parts
    try:
        return prefix, int(layer_str)
    except ValueError:
        return None


def extract_and_save_features(
    cfg: DictConfig,
    raw_dir_override: str | None = None,
    sae_dir_override: str | None = None,
) -> int:
    """
    Project all raw activation files through their SAEs and save feature tensors.

    Args:
        cfg:              Hydra DictConfig (uses cfg.sae.*, cfg.activations.*).
        raw_dir_override: If provided, overrides cfg.activations.raw_dir.
                          Useful for pointing at ``activations/raw_10_sanity/``.
        sae_dir_override: If provided, overrides cfg.activations.sae_dir.

    Returns:
        Number of files written (skipped files not counted).
    """
    raw_dir = Path(raw_dir_override or cfg.activations.raw_dir)
    sae_dir = Path(sae_dir_override or cfg.activations.sae_dir)
    sae_dir.mkdir(parents=True, exist_ok=True)

    layers: list[int] = list(cfg.sae.layers_to_analyze)
    hf_repo: str = cfg.sae.hf_repo
    top_k: int = cfg.sae.top_k_features

    # --- Load all SAEs upfront ---
    logger.info(f"Loading SAEs for layers {layers} from {hf_repo} …")
    saes = load_saes_for_layers(hf_repo, layers, top_k=top_k, device="cpu")

    # --- Collect raw files ---
    raw_files = sorted(raw_dir.glob("*.pt"))
    if not raw_files:
        logger.warning(f"No .pt files found in {raw_dir}")
        return 0

    logger.info(f"Found {len(raw_files)} raw activation file(s) in {raw_dir}")

    n_written = 0
    for raw_file in tqdm(raw_files, desc="SAE projection"):
        parsed = _parse_stem(raw_file.stem)
        if parsed is None:
            logger.warning(f"Skipping unrecognised filename: {raw_file.name}")
            continue

        prefix, layer = parsed
        if layer not in saes:
            logger.debug(f"Layer {layer} not in SAE config — skipping {raw_file.name}")
            continue

        out_path = sae_dir / f"{prefix}_{layer}.pt"
        if out_path.exists():
            logger.debug(f"Already exists, skipping: {out_path.name}")
            continue

        # Load raw activation [1, seq_len, d_model] float16
        residual: torch.Tensor = torch.load(
            raw_file, map_location="cpu", weights_only=True
        )

        # Project through SAE → [seq_len, d_sae] float32
        acts = project_topk(residual, saes[layer])

        # Save as float16 to halve storage
        torch.save(acts.to(torch.float16), out_path)
        logger.debug(
            f"Saved {out_path.name} | shape={tuple(acts.shape)} | "
            f"nnz/token={int((acts != 0).sum() / acts.shape[0])}"
        )
        n_written += 1

    logger.info(f"Done. {n_written} feature file(s) written to {sae_dir}")
    return n_written
