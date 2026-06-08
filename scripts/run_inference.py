"""
scripts/run_inference.py
=========================
CLI entry point for Phase 2 — Inference & Activation Collection.

Usage:
    uv run scripts/run_inference.py
    uv run scripts/run_inference.py model.device=cpu
    uv run scripts/run_inference.py +dry_run=true dataset.n_problems=2

Notes:
- activations.batch_size is defined in the config schema but not implemented;
  inference always runs one prompt at a time (batch_size=1) to avoid GPU OOM.
- The +dry_run=true flag skips model loading and forward passes entirely —
  useful for validating the pipeline without a GPU.
- In non-dry_run mode, run_inference() writes outputs.json incrementally
  after every prompt. The final save_outputs() call below is a no-op that
  serves as an explicit final-sync guard.
"""

from __future__ import annotations

import json

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from src.config import register_configs

register_configs()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    dry_run: bool = OmegaConf.select(cfg, "dry_run", default=False)

    logger.info(f"Run config:\n{OmegaConf.to_yaml(cfg)}")
    logger.info(f"dry_run={dry_run}")

    # Load prompts
    with open(cfg.dataset.prompts_path, "r", encoding="utf-8") as f:
        all_prompts: list[dict] = json.load(f)

    n_problems: int = cfg.dataset.n_problems
    # Collect the first n_problems unique problem IDs in order.
    # prompts.json may interleave conditions, so we de-duplicate by problem_id
    # while preserving the original ordering.
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []
    for item in all_prompts:
        pid = item["problem_id"]
        if pid not in seen_ids:
            seen_ids.add(pid)
            ordered_ids.append(pid)
        if len(ordered_ids) == n_problems:
            break

    # Keep only prompts that belong to the selected problem IDs
    selected_ids = set(ordered_ids)
    prompts = [p for p in all_prompts if p["problem_id"] in selected_ids]

    logger.info(
        f"Loaded {len(prompts)} prompts for {len(selected_ids)} problems "
        f"({len(selected_ids) * 3} expected — 3 conditions each)."
    )

    if dry_run:
        from src.inference.runner import run_inference, save_outputs

        outputs = run_inference(
            prompts=prompts,
            model=None,  # type: ignore[arg-type]
            tokenizer=None,  # type: ignore[arg-type]
            cfg=cfg,
            dry_run=True,
        )
        save_outputs(outputs, cfg.dataset.outputs_path)
        logger.info("Dry run complete.")
        return

    # Full run: load model, then run inference
    from src.inference.model_loader import load_model_and_tokenizer
    from src.inference.runner import run_inference, save_outputs

    model, tokenizer = load_model_and_tokenizer(cfg)

    outputs = run_inference(
        prompts=prompts,
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        dry_run=False,
    )

    # run_inference() already wrote outputs.json after every prompt.
    # This final call is an explicit flush guard in case of unexpected early exit.
    save_outputs(outputs, cfg.dataset.outputs_path)
    logger.info(
        f"Inference complete. {len(outputs)} records in {cfg.dataset.outputs_path}"
    )


if __name__ == "__main__":
    main()
