"""
src/inference/runner.py
========================
Iterates over the prompt dataset, runs each prompt through Qwen3-1.7B-Base,
and saves:

  1. Raw residual stream tensors to:
       activations/raw/{problem_id}_{condition}_{layer}.pt   (float16)
  2. Greedy-decoded model answers + metadata to:
       cfg.dataset.outputs_path   (JSON, written after every prompt)

Key design points:

Two-pass per prompt:
  Pass 1 — capture_prefill_activations() runs a clean single forward pass
  on the prompt tokens via TransformerLens run_with_cache(). No hooks during
  generation; no decode-step overwrite risk.
  Pass 2 — model.generate() runs the autoregressive decode to get the text
  answer. Completely decoupled from activation capture.

Incremental save:
  outputs.json is written to disk after every successfully processed prompt.
  A crash at prompt 250 leaves prompts 1–249 safely in outputs.json and all
  their .pt files on disk. Re-running resumes from where it left off.

Checkpoint/resume:
  A (problem_id, condition) pair is skipped if it already has an entry in
  outputs.json AND all expected .pt files on disk.

prompt_len metadata:
  Each output record stores prompt_len (number of prompt tokens). Phase 4's
  token_position="last_generated" logic needs this to slice the correct
  token from the [1, prompt_len, d_model] activation tensor.

dry_run mode:
  Skips model forward passes and disk writes entirely. Useful for validating
  the pipeline without a GPU.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

from src.inference.activation_extractor import capture_prefill_activations


def _activation_path(raw_dir: str, problem_id: str, condition: str, layer: int) -> Path:
    """Return the canonical save path for a raw activation tensor."""
    return Path(raw_dir) / f"{problem_id}_{condition}_{layer}.pt"


def _all_layers_saved(
    raw_dir: str, problem_id: str, condition: str, layers: list[int]
) -> bool:
    """Return True if every layer's .pt file already exists on disk."""
    return all(
        _activation_path(raw_dir, problem_id, condition, layer).exists()
        for layer in layers
    )


def _load_existing_outputs(outputs_path: str) -> list[dict[str, Any]]:
    """
    Load existing outputs.json records so a resumed run can skip already-done
    prompts. Returns [] if the file is absent, empty, or unreadable.
    """
    path = Path(outputs_path)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.warning(f"Could not read {outputs_path}; starting outputs fresh.")
        return []


def run_inference(
    prompts: list[dict[str, Any]],
    model: HookedTransformer | None,
    tokenizer: PreTrainedTokenizerBase | None,
    cfg: DictConfig,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """
    Run inference over a list of prompts, saving raw activations and answers.

    Args:
        prompts:   List of prompt dicts with keys: problem_id, category,
                   condition, prompt, answer.
        model:     HookedTransformer in eval mode. None only in dry_run mode.
        tokenizer: model.tokenizer. None only in dry_run mode.
        cfg:       Hydra DictConfig (uses cfg.sae.layers_to_analyze,
                   cfg.activations.raw_dir, cfg.model.max_new_tokens,
                   cfg.dataset.outputs_path).
        dry_run:   If True, skip model passes and disk writes.

    Returns:
        List of output records. In dry_run mode contains placeholder answers;
        in full mode contains greedy-decoded answers plus prompt metadata.
    """
    layers: list[int] = list(cfg.sae.layers_to_analyze)
    raw_dir: str = cfg.activations.raw_dir
    max_new_tokens: int = cfg.model.max_new_tokens
    repetition_penalty: float = float(getattr(cfg.model, "repetition_penalty", 1.0))
    outputs_path: str = cfg.dataset.outputs_path

    # --- Dry-run path: log what would happen, return placeholder records ---
    if dry_run:
        outputs: list[dict[str, Any]] = []
        for item in tqdm(prompts, desc="Dry run"):
            problem_id: str = item["problem_id"]
            condition: str = item["condition"]
            gold_answer: str = item["answer"]
            logger.info(
                f"[dry_run] Would process: {problem_id}/{condition} | layers={layers}"
            )
            outputs.append(
                {
                    "problem_id": problem_id,
                    "condition": condition,
                    "model_answer": "<dry_run>",
                    "gold_answer": gold_answer,
                    "prompt_len": -1,
                }
            )
        return outputs

    # --- Full-run path ---
    assert model is not None and tokenizer is not None

    Path(raw_dir).mkdir(parents=True, exist_ok=True)

    # Resume: load any existing records and build a skip-set
    outputs = _load_existing_outputs(outputs_path)
    already_done: set[tuple[str, str]] = {
        (r["problem_id"], r["condition"]) for r in outputs
    }
    if already_done:
        logger.info(f"Resuming: {len(already_done)} prompt(s) already completed.")

    for item in tqdm(prompts, desc="Running inference"):
        problem_id = item["problem_id"]
        condition = item["condition"]
        prompt_text: str = item["prompt"]
        gold_answer = item["answer"]

        # Skip if both outputs.json entry and .pt files already exist
        if (problem_id, condition) in already_done and _all_layers_saved(
            raw_dir, problem_id, condition, layers
        ):
            logger.debug(f"Skipping {problem_id}/{condition} — already saved.")
            continue

        # --- Tokenize ---
        # to_tokens returns [1, prompt_len] on the model's device
        tokens = model.to_tokens(prompt_text, prepend_bos=False)
        prompt_len: int = tokens.shape[1]

        # --- Pass 1: capture prefill activations (prompt-only forward pass) ---
        captured = capture_prefill_activations(model, tokens, layers)

        # --- Pass 2: generate text answer (no hooks involved) ---
        with torch.inference_mode():
            generated = cast(
                torch.Tensor,
                model.generate(
                    tokens,
                    max_new_tokens=max_new_tokens,
                    stop_at_eos=True,
                    do_sample=False,
                    prepend_bos=False,
                    repetition_penalty=repetition_penalty,
                    verbose=False,
                ),
            )
        # generated: [1, prompt_len + n_new_tokens]
        new_token_ids = generated[0, prompt_len:]
        decoded = tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True)
        decoded_text = decoded[0] if isinstance(decoded, list) else decoded

        # Belt-and-suspenders: prefer the explicit <END> marker, then fall back
        # to older continuation boundaries if the model ignores the new format.
        model_answer = re.split(
            r"<END>|\n\n|\nProblem:|\nQuestion:", decoded_text, maxsplit=1
        )[0].strip()
        model_answer = re.sub(r"^Final answer:\s*", "", model_answer, flags=re.I)

        # --- Save activations (float16) ---
        for layer_idx, tensor in captured.items():
            save_path = _activation_path(raw_dir, problem_id, condition, layer_idx)
            torch.save(tensor.to(torch.float16), save_path)
            logger.debug(f"Saved: {save_path} | shape={tuple(tensor.shape)}")

        # --- Record output and persist immediately ---
        record: dict[str, Any] = {
            "problem_id": problem_id,
            "condition": condition,
            "model_answer": model_answer,
            "gold_answer": gold_answer,
            "prompt_len": prompt_len,
        }
        outputs.append(record)
        already_done.add((problem_id, condition))
        save_outputs(outputs, outputs_path)

    return outputs


def save_outputs(outputs: list[dict[str, Any]], outputs_path: str) -> None:
    """Persist output records to disk as JSON."""
    Path(outputs_path).parent.mkdir(parents=True, exist_ok=True)
    with open(outputs_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2)
    logger.debug(f"Wrote {len(outputs)} record(s) to {outputs_path}")
