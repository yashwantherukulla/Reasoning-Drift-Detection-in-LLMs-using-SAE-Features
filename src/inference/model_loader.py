"""
src/inference/model_loader.py
==============================
Loads Qwen3-1.7B-Base via TransformerLens HookedTransformer.
Model dtype and device are driven entirely by config (cfg.model.dtype,
cfg.model.device) — no hard-coded defaults here.

Using HookedTransformer provides named hook points
(e.g. blocks.{i}.hook_resid_post) that we can target with run_with_cache()
for lifecycle-free activation capture — no manual register/remove required
and no decode-step overwrite risk.

Notes:
- batch_size=1 is the intended operating mode (single prompt per forward pass).
  Batched inference would require left-padding alignment; not implemented.
- device_map is not used: HookedTransformer.from_pretrained(device=...) places
  the model directly on the requested device.
- No commit hash is pinned. For exact reproducibility, cache the downloaded
  weights and set HF_HUB_OFFLINE=1, or pass revision= once a stable tag is
  confirmed from https://huggingface.co/Qwen/Qwen3-1.7B-Base/commits/main.
"""

from __future__ import annotations

import torch
from loguru import logger
from omegaconf import DictConfig
from transformer_lens import HookedTransformer
from transformers import PreTrainedTokenizerBase

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _resolve_device(requested_device: str) -> str:
    """Resolve the configured device against the current PyTorch build."""
    normalized = requested_device.lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        logger.warning(
            "model.device='cuda' requested, but this PyTorch build has no CUDA support. "
            "Falling back to CPU."
        )
        return "cpu"
    if normalized not in {"auto", "cuda", "cpu"}:
        raise ValueError(
            f"Unsupported device '{requested_device}'. Choose from: ['auto', 'cuda', 'cpu']"
        )
    return normalized


def load_model_and_tokenizer(
    cfg: DictConfig,
) -> tuple[HookedTransformer, PreTrainedTokenizerBase]:
    """
    Load Qwen3-1.7B-Base as a HookedTransformer.

    Args:
        cfg: Hydra DictConfig — uses cfg.model.name, cfg.model.dtype,
             cfg.model.device.

    Returns:
        A (model, tokenizer) tuple. The model is in eval mode on the
        requested device. batch_size=1 is assumed; see module docstring.
    """
    model_name: str = cfg.model.name
    dtype_str: str = cfg.model.dtype
    requested_device: str = cfg.model.device

    if dtype_str not in _DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype '{dtype_str}'. Choose from: {list(_DTYPE_MAP)}"
        )
    torch_dtype = _DTYPE_MAP[dtype_str]
    device = _resolve_device(requested_device)

    if device == "cpu" and dtype_str == "float16":
        logger.warning(
            "float16 on CPU is often unsupported or very slow. "
            "If loading fails or performance is poor, try model.dtype=float32 or bfloat16."
        )

    logger.info(
        "Loading HookedTransformer: "
        f"{model_name} | dtype={dtype_str} | device={device}"
        + (" (resolved from auto)" if requested_device.lower() == "auto" else "")
    )
    logger.info(
        "Using TransformerLens from_pretrained_no_processing with trust_remote_code=True"
    )
    model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        dtype=torch_dtype,
        device=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info(
        f"Model loaded. n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}"
    )

    tokenizer = model.tokenizer
    if tokenizer is None:
        raise RuntimeError(
            f"HookedTransformer returned no tokenizer for '{model_name}'. "
            "This is unexpected — check TransformerLens support for this model."
        )
    return model, tokenizer
