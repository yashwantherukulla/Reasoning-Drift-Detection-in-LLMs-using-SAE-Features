"""
src/inference/activation_extractor.py
=======================================
Captures post-block residual stream activations using TransformerLens's
run_with_cache() API.

Why TransformerLens over raw PyTorch hooks:
- run_with_cache() is a single forward pass: activations are captured and
  returned cleanly, with zero hook lifecycle to manage (no register, no
  remove, no accumulation risk).
- No decode-step overwrite problem. The old PyTorch hook approach would
  overwrite captured[layer] on every autoregressive step because Qwen3
  always produces [1, 1, d_model] tensors (ndim=3) even for single-token
  decode steps — identical shape to the prefill pass.
- Hook names (blocks.{i}.hook_resid_post) are stable strings that match
  the convention used by the Qwen3 residual-stream SAEs.
- Hidden size is read from model.cfg.d_model — no hard-coded constant that
  would silently break when switching to a different model variant.

Usage pattern in runner.py:
    tokens = model.to_tokens(prompt_text, prepend_bos=False)   # prefill
    captured = capture_prefill_activations(model, tokens, layers)
    # ... separately: model.generate(tokens, ...) for text answer
"""

from __future__ import annotations

import torch
from loguru import logger
from torch import Tensor
from transformer_lens import HookedTransformer


def capture_prefill_activations(
    model: HookedTransformer,
    tokens: Tensor,  # [1, prompt_len] int64, on model's device
    layers: list[int],
) -> dict[int, Tensor]:
    """
    Run a single forward pass and return post-block residual stream
    activations at the requested layer indices.

    The forward pass sees only the prompt tokens — completely decoupled from
    generation. This is clean prefill-only capture.

    Args:
        model:   HookedTransformer in eval mode.
        tokens:  Token ID tensor of shape [1, prompt_len], on model device.
        layers:  Layer indices to capture (e.g. [6, 12, 18, 20, 24, 27]).

    Returns:
        Dict mapping layer_idx → CPU Tensor of shape [1, prompt_len, d_model].
        Caller casts to float16 before saving (see runner.py).

    Raises:
        ValueError: if any layer index is out of range for the model.
    """
    n_layers = model.cfg.n_layers
    for i in layers:
        if not (0 <= i < n_layers):
            raise ValueError(
                f"Layer index {i} is out of range for model with {n_layers} layers."
            )

    hook_names = {i: f"blocks.{i}.hook_resid_post" for i in layers}
    target_names = set(hook_names.values())

    with torch.inference_mode():
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: name in target_names,
            prepend_bos=False,
        )

    d_model = model.cfg.d_model
    captured: dict[int, Tensor] = {}
    for i, hook_name in hook_names.items():
        tensor = cache[hook_name]
        assert tensor.shape[-1] == d_model, (
            f"Layer {i}: expected d_model={d_model}, got {tensor.shape[-1]}"
        )
        captured[i] = tensor.detach().cpu()
        logger.debug(f"Captured layer {i}: shape={tuple(tensor.shape)}")

    return captured
