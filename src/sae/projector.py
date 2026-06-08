"""
src/sae/projector.py
====================
Projects a raw residual stream tensor through a SAELens TopKSAE to produce
a sparse feature activation vector.

The core function ``project_topk`` wraps ``sae.encode()``, which internally:
  1. Computes pre-activations:  sae_in @ W_enc + b_enc   → [seq_len, d_sae]
  2. Applies TopK sparsification: keeps the k largest values, zeros the rest.

This produces a dense tensor of shape [seq_len, d_sae] (32 768 features for
Qwen-Scope) where exactly k values per token row are non-zero.

Usage:
    from src.sae.projector import project_topk
    from src.sae.sae_loader import load_sae_for_layer

    sae = load_sae_for_layer("Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50", layer=6)
    acts = project_topk(residual, sae)   # residual: [1, seq_len, 2048]
    # acts: [seq_len, 32768], exactly sae.cfg.k non-zeros per row
"""

from __future__ import annotations

import torch
from sae_lens.saes.topk_sae import TopKSAE


def project_topk(
    residual: torch.Tensor,
    sae: TopKSAE,
) -> torch.Tensor:
    """
    Project a residual stream tensor through a TopK SAE.

    Args:
        residual: Raw activation tensor.  Accepted shapes:
                  - ``[1, seq_len, d_model]``  (as saved by runner.py)
                  - ``[seq_len, d_model]``      (already squeezed)
        sae:      A SAELens ``TopKSAE`` (e.g. from ``load_sae_for_layer``).

    Returns:
        Dense tensor of shape ``[seq_len, d_sae]`` (e.g. [seq_len, 32 768])
        with **exactly** ``sae.cfg.k`` non-zero values per token row.

    Notes:
        - Input is cast to the SAE's dtype (float32) before encoding.
        - No gradients are computed (``torch.inference_mode``).
        - Output stays on the same device as the SAE.
    """
    # Squeeze batch dimension if present
    if residual.ndim == 3:
        if residual.shape[0] != 1:
            raise ValueError(
                f"project_topk expects batch size 1, got shape {tuple(residual.shape)}"
            )
        residual = residual.squeeze(0)  # [seq_len, d_model]

    # Move to SAE device and cast to float32 for numerically stable TopK
    x = residual.to(dtype=torch.float32, device=sae.device)

    with torch.inference_mode():
        acts: torch.Tensor = sae.encode(x)  # [seq_len, d_sae]

    return acts  # still on sae.device
