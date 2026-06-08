"""
src/sae/sae_loader.py
=====================
Downloads and wraps official Qwen-Scope Top-K SAE checkpoints as SAELens
TopKSAE objects, enabling the rest of Phase 3 to call sae.encode() for
sparse feature extraction.

HuggingFace checkpoint format (Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50):
  Each  layer{N}.sae.pt  is a plain Python dict with four tensors:
    W_enc  [d_sae, d_in]   encoder weight  (Qwen layout)
    b_enc  [d_sae]         encoder bias
    W_dec  [d_sae, d_in]   decoder weight
    b_dec  [d_in]          decoder bias

SAELens TopKSAE layout (what the library expects):
    W_enc  [d_in, d_sae]   ← TRANSPOSED relative to Qwen
    b_enc  [d_sae]
    W_dec  [d_sae, d_in]   ← same as Qwen
    b_dec  [d_in]

So the only adjustment needed when loading is W_enc.T.

Usage:
    from src.sae.sae_loader import load_sae_for_layer

    sae = load_sae_for_layer(
        hf_repo="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
        layer=6,
        top_k=50,
        device="cpu",
    )
    # sae is a SAELens TopKSAE; use sae.encode(residual) for feature extraction.
"""

from __future__ import annotations

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from loguru import logger
from sae_lens.saes.sae import SAEMetadata
from sae_lens.saes.topk_sae import TopKSAE, TopKSAEConfig


def load_sae_for_layer(
    hf_repo: str,
    layer: int,
    top_k: int = 50,
    device: str = "cpu",
) -> TopKSAE:
    """
    Download ``layer{N}.sae.pt`` from ``hf_repo`` and return a SAELens
    ``TopKSAE`` with the Qwen weights loaded into it.

    Args:
        hf_repo:  HuggingFace repo ID, e.g.
                  ``"Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50"``.
        layer:    Transformer layer index (0-based).
        top_k:    Number of features to keep non-zero per token position.
                  Must match the training configuration (50 for the
                  L0_50 checkpoint family).
        device:   Torch device string (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        A ``TopKSAE`` in eval mode with Qwen weights loaded.
        Call ``sae.encode(residual)`` to get sparse feature activations.

    Raises:
        KeyError: If the downloaded ``.pt`` dict is missing expected keys.
        RuntimeError: If weight shapes are inconsistent.
    """
    filename = f"layer{layer}.sae.pt"
    logger.info(f"Downloading {filename} from {hf_repo} …")
    local_path: str = hf_hub_download(repo_id=hf_repo, filename=filename)
    logger.debug(f"Cached at: {local_path}")

    checkpoint = torch.load(local_path, map_location="cpu", weights_only=True)

    # --- Validate expected keys ---
    for key in ("W_enc", "b_enc", "W_dec", "b_dec"):
        if key not in checkpoint:
            raise KeyError(f"Checkpoint {filename} missing key '{key}'.")

    W_enc_qwen: torch.Tensor = checkpoint["W_enc"]  # [d_sae, d_in]
    b_enc: torch.Tensor = checkpoint["b_enc"]  # [d_sae]
    W_dec: torch.Tensor = checkpoint["W_dec"]  # [d_sae, d_in]
    b_dec: torch.Tensor = checkpoint["b_dec"]  # [d_in]

    d_sae, d_in = W_enc_qwen.shape
    logger.debug(f"Layer {layer}: d_in={d_in}, d_sae={d_sae}, k={top_k}")

    # --- Build SAELens config ---
    cfg = TopKSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=top_k,
        apply_b_dec_to_input=False,
        normalize_activations="none",
        reshape_activations="none",
        dtype="float32",
        device="cpu",  # Place on CPU first; move to target device after weight load
        metadata=SAEMetadata(
            model_name="qwen3-1.7b",
            hook_name=f"blocks.{layer}.hook_resid_post",
            hook_layer=layer,
        ),
    )

    sae = TopKSAE(cfg)

    # --- Load weights (note: SAELens W_enc is [d_in, d_sae] = Qwen W_enc.T) ---
    with torch.no_grad():
        sae.W_enc.data = W_enc_qwen.T.contiguous().to(torch.float32)
        sae.b_enc.data = b_enc.to(torch.float32)
        sae.W_dec.data = W_dec.T.contiguous().to(torch.float32)
        sae.b_dec.data = b_dec.to(torch.float32)

    sae = sae.to(device)
    sae.eval()
    logger.info(f"SAE layer {layer} loaded — W_enc{tuple(sae.W_enc.shape)}, k={top_k}")
    return sae


def load_saes_for_layers(
    hf_repo: str,
    layers: list[int],
    top_k: int = 50,
    device: str = "cpu",
) -> dict[int, TopKSAE]:
    """
    Convenience wrapper: load SAEs for multiple layers at once.

    Returns:
        Dict mapping layer_idx → TopKSAE.
    """
    return {
        layer: load_sae_for_layer(hf_repo, layer, top_k=top_k, device=device)
        for layer in layers
    }
