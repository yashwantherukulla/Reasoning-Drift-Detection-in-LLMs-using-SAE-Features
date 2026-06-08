"""
tests/test_activation_extractor.py
====================================
Unit tests for capture_prefill_activations() in src/inference/activation_extractor.py.

These tests use a tiny randomly-initialized HookedTransformer (no network
access, no large weights) to exercise the TL-based activation capture API.

HookedTransformerConfig lets us build an arbitrarily small model that still
exercises the full run_with_cache() / hook naming / tensor shape path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

# ---------------------------------------------------------------------------
# Tiny HookedTransformer — randomly initialized, no downloads
# ---------------------------------------------------------------------------

STUB_D_MODEL = 16
STUB_N_LAYERS = 3
STUB_N_HEADS = 2
STUB_D_HEAD = 8
STUB_D_MLP = 64
STUB_N_CTX = 32
STUB_D_VOCAB = 200
HOOK_LAYERS = [0, 2]
BATCH_SIZE = 1
SEQ_LEN = 5


@pytest.fixture(scope="module")
def stub_model() -> HookedTransformer:
    """
    A minimal HookedTransformer with random weights. Scope=module so it is
    constructed once per test session — construction is cheap but not instant.
    """
    cfg = HookedTransformerConfig(
        d_model=STUB_D_MODEL,
        n_layers=STUB_N_LAYERS,
        n_heads=STUB_N_HEADS,
        d_head=STUB_D_HEAD,
        d_mlp=STUB_D_MLP,
        n_ctx=STUB_N_CTX,
        d_vocab=STUB_D_VOCAB,
        act_fn="gelu_new",
        normalization_type="LN",
        device="cpu",
        dtype=torch.float32,
    )
    return HookedTransformer(cfg).eval()


@pytest.fixture()
def stub_tokens() -> torch.Tensor:
    """Random token IDs of shape [1, SEQ_LEN]."""
    return torch.randint(0, STUB_D_VOCAB, (BATCH_SIZE, SEQ_LEN))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_capture_returns_all_requested_layers(
    stub_model: HookedTransformer, stub_tokens: torch.Tensor
) -> None:
    """captured dict must have an entry for every requested layer index."""
    from src.inference.activation_extractor import capture_prefill_activations

    captured = capture_prefill_activations(stub_model, stub_tokens, HOOK_LAYERS)

    assert set(captured.keys()) == set(HOOK_LAYERS), (
        f"Expected keys {HOOK_LAYERS}, got {list(captured.keys())}"
    )


def test_captured_tensor_shape(
    stub_model: HookedTransformer, stub_tokens: torch.Tensor
) -> None:
    """Captured tensors must have shape [batch, seq_len, d_model] on CPU."""
    from src.inference.activation_extractor import capture_prefill_activations

    captured = capture_prefill_activations(stub_model, stub_tokens, HOOK_LAYERS)

    for layer_idx in HOOK_LAYERS:
        t = captured[layer_idx]
        assert t.shape == (BATCH_SIZE, SEQ_LEN, STUB_D_MODEL), (
            f"Layer {layer_idx}: expected ({BATCH_SIZE}, {SEQ_LEN}, {STUB_D_MODEL}), "
            f"got {tuple(t.shape)}"
        )
        assert t.device.type == "cpu", (
            f"Layer {layer_idx}: tensor should be on CPU, got {t.device}"
        )


def test_capture_is_detached(
    stub_model: HookedTransformer, stub_tokens: torch.Tensor
) -> None:
    """Captured tensors must be detached from the computation graph."""
    from src.inference.activation_extractor import capture_prefill_activations

    captured = capture_prefill_activations(stub_model, stub_tokens, HOOK_LAYERS)

    for layer_idx, t in captured.items():
        assert not t.requires_grad, f"Layer {layer_idx}: tensor should not require grad"


def test_two_calls_are_independent(stub_model: HookedTransformer) -> None:
    """
    Two calls with different tokens must produce independent results.
    There is no shared mutable state between calls — unlike the old
    register_residual_hooks approach which reused a single captured dict.
    """
    from src.inference.activation_extractor import capture_prefill_activations

    # Use deterministic, non-identical token sequences
    tokens_a = torch.arange(0, SEQ_LEN, dtype=torch.long).unsqueeze(0)
    tokens_b = torch.arange(SEQ_LEN, 2 * SEQ_LEN, dtype=torch.long).unsqueeze(0)

    cap_a = capture_prefill_activations(stub_model, tokens_a, HOOK_LAYERS)
    cap_b = capture_prefill_activations(stub_model, tokens_b, HOOK_LAYERS)

    for layer_idx in HOOK_LAYERS:
        assert not torch.allclose(cap_a[layer_idx], cap_b[layer_idx]), (
            f"Layer {layer_idx}: different inputs should produce different activations"
        )


def test_activation_saved_to_disk(
    stub_model: HookedTransformer, stub_tokens: torch.Tensor, tmp_path: Path
) -> None:
    """Saved .pt files must exist, have float16 dtype, and correct shape."""
    from src.inference.activation_extractor import capture_prefill_activations

    captured = capture_prefill_activations(stub_model, stub_tokens, HOOK_LAYERS)

    # Mimic runner.py: cast to float16 and save
    problem_id, condition = "test_001", "clean"
    for layer_idx, tensor in captured.items():
        save_path = tmp_path / f"{problem_id}_{condition}_{layer_idx}.pt"
        torch.save(tensor.to(torch.float16), save_path)

    for layer_idx in HOOK_LAYERS:
        path = tmp_path / f"{problem_id}_{condition}_{layer_idx}.pt"
        assert path.exists(), f"Expected file not found: {path}"
        loaded = torch.load(path, weights_only=True)
        assert loaded.dtype == torch.float16
        assert loaded.shape == (BATCH_SIZE, SEQ_LEN, STUB_D_MODEL)


def test_invalid_layer_raises(
    stub_model: HookedTransformer, stub_tokens: torch.Tensor
) -> None:
    """Requesting an out-of-range layer index must raise ValueError."""
    from src.inference.activation_extractor import capture_prefill_activations

    with pytest.raises(ValueError, match="out of range"):
        capture_prefill_activations(stub_model, stub_tokens, [STUB_N_LAYERS + 1])
