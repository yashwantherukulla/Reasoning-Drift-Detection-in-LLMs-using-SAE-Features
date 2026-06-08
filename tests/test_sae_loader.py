"""
tests/test_sae_loader.py
========================
Unit tests for Phase 3 — SAE Projection.

Tests are fully offline: they create synthetic checkpoints that mimic the
Qwen-Scope format and verify that:
  - Weight shapes are correct after loading into SAELens TopKSAE.
  - project_topk returns the expected output shape.
  - Exactly k non-zero values per token position in the output.
  - _parse_stem correctly splits various filename patterns.
  - feature_extractor skips missing layers and existing outputs.

No internet access is needed; HuggingFace downloads are mocked.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.sae.feature_extractor import _parse_stem
from src.sae.projector import project_topk
from src.sae.sae_loader import load_sae_for_layer

# ── Fixture dimensions ─────────────────────────────────────────────────────
# Use small dimensions for unit tests to keep memory and runtime minimal.
# Shape relationships are identical to the real Qwen checkpoint.
D_MODEL = 64  # real: 2048
D_SAE = 256  # real: 32768
TOP_K = 10  # real: 50

# Real Qwen-Scope expected shapes (used in shape assertion docstrings / notes)
_REAL_D_MODEL = 2048
_REAL_D_SAE = 32768


# ── Helpers ─────────────────────────────────────────────────────────────────


def _make_fake_checkpoint(d_sae: int = D_SAE, d_in: int = D_MODEL) -> dict:
    """Return a dict matching the Qwen-Scope .pt format (random weights)."""
    return {
        "W_enc": torch.randn(d_sae, d_in),  # [d_sae, d_in]
        "b_enc": torch.zeros(d_sae),  # [d_sae]
        "W_dec": torch.randn(d_sae, d_in),  # [d_sae, d_in]
        "b_dec": torch.zeros(d_in),  # [d_in]
    }


def _load_sae_from_fake_ckpt(
    layer: int = 6,
    top_k: int = TOP_K,
    d_sae: int = D_SAE,
    d_in: int = D_MODEL,
):
    """
    Create a fake checkpoint file and load it via load_sae_for_layer,
    bypassing the HuggingFace download.
    """
    ckpt = _make_fake_checkpoint(d_sae=d_sae, d_in=d_in)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp_path = f.name
        torch.save(ckpt, tmp_path)

    with patch(
        "src.sae.sae_loader.hf_hub_download",
        return_value=tmp_path,
    ):
        sae = load_sae_for_layer(
            hf_repo="Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
            layer=layer,
            top_k=top_k,
            device="cpu",
        )

    return sae


# ── Shape tests ─────────────────────────────────────────────────────────────


class TestSAELoaderShapes:
    """Verify SAELens weight tensor shapes after loading Qwen checkpoint."""

    def test_W_enc_shape(self):
        """SAELens expects W_enc as [d_in, d_sae] (Qwen stores [d_sae, d_in]).
        Real shapes: [{_REAL_D_MODEL}, {_REAL_D_SAE}].
        """
        sae = _load_sae_from_fake_ckpt()
        assert sae.W_enc.shape == (D_MODEL, D_SAE), (
            f"Expected W_enc [{D_MODEL}, {D_SAE}], got {tuple(sae.W_enc.shape)}"
        )

    def test_b_enc_shape(self):
        """b_enc shape: [d_sae]. Real: [{_REAL_D_SAE}]."""
        sae = _load_sae_from_fake_ckpt()
        assert sae.b_enc.shape == (D_SAE,), (
            f"Expected b_enc [{D_SAE}], got {tuple(sae.b_enc.shape)}"
        )

    def test_W_dec_shape(self):
        """W_dec is [d_sae, d_in] in both Qwen and SAELens.
        Real shapes: [{_REAL_D_SAE}, {_REAL_D_MODEL}].
        """
        sae = _load_sae_from_fake_ckpt()
        assert sae.W_dec.shape == (D_SAE, D_MODEL), (
            f"Expected W_dec [{D_SAE}, {D_MODEL}], got {tuple(sae.W_dec.shape)}"
        )

    def test_b_dec_shape(self):
        """b_dec shape: [d_in]. Real: [{_REAL_D_MODEL}]."""
        sae = _load_sae_from_fake_ckpt()
        assert sae.b_dec.shape == (D_MODEL,), (
            f"Expected b_dec [{D_MODEL}], got {tuple(sae.b_dec.shape)}"
        )

    def test_cfg_k(self):
        sae = _load_sae_from_fake_ckpt(top_k=50)
        assert sae.cfg.k == 50

    def test_eval_mode(self):
        """SAE must be in eval mode after loading."""
        sae = _load_sae_from_fake_ckpt()
        assert not sae.training, "SAE should be in eval mode"


# ── project_topk tests ──────────────────────────────────────────────────────


class TestProjectTopK:
    """Verify project_topk output properties."""

    @pytest.fixture
    def sae(self):
        return _load_sae_from_fake_ckpt(top_k=TOP_K)

    def test_output_shape_batch1(self, sae):
        """Input [1, seq_len, d_model] → output [seq_len, d_sae]."""
        seq_len = 32
        residual = torch.randn(1, seq_len, D_MODEL)
        acts = project_topk(residual, sae)
        assert acts.shape == (seq_len, D_SAE)

    def test_output_shape_no_batch(self, sae):
        """Input [seq_len, d_model] → output [seq_len, d_sae]."""
        seq_len = 15
        residual = torch.randn(seq_len, D_MODEL)
        acts = project_topk(residual, sae)
        assert acts.shape == (seq_len, D_SAE)

    def test_exactly_k_nonzeros_per_token(self, sae):
        """Every token row must have exactly TOP_K non-zero features."""
        seq_len = 20
        residual = torch.randn(1, seq_len, D_MODEL)
        acts = project_topk(residual, sae)
        nnz_per_token = (acts != 0).sum(dim=-1)  # [seq_len]
        assert (nnz_per_token == TOP_K).all(), (
            f"Expected {TOP_K} non-zeros per token; got min={nnz_per_token.min()}, "
            f"max={nnz_per_token.max()}"
        )

    def test_invalid_batch_size_raises(self, sae):
        """Batch size > 1 should raise ValueError."""
        residual = torch.randn(3, 10, D_MODEL)
        with pytest.raises(ValueError, match="batch size 1"):
            project_topk(residual, sae)

    def test_output_dtype(self, sae):
        """Output should be float32 (SAE operates in float32)."""
        residual = torch.randn(1, 8, D_MODEL, dtype=torch.float16)
        acts = project_topk(residual, sae)
        assert acts.dtype == torch.float32


# ── _parse_stem tests ────────────────────────────────────────────────────────


class TestParseStem:
    """Verify filename stem parsing for the feature extractor."""

    @pytest.mark.parametrize(
        "stem, expected",
        [
            ("arith_5_clean_6", ("arith_5_clean", 6)),
            ("arith_5_clean_27", ("arith_5_clean", 27)),
            ("gsm8k_10_helpful_hint_12", ("gsm8k_10_helpful_hint", 12)),
            ("logic_23_misleading_hint_20", ("logic_23_misleading_hint", 20)),
            ("symb_3_clean_18", ("symb_3_clean", 18)),
        ],
    )
    def test_valid_stems(self, stem, expected):
        assert _parse_stem(stem) == expected

    @pytest.mark.parametrize(
        "bad_stem",
        [
            "no_layer",  # no numeric suffix
            "layer_abc",  # non-integer suffix
            "single",  # only one component
            "",  # empty string
        ],
    )
    def test_invalid_stems(self, bad_stem):
        assert _parse_stem(bad_stem) is None


# ── feature extractor integration test ──────────────────────────────────────


class TestFeatureExtractorIntegration:
    """
    Smoke test: run extract_and_save_features against a small synthetic
    directory, using a mocked SAE loader to avoid HuggingFace downloads.
    """

    def test_writes_feature_files(self, tmp_path):
        """Feature files should be written with shape [seq_len, d_sae]."""
        from unittest.mock import MagicMock

        from omegaconf import OmegaConf

        # Build minimal config
        cfg = OmegaConf.create(
            {
                "sae": {
                    "hf_repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
                    "layers_to_analyze": [6],
                    "top_k_features": TOP_K,
                },
                "activations": {
                    "raw_dir": str(tmp_path / "raw"),
                    "sae_dir": str(tmp_path / "sae"),
                },
            }
        )

        # Create a fake raw activation file
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        seq_len = 12
        fake_residual = torch.randn(1, seq_len, D_MODEL, dtype=torch.float16)
        torch.save(fake_residual, raw_dir / "arith_5_clean_6.pt")

        # Mock load_saes_for_layers to return a real (fake-weight) SAE
        fake_sae = _load_sae_from_fake_ckpt(layer=6, top_k=TOP_K)

        from src.sae import feature_extractor

        with patch.object(
            feature_extractor,
            "load_saes_for_layers",
            return_value={6: fake_sae},
        ):
            from src.sae.feature_extractor import extract_and_save_features

            n = extract_and_save_features(cfg)

        assert n == 1

        out_file = tmp_path / "sae" / "arith_5_clean_6.pt"
        assert out_file.exists(), "Output feature file not found"

        acts = torch.load(out_file, weights_only=True)
        assert acts.shape == (seq_len, D_SAE), (
            f"Expected [{seq_len}, {D_SAE}], got {tuple(acts.shape)}"
        )
        assert acts.dtype == torch.float16

    def test_skips_existing_output(self, tmp_path):
        """Already-written output files should not be overwritten."""
        from omegaconf import OmegaConf

        cfg = OmegaConf.create(
            {
                "sae": {
                    "hf_repo": "Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50",
                    "layers_to_analyze": [6],
                    "top_k_features": TOP_K,
                },
                "activations": {
                    "raw_dir": str(tmp_path / "raw"),
                    "sae_dir": str(tmp_path / "sae"),
                },
            }
        )

        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        sae_dir = tmp_path / "sae"
        sae_dir.mkdir()

        fake_residual = torch.randn(1, 8, D_MODEL, dtype=torch.float16)
        torch.save(fake_residual, raw_dir / "arith_5_clean_6.pt")

        # Pre-create the output file with a sentinel
        sentinel = torch.zeros(1)
        torch.save(sentinel, sae_dir / "arith_5_clean_6.pt")

        fake_sae = _load_sae_from_fake_ckpt(layer=6, top_k=TOP_K)

        from src.sae import feature_extractor

        with patch.object(
            feature_extractor,
            "load_saes_for_layers",
            return_value={6: fake_sae},
        ):
            from src.sae.feature_extractor import extract_and_save_features

            n = extract_and_save_features(cfg)

        # Nothing should have been written
        assert n == 0
        # Sentinel should still be there unchanged
        reloaded = torch.load(sae_dir / "arith_5_clean_6.pt", weights_only=True)
        assert reloaded.shape == (1,)
