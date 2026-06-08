"""
tests/test_metrics.py
=====================
Unit tests for src/analysis/metrics.py.

Coverage:
    - jaccard: known inputs, edge cases (identical, disjoint, partial overlap)
    - weighted_jaccard: magnitude weighting, identical/disjoint
    - cosine_sim: orthogonal, parallel, antiparallel, zero vectors
    - l1_distance: known values, symmetry
    - rds / weighted_rds: bounds and complements
    - exclusive_mass_asymmetry: directional drift detection
    - drift_direction_cosine: same/opposite/orthogonal directions
    - resolve_position: last_generated, last_prefix, first_generated
    - compare_pair: integration test, RDS=0 for identical, RDS=1 for disjoint
"""

from __future__ import annotations

import math

import pytest
import torch

from src.analysis.metrics import (
    compare_pair,
    cosine_sim,
    drift_direction_cosine,
    exclusive_mass_asymmetry,
    jaccard,
    l1_distance,
    rds,
    resolve_position,
    weighted_jaccard,
    weighted_rds,
)

# ---------------------------------------------------------------------------
# Fixtures: synthetic sparse vectors
# ---------------------------------------------------------------------------

D_SAE = 32768  # realistic SAE width
K = 50  # typical TopK


def _sparse(indices: list[int], values: list[float], d: int = D_SAE) -> torch.Tensor:
    """Create a 1-D sparse-like dense tensor with given active indices/values."""
    vec = torch.zeros(d)
    for i, v in zip(indices, values):
        vec[i] = v
    return vec


def _topk_uniform(indices: list[int], d: int = D_SAE) -> torch.Tensor:
    """All active features have magnitude 1.0."""
    return _sparse(indices, [1.0] * len(indices), d)


# ---------------------------------------------------------------------------
# jaccard
# ---------------------------------------------------------------------------


class TestJaccard:
    def test_identical_features_returns_one(self):
        idx = list(range(50))
        va = _topk_uniform(idx)
        vb = _topk_uniform(idx)
        assert jaccard(va, vb) == pytest.approx(1.0)

    def test_disjoint_features_returns_zero(self):
        va = _topk_uniform(list(range(50)))
        vb = _topk_uniform(list(range(50, 100)))
        assert jaccard(va, vb) == pytest.approx(0.0)

    def test_partial_overlap_correct_value(self):
        # 25 shared, 25 exclusive each → |intersection|=25, |union|=75
        shared = list(range(25))
        only_a = list(range(25, 50))
        only_b = list(range(50, 75))
        va = _topk_uniform(shared + only_a)
        vb = _topk_uniform(shared + only_b)
        expected = 25 / 75
        assert jaccard(va, vb) == pytest.approx(expected)

    def test_both_zero_returns_zero(self):
        va = torch.zeros(D_SAE)
        vb = torch.zeros(D_SAE)
        assert jaccard(va, vb) == pytest.approx(0.0)

    def test_one_zero_returns_zero(self):
        va = _topk_uniform(list(range(50)))
        vb = torch.zeros(D_SAE)
        assert jaccard(va, vb) == pytest.approx(0.0)

    def test_result_in_unit_interval(self):
        import random

        rng = random.Random(42)
        idx_a = rng.sample(range(D_SAE), K)
        idx_b = rng.sample(range(D_SAE), K)
        va = _topk_uniform(idx_a)
        vb = _topk_uniform(idx_b)
        j = jaccard(va, vb)
        assert 0.0 <= j <= 1.0

    def test_symmetry(self):
        va = _topk_uniform(list(range(50)))
        vb = _topk_uniform(list(range(30, 80)))
        assert jaccard(va, vb) == pytest.approx(jaccard(vb, va))


# ---------------------------------------------------------------------------
# weighted_jaccard
# ---------------------------------------------------------------------------


class TestWeightedJaccard:
    def test_identical_returns_one(self):
        va = _sparse(list(range(50)), [2.0] * 50)
        vb = va.clone()
        assert weighted_jaccard(va, vb) == pytest.approx(1.0)

    def test_disjoint_returns_zero(self):
        va = _sparse(list(range(50)), [1.0] * 50)
        vb = _sparse(list(range(50, 100)), [1.0] * 50)
        assert weighted_jaccard(va, vb) == pytest.approx(0.0)

    def test_same_support_different_magnitudes_less_than_one(self):
        # Same features active but magnitudes differ ⇒ WJ < 1
        idx = list(range(50))
        va = _sparse(idx, [1.0] * 50)
        vb = _sparse(idx, [10.0] * 50)
        wj = weighted_jaccard(va, vb)
        assert wj < 1.0
        assert wj > 0.0
        # WJ = sum(min(1,10)) / sum(max(1,10)) = 50*1 / 50*10 = 0.1
        assert wj == pytest.approx(0.1)

    def test_reduces_to_binary_for_equal_magnitudes(self):
        # With equal magnitudes, WJ = binary Jaccard
        shared = list(range(25))
        only_a = list(range(25, 50))
        only_b = list(range(50, 75))
        va = _topk_uniform(shared + only_a)
        vb = _topk_uniform(shared + only_b)
        wj = weighted_jaccard(va, vb)
        j = jaccard(va, vb)
        assert wj == pytest.approx(j)

    def test_result_in_unit_interval(self):
        import random

        rng = random.Random(7)
        idx_a = rng.sample(range(D_SAE), K)
        vals_a = [rng.uniform(0.1, 5.0) for _ in range(K)]
        idx_b = rng.sample(range(D_SAE), K)
        vals_b = [rng.uniform(0.1, 5.0) for _ in range(K)]
        va = _sparse(idx_a, vals_a)
        vb = _sparse(idx_b, vals_b)
        assert 0.0 <= weighted_jaccard(va, vb) <= 1.0

    def test_both_zero_returns_zero(self):
        assert weighted_jaccard(
            torch.zeros(D_SAE), torch.zeros(D_SAE)
        ) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# cosine_sim
# ---------------------------------------------------------------------------


class TestCosineSim:
    def test_parallel_returns_one(self):
        va = _topk_uniform(list(range(50)))
        vb = va.clone() * 3.0  # same direction, different magnitude
        assert cosine_sim(va, vb) == pytest.approx(1.0, abs=1e-5)

    def test_antiparallel_returns_negative_one(self):
        va = _topk_uniform([0])
        vb = _sparse([0], [-1.0])
        assert cosine_sim(va, vb) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_returns_zero(self):
        va = _sparse([0], [1.0])
        vb = _sparse([1], [1.0])
        assert cosine_sim(va, vb) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_zero(self):
        va = _topk_uniform(list(range(50)))
        vb = torch.zeros(D_SAE)
        assert cosine_sim(va, vb) == pytest.approx(0.0)

    def test_result_in_minus_one_to_one(self):
        import random

        rng = random.Random(13)
        va = torch.tensor([rng.gauss(0, 1) for _ in range(128)])
        vb = torch.tensor([rng.gauss(0, 1) for _ in range(128)])
        cs = cosine_sim(va, vb)
        assert -1.0 <= cs <= 1.0


# ---------------------------------------------------------------------------
# l1_distance
# ---------------------------------------------------------------------------


class TestL1Distance:
    def test_identical_returns_zero(self):
        va = _topk_uniform(list(range(50)))
        assert l1_distance(va, va.clone()) == pytest.approx(0.0)

    def test_known_value(self):
        va = _sparse([0, 1], [3.0, 4.0], d=10)
        vb = _sparse([0, 1], [1.0, 2.0], d=10)
        # |3-1| + |4-2| = 2 + 2 = 4
        assert l1_distance(va, vb) == pytest.approx(4.0)

    def test_symmetry(self):
        va = _sparse([0, 1], [3.0, 4.0], d=10)
        vb = _sparse([0, 2], [1.0, 5.0], d=10)
        assert l1_distance(va, vb) == pytest.approx(l1_distance(vb, va))

    def test_non_negative(self):
        import random

        rng = random.Random(99)
        va = torch.tensor([rng.gauss(0, 1) for _ in range(256)])
        vb = torch.tensor([rng.gauss(0, 1) for _ in range(256)])
        assert l1_distance(va, vb) >= 0.0


# ---------------------------------------------------------------------------
# rds and weighted_rds
# ---------------------------------------------------------------------------


class TestRDS:
    def test_rds_identical_is_zero(self):
        idx = list(range(50))
        va = _topk_uniform(idx)
        assert rds(va, va.clone()) == pytest.approx(0.0)

    def test_rds_disjoint_is_one(self):
        va = _topk_uniform(list(range(50)))
        vb = _topk_uniform(list(range(50, 100)))
        assert rds(va, vb) == pytest.approx(1.0)

    def test_rds_complement_of_jaccard(self):
        va = _topk_uniform(list(range(50)))
        vb = _topk_uniform(list(range(30, 80)))
        assert rds(va, vb) == pytest.approx(1.0 - jaccard(va, vb))

    def test_weighted_rds_identical_is_zero(self):
        va = _sparse(list(range(50)), [2.5] * 50)
        assert weighted_rds(va, va.clone()) == pytest.approx(0.0)

    def test_weighted_rds_disjoint_is_one(self):
        va = _sparse(list(range(50)), [1.0] * 50)
        vb = _sparse(list(range(50, 100)), [1.0] * 50)
        assert weighted_rds(va, vb) == pytest.approx(1.0)

    def test_weighted_rds_complement_of_weighted_jaccard(self):
        va = _sparse(list(range(50)), [1.0, 2.0] * 25)
        vb = _sparse(list(range(25, 75)), [0.5] * 50)
        assert weighted_rds(va, vb) == pytest.approx(1.0 - weighted_jaccard(va, vb))


# ---------------------------------------------------------------------------
# exclusive_mass_asymmetry
# ---------------------------------------------------------------------------


class TestExclusiveMassAsymmetry:
    def test_identical_returns_zero(self):
        va = _topk_uniform(list(range(50)))
        assert exclusive_mass_asymmetry(va, va.clone()) == pytest.approx(0.0)

    def test_asymmetry_positive_when_b_gains_more(self):
        # va has feature 0 (mass 1.0), vb has feature 1 (mass 100.0)
        va = _sparse([0], [1.0], d=10)
        vb = _sparse([1], [100.0], d=10)
        # excl_b_mass=100, excl_a_mass=1 → asym = (100-1)/(101) > 0
        asym = exclusive_mass_asymmetry(va, vb)
        assert asym > 0.0

    def test_asymmetry_negative_when_b_loses_more(self):
        va = _sparse([0], [100.0], d=10)
        vb = _sparse([1], [1.0], d=10)
        asym = exclusive_mass_asymmetry(va, vb)
        assert asym < 0.0

    def test_result_in_minus_one_to_one(self):
        va = _sparse(list(range(50)), [float(i) for i in range(1, 51)])
        vb = _sparse(list(range(25, 75)), [0.5] * 50)
        asym = exclusive_mass_asymmetry(va, vb)
        assert -1.0 <= asym <= 1.0


# ---------------------------------------------------------------------------
# drift_direction_cosine
# ---------------------------------------------------------------------------


class TestDriftDirectionCosine:
    def test_same_direction_returns_one(self):
        ref = torch.zeros(D_SAE)
        # Both drift along the same direction: [1, 0, 0, ...]
        drift_vec = _sparse([0], [1.0])
        b = ref + drift_vec
        c = ref + drift_vec * 2
        ddc = drift_direction_cosine(ref, b, c)
        assert ddc == pytest.approx(1.0, abs=1e-5)

    def test_opposite_direction_returns_neg_one(self):
        ref = torch.zeros(D_SAE)
        b = _sparse([0], [1.0])
        c = _sparse([0], [-1.0])
        ddc = drift_direction_cosine(ref, b, c)
        assert ddc == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_returns_zero(self):
        ref = torch.zeros(D_SAE)
        b = _sparse([0], [1.0])
        c = _sparse([1], [1.0])
        ddc = drift_direction_cosine(ref, b, c)
        assert ddc == pytest.approx(0.0, abs=1e-5)

    def test_zero_delta_returns_zero(self):
        ref = _topk_uniform(list(range(50)))
        b = ref.clone()  # no drift
        c = _topk_uniform(list(range(100, 150)))
        ddc = drift_direction_cosine(ref, b, c)
        assert ddc == pytest.approx(0.0)

    def test_result_in_minus_one_to_one(self):
        import random

        rng = random.Random(42)
        ref = torch.tensor([rng.gauss(0, 1) for _ in range(256)])
        b = torch.tensor([rng.gauss(0, 1) for _ in range(256)])
        c = torch.tensor([rng.gauss(0, 1) for _ in range(256)])
        ddc = drift_direction_cosine(ref, b, c)
        assert -1.0 <= ddc <= 1.0


# ---------------------------------------------------------------------------
# resolve_position
# ---------------------------------------------------------------------------


class TestResolvePosition:
    def _make_feat(self, seq_len: int) -> torch.Tensor:
        return torch.zeros(seq_len, D_SAE)

    def test_last_generated(self):
        feat = self._make_feat(100)
        assert resolve_position(feat, "last_generated") == 99

    def test_last_prefix_within_bounds(self):
        feat = self._make_feat(150)
        pos = resolve_position(feat, "last_prefix", prompt_len=50)
        assert pos == 49

    def test_last_prefix_clamps_to_seq_len(self):
        feat = self._make_feat(10)
        pos = resolve_position(feat, "last_prefix", prompt_len=100)
        assert pos == 9  # clamped to seq_len - 1

    def test_first_generated(self):
        feat = self._make_feat(100)
        pos = resolve_position(feat, "first_generated", prompt_len=50)
        assert pos == 50

    def test_first_generated_clamps(self):
        feat = self._make_feat(5)
        pos = resolve_position(feat, "first_generated", prompt_len=10)
        assert pos == 4  # clamped to seq_len - 1

    def test_last_prefix_requires_prompt_len(self):
        feat = self._make_feat(100)
        with pytest.raises(ValueError, match="prompt_len"):
            resolve_position(feat, "last_prefix")

    def test_first_generated_requires_prompt_len(self):
        feat = self._make_feat(100)
        with pytest.raises(ValueError, match="prompt_len"):
            resolve_position(feat, "first_generated")

    def test_unknown_position_raises(self):
        feat = self._make_feat(100)
        with pytest.raises(ValueError, match="Unknown"):
            resolve_position(feat, "middle_of_nowhere")


# ---------------------------------------------------------------------------
# compare_pair — integration
# ---------------------------------------------------------------------------


class TestComparePair:
    def _make_feat_2d(
        self, seq_len: int, idx: list[int], val: float = 1.0
    ) -> torch.Tensor:
        feat = torch.zeros(seq_len, D_SAE)
        feat[-1, idx] = val
        return feat

    def test_identical_feat_rds_zero(self):
        idx = list(range(50))
        feat = self._make_feat_2d(10, idx)
        result = compare_pair(feat, feat, pos_a=9, pos_b=9)
        assert result["rds"] == pytest.approx(0.0)
        assert result["jaccard"] == pytest.approx(1.0)
        assert result["weighted_jaccard"] == pytest.approx(1.0)
        assert result["weighted_rds"] == pytest.approx(0.0)
        assert result["cosine_sim"] == pytest.approx(1.0, abs=1e-5)
        assert result["l1_distance"] == pytest.approx(0.0)

    def test_disjoint_feat_rds_one(self):
        feat_a = self._make_feat_2d(10, list(range(50)))
        feat_b = self._make_feat_2d(10, list(range(50, 100)))
        result = compare_pair(feat_a, feat_b, pos_a=9, pos_b=9)
        assert result["rds"] == pytest.approx(1.0)
        assert result["jaccard"] == pytest.approx(0.0)
        assert result["weighted_rds"] == pytest.approx(1.0)
        assert result["cosine_sim"] == pytest.approx(0.0)

    def test_returns_all_expected_keys(self):
        feat_a = self._make_feat_2d(5, list(range(50)))
        feat_b = self._make_feat_2d(5, list(range(25, 75)))
        result = compare_pair(feat_a, feat_b, pos_a=4, pos_b=4)
        expected_keys = {
            "jaccard",
            "weighted_jaccard",
            "cosine_sim",
            "l1_distance",
            "rds",
            "weighted_rds",
            "exclusive_mass_asymmetry",
        }
        assert expected_keys.issubset(result.keys())

    def test_different_positions(self):
        # Verify pos_a and pos_b are used independently
        feat_a = torch.zeros(5, D_SAE)
        feat_b = torch.zeros(8, D_SAE)
        feat_a[2, list(range(50))] = 1.0
        feat_b[7, list(range(50))] = 1.0
        result = compare_pair(feat_a, feat_b, pos_a=2, pos_b=7)
        assert result["jaccard"] == pytest.approx(1.0)

    def test_sanity_clean_vs_clean_all_zero_rds(self):
        """Task 4.7 equivalent: identical features → RDS = 0."""
        import random

        rng = random.Random(100)
        idx = rng.sample(range(D_SAE), K)
        vals = [rng.uniform(0.5, 3.0) for _ in range(K)]
        feat = self._make_feat_2d(20, idx, val=0.0)
        for i, v in zip(idx, vals):
            feat[-1, i] = v

        result = compare_pair(feat, feat.clone(), pos_a=19, pos_b=19)
        assert result["rds"] == pytest.approx(0.0, abs=1e-6)
        assert result["weighted_rds"] == pytest.approx(0.0, abs=1e-6)
