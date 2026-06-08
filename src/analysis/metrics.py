"""
src/analysis/metrics.py
=======================
Core metric functions for comparing SAE feature activation vectors between
different prompt conditions at a specific token position.

All metric functions accept **1-D** float tensors ``[d_sae]`` — i.e. the
feature vector already sliced to a single token position (``feat[pos]``).

Metric stack
------------
Primary metrics (as defined in plan.md / phases.md):
    jaccard           Binary Jaccard over active feature index sets
    rds               Reasoning Divergence Score = 1 − Jaccard

Improved metrics (added beyond the plan):
    weighted_jaccard  Soft Jaccard weighted by activation magnitudes.
                      Strictly more informative than binary Jaccard for TopK
                      SAEs because it captures magnitude-recalibration of
                      shared features in addition to structural drift.
                      Bounded [0, 1]; RDS analog = 1 − weighted_jaccard.
    cosine_sim        Cosine similarity of full feature vectors.
    l1_distance       L1 (Manhattan) distance — captures total activation
                      budget shift, complementary to cosine.
    exclusive_mass_asymmetry
                      Signed asymmetry of activation mass in features unique
                      to each condition.  Positive ⇒ cond_b brought in new
                      features with more total mass than it dropped; captures
                      *directional* structural drift.

3-condition metric:
    drift_direction_cosine
                      Cosine between (feat_b − feat_ref) and (feat_c − feat_ref).
                      +1 ⇒ both conditions drift the same way from the reference;
                      −1 ⇒ they push in opposite directions.  Directly answers:
                      "do helpful_hint and misleading_hint displace the clean
                      representation in the same or opposite direction?"

Utility:
    resolve_position  Convert a named position string to an integer index.
    compare_pair      Compute all pairwise metrics in one call.

Background note on TopK SAEs
-----------------------------
With k = 50 non-zeros and d_sae = 32 768, the expected binary Jaccard under
random activation is ≈ 0.00076 (≈ 0.076 shared features on average).
Any Jaccard > ~0.01 is therefore ~13× above chance and strongly meaningful.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Position resolver
# ---------------------------------------------------------------------------


def resolve_position(
    feat: torch.Tensor,
    token_position: str,
    prompt_len: int | None = None,
) -> int:
    """
    Convert a named token position to a concrete integer index for *feat*.

    Args:
        feat:           Feature tensor ``[seq_len, d_sae]``.
        token_position: One of:
                        ``"last_generated"`` — last token of the full sequence
                            (index = seq_len − 1).
                        ``"last_prefix"`` — last token of the prompt before
                            generation starts (index = prompt_len − 1).
                            Requires *prompt_len*.
                        ``"first_generated"`` — first generated token
                            (index = prompt_len).
                            Requires *prompt_len*.
        prompt_len:     Number of prompt tokens (required for prefix-relative
                        positions).

    Returns:
        Integer index in [0, seq_len − 1].

    Raises:
        ValueError: For unknown position names or missing prompt_len.
    """
    seq_len = feat.shape[0]

    if token_position == "last_generated":
        return seq_len - 1

    if token_position in ("last_prefix", "first_generated"):
        if prompt_len is None:
            raise ValueError(
                f"token_position='{token_position}' requires prompt_len to be provided."
            )
        if token_position == "last_prefix":
            return min(prompt_len - 1, seq_len - 1)
        else:  # first_generated
            return min(prompt_len, seq_len - 1)

    raise ValueError(
        f"Unknown token_position: {token_position!r}. "
        "Choose from: 'last_generated', 'last_prefix', 'first_generated'."
    )


# ---------------------------------------------------------------------------
# 1-D vector metrics  (all take feat_a[pos], feat_b[pos] as 1-D tensors)
# ---------------------------------------------------------------------------


def _active_indices(vec: torch.Tensor) -> torch.Tensor:
    """Return sorted index tensor of non-zero elements in *vec*."""
    return vec.nonzero(as_tuple=True)[0]


def jaccard(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Binary Jaccard similarity over the active (non-zero) feature index sets.

    Both *vec_a* and *vec_b* must be 1-D tensors ``[d_sae]``.
    For a TopK SAE with k = 50, |A| = |B| = 50 and
    Jaccard ∈ [0, 1] with E[J_random] ≈ 0.00076.

    Returns:
        Jaccard similarity in [0.0, 1.0].
        Returns 0.0 if both vectors are all-zero (degenerate case).
    """
    idx_a = set(_active_indices(vec_a).tolist())
    idx_b = set(_active_indices(vec_b).tolist())

    n_union = len(idx_a | idx_b)
    if n_union == 0:
        return 0.0

    n_intersect = len(idx_a & idx_b)
    return n_intersect / n_union


def weighted_jaccard(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Soft (weighted) Jaccard similarity using activation magnitudes.

    Formula:  WJ(a, b) = Σ min(|aᵢ|, |bᵢ|) / Σ max(|aᵢ|, |bᵢ|)

    This is strictly more informative than binary Jaccard for TopK SAEs:
    - Identical feature *sets* with very different *magnitudes* score < 1.
    - Features with higher magnitude contribute more to the score.
    - Reduces to binary Jaccard when all active features have equal magnitude.

    Returns:
        Weighted Jaccard in [0.0, 1.0].
        Returns 0.0 if both vectors are all-zero.
    """
    a = vec_a.abs().float()
    b = vec_b.abs().float()

    denom = torch.max(a, b).sum().item()
    if denom == 0.0:
        return 0.0

    numer = torch.min(a, b).sum().item()
    return numer / denom


def cosine_sim(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Cosine similarity between two feature vectors.

    Returns:
        Cosine similarity in [-1.0, 1.0].
        Returns 0.0 if either vector is all-zero.
    """
    a = vec_a.float()
    b = vec_b.float()
    norm_a = a.norm()
    norm_b = b.norm()
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def l1_distance(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    L1 (Manhattan) distance between two feature vectors.

    Captures total activation budget shift, complementary to cosine similarity.
    Two vectors can have high cosine similarity but large L1 distance if one
    has much larger magnitudes overall.

    Returns:
        L1 distance ≥ 0.
    """
    return float((vec_a.float() - vec_b.float()).abs().sum().item())


def rds(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Reasoning Divergence Score = 1 − Jaccard.

    Range: [0.0, 1.0].  0.0 = identical feature sets; 1.0 = fully disjoint.
    """
    return 1.0 - jaccard(vec_a, vec_b)


def weighted_rds(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Weighted Reasoning Divergence Score = 1 − weighted_jaccard.

    Range: [0.0, 1.0].  Strictly more informative than binary RDS.
    """
    return 1.0 - weighted_jaccard(vec_a, vec_b)


def exclusive_mass_asymmetry(vec_a: torch.Tensor, vec_b: torch.Tensor) -> float:
    """
    Signed asymmetry of activation mass in condition-exclusive features.

    Exclusive mass of B relative to A:  sum of vec_b[i] for i ∉ active(A)
    Exclusive mass of A relative to B:  sum of vec_a[i] for i ∉ active(B)

    Asymmetry = (excl_B − excl_A) / (excl_B + excl_A + ε)

    Returns value in (−1, 1):
        > 0: condition B brought in new features with more total activation
             than it dropped (net recruitment).
        < 0: condition B dropped more activation mass than it gained.
        ≈ 0: balanced structural change.
        Returns 0.0 if no exclusive features exist.
    """
    idx_a = set(_active_indices(vec_a).tolist())
    idx_b = set(_active_indices(vec_b).tolist())

    excl_b_idx = list(idx_b - idx_a)
    excl_a_idx = list(idx_a - idx_b)

    excl_b_mass = float(vec_b[excl_b_idx].abs().sum().item()) if excl_b_idx else 0.0
    excl_a_mass = float(vec_a[excl_a_idx].abs().sum().item()) if excl_a_idx else 0.0

    denom = excl_b_mass + excl_a_mass
    if denom < 1e-12:
        return 0.0

    return (excl_b_mass - excl_a_mass) / denom


def drift_direction_cosine(
    vec_ref: torch.Tensor,
    vec_b: torch.Tensor,
    vec_c: torch.Tensor,
) -> float:
    """
    Cosine similarity between the drift vectors (vec_b − vec_ref) and
    (vec_c − vec_ref).

    Use this metric for 3-condition comparisons:
        vec_ref = clean activations
        vec_b   = helpful_hint activations
        vec_c   = misleading_hint activations

    Returns:
        Cosine in [-1, 1]:
        +1 ⇒ both hints cause the same kind of activation shift from clean.
        −1 ⇒ the hints push activations in opposite directions.
         0 ⇒ orthogonal drifts (independent effects).
        Returns 0.0 if either delta is all-zero.
    """
    delta_b = vec_b.float() - vec_ref.float()
    delta_c = vec_c.float() - vec_ref.float()
    norm_b = delta_b.norm()
    norm_c = delta_c.norm()
    if norm_b < 1e-12 or norm_c < 1e-12:
        return 0.0
    return float(F.cosine_similarity(delta_b.unsqueeze(0), delta_c.unsqueeze(0)).item())


# ---------------------------------------------------------------------------
# High-level: compute all pairwise metrics in one call
# ---------------------------------------------------------------------------


def compare_pair(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    pos_a: int,
    pos_b: int,
) -> dict[str, float]:
    """
    Compute all pairwise metrics between feat_a at pos_a and feat_b at pos_b.

    Args:
        feat_a: Feature tensor ``[seq_len_a, d_sae]``.
        feat_b: Feature tensor ``[seq_len_b, d_sae]``.
        pos_a:  Token position index into feat_a.
        pos_b:  Token position index into feat_b.

    Returns:
        Dict with keys: jaccard, weighted_jaccard, cosine_sim, l1_distance,
        rds, weighted_rds, exclusive_mass_asymmetry.
    """
    va = feat_a[pos_a]
    vb = feat_b[pos_b]

    j = jaccard(va, vb)
    wj = weighted_jaccard(va, vb)

    return {
        "jaccard": j,
        "weighted_jaccard": wj,
        "cosine_sim": cosine_sim(va, vb),
        "l1_distance": l1_distance(va, vb),
        "rds": 1.0 - j,
        "weighted_rds": 1.0 - wj,
        "exclusive_mass_asymmetry": exclusive_mass_asymmetry(va, vb),
    }
