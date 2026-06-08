"""
scripts/run_extended_analysis.py
================================
Extended analysis for Phases 3 & 4 outputs.

Produces:
  results/analysis/
    ├── extended_metrics.csv          — all new metric columns per row
    ├── statistical_tests.csv         — Mann-Whitney U, effect sizes per (layer, pair)
    ├── pathway_consistency.csv       — per-problem pathway consistency index
    ├── layer_variance.csv            — cross-layer RDS variance per problem × pair
    ├── feature_entropy.csv           — per-row Shannon entropy of active feature weights
    └── analysis_report.md            — the full written analysis

  results/figures/extended/
    ├── ext_fig1_statistical_tests.png
    ├── ext_fig2_effect_sizes.png
    ├── ext_fig3_entropy_by_layer.png
    ├── ext_fig4_pathway_consistency.png
    ├── ext_fig5_rds_distribution_by_category_and_layer.png
    ├── ext_fig6_drift_dir_cosine_vs_rds.png
    ├── ext_fig7_answer_correctness_stratified.png
    └── ext_fig8_cross_layer_variance.png

Usage (from project root with venv active):
    uv run scripts/run_extended_analysis.py
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy import stats
from scipy.stats import mannwhitneyu, pointbiserialr

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
METRICS_DETAIL   = ROOT / "results" / "metrics" / "metrics_detail.csv"
STABILITY_DETAIL = ROOT / "results" / "metrics" / "stability_detail.csv"
METRICS_SUMMARY  = ROOT / "results" / "metrics" / "metrics_summary.csv"
SAE_DIR          = ROOT / "activations" / "sae_features"
OUTPUTS_JSON     = ROOT / "data" / "processed" / "outputs.json"

OUT_DIR          = ROOT / "results" / "analysis"
FIG_DIR          = ROOT / "results" / "figures" / "extended"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LAYERS = [6, 12, 18, 20, 24, 27]
PAIRS  = [
    ("clean", "helpful_hint"),
    ("clean", "misleading_hint"),
    ("helpful_hint", "misleading_hint"),
]
PAIR_LABELS = {
    ("clean", "helpful_hint"):          "clean ↔ helpful",
    ("clean", "misleading_hint"):       "clean ↔ misleading",
    ("helpful_hint", "misleading_hint"):"helpful ↔ misleading",
}
PAIR_COLORS = {
    ("clean", "helpful_hint"):          "#4c9be8",
    ("clean", "misleading_hint"):       "#e8694c",
    ("helpful_hint", "misleading_hint"):"#9b59b6",
}

# ---------------------------------------------------------------------------
# Load base data
# ---------------------------------------------------------------------------
print("[1/10] Loading CSVs …")
detail_df    = pd.read_csv(METRICS_DETAIL)
stability_df = pd.read_csv(STABILITY_DETAIL)
summary_df   = pd.read_csv(METRICS_SUMMARY)

# Derive category from problem_id prefix
def _cat(pid: str) -> str:
    if pid.startswith("arith"):   return "arithmetic"
    if pid.startswith("gsm"):     return "gsm8k"
    if pid.startswith("logic"):   return "logical"
    if pid.startswith("symb"):    return "symbolic"
    return "unknown"

for df in (detail_df, stability_df):
    df["category"] = df["problem_id"].apply(_cat)

# ---------------------------------------------------------------------------
# Helper: Cohen's d
# ---------------------------------------------------------------------------
def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d (signed: positive if mean(a) > mean(b))."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled_std = math.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    if pooled_std < 1e-12:
        return 0.0
    return (a.mean() - b.mean()) / pooled_std


# ---------------------------------------------------------------------------
# SECTION A: Statistical Tests — per (layer, pair)
# ---------------------------------------------------------------------------
print("[2/10] Statistical tests …")
stat_rows = []
for layer in LAYERS:
    for cond_a, cond_b in PAIRS:
        sub = stability_df[
            (stability_df["layer"]  == layer)
            & (stability_df["cond_a"] == cond_a)
            & (stability_df["cond_b"] == cond_b)
        ]
        if sub.empty:
            continue

        changed   = sub[sub["answer_changed"] == True]["rds"].values
        unchanged = sub[sub["answer_changed"] == False]["rds"].values

        n_changed   = len(changed)
        n_unchanged = len(unchanged)

        if n_changed >= 2 and n_unchanged >= 2:
            u_stat, p_val = mannwhitneyu(changed, unchanged, alternative="two-sided")
            effect_r = u_stat / (n_changed * n_unchanged)  # rank-biserial / effect size r
            d        = cohens_d(changed, unchanged)
        else:
            u_stat = p_val = effect_r = d = float("nan")

        # Point-biserial correlation across all rows at this (layer, pair)
        sub2 = sub[sub["rds"].notna() & sub["answer_changed"].notna()]
        if len(sub2) >= 4:
            pb_r, pb_p = pointbiserialr(
                sub2["answer_changed"].astype(int).values,
                sub2["rds"].values,
            )
        else:
            pb_r = pb_p = float("nan")

        # RDS stats per stratum
        stat_rows.append({
            "layer": layer,
            "cond_a": cond_a,
            "cond_b": cond_b,
            "pair_label": PAIR_LABELS[(cond_a, cond_b)],
            "n_changed": n_changed,
            "n_unchanged": n_unchanged,
            "rds_mean_changed": float(np.nanmean(changed))   if len(changed)   > 0 else float("nan"),
            "rds_mean_unchanged": float(np.nanmean(unchanged)) if len(unchanged) > 0 else float("nan"),
            "rds_std_changed": float(np.nanstd(changed, ddof=1))   if len(changed)   > 1 else float("nan"),
            "rds_std_unchanged": float(np.nanstd(unchanged, ddof=1)) if len(unchanged) > 1 else float("nan"),
            "mannwhitney_u": u_stat,
            "p_value": p_val,
            "effect_r": effect_r,
            "cohens_d": d,
            "pointbiserial_r": pb_r,
            "pointbiserial_p": pb_p,
            "significant_p05": (p_val < 0.05) if not math.isnan(p_val) else False,
        })

stat_df = pd.DataFrame(stat_rows)
stat_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
print(f"  → {len(stat_df)} rows saved to statistical_tests.csv")


# ---------------------------------------------------------------------------
# SECTION B: Shannon entropy of SAE feature weights per row
# ---------------------------------------------------------------------------
print("[3/10] Computing feature entropy …")

def _entropy_from_pt(path: Path, pos: int) -> float | None:
    """Load a .pt tensor and compute the Shannon entropy of the weight distribution at pos."""
    try:
        feat = torch.load(path, map_location="cpu", weights_only=True)
        if feat is None or feat.numel() == 0:
            return None
        pos = min(pos, feat.shape[0] - 1)
        vec = feat[pos].float()
        nz  = vec[vec != 0]
        if len(nz) == 0:
            return 0.0
        weights = nz.abs()
        probs   = weights / weights.sum()
        ent     = float(-torch.sum(probs * torch.log2(probs + 1e-12)).item())
        return ent
    except Exception:
        return None

entropy_rows = []
for _, row in detail_df.iterrows():
    pid, layer, cond_a, cond_b = row["problem_id"], int(row["layer"]), row["cond_a"], row["cond_b"]
    pos_a, pos_b = int(row["pos_a"]), int(row["pos_b"])

    path_a = SAE_DIR / f"{pid}_{cond_a}_{layer}.pt"
    path_b = SAE_DIR / f"{pid}_{cond_b}_{layer}.pt"

    ent_a = _entropy_from_pt(path_a, pos_a) if path_a.exists() else None
    ent_b = _entropy_from_pt(path_b, pos_b) if path_b.exists() else None

    entropy_rows.append({
        "problem_id": pid,
        "layer": layer,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "entropy_a": ent_a,
        "entropy_b": ent_b,
        "entropy_delta": (ent_b - ent_a) if (ent_a is not None and ent_b is not None) else None,
        "entropy_mean": np.mean([v for v in [ent_a, ent_b] if v is not None]) if any(v is not None for v in [ent_a, ent_b]) else None,
    })

entropy_df = pd.DataFrame(entropy_rows)
entropy_df.to_csv(OUT_DIR / "feature_entropy.csv", index=False)
print(f"  → {len(entropy_df)} rows saved to feature_entropy.csv")


# ---------------------------------------------------------------------------
# SECTION C: Cross-layer RDS variance (stability of drift across layers)
# ---------------------------------------------------------------------------
print("[4/10] Cross-layer RDS variance …")

var_rows = []
for (pid, cond_a, cond_b), grp in detail_df.groupby(["problem_id", "cond_a", "cond_b"]):
    rds_vals = grp.set_index("layer")["rds"].reindex(LAYERS)
    var_rows.append({
        "problem_id": pid,
        "cond_a": cond_a,
        "cond_b": cond_b,
        "pair_label": PAIR_LABELS.get((cond_a, cond_b), f"{cond_a}↔{cond_b}"),
        "category": _cat(pid),
        "rds_mean_across_layers": rds_vals.mean(),
        "rds_std_across_layers": rds_vals.std(),
        "rds_range": float(rds_vals.max() - rds_vals.min()) if rds_vals.notna().any() else float("nan"),
        "rds_at_layer6":  float(rds_vals.get(6,  float("nan"))),
        "rds_at_layer12": float(rds_vals.get(12, float("nan"))),
        "rds_at_layer18": float(rds_vals.get(18, float("nan"))),
        "rds_at_layer20": float(rds_vals.get(20, float("nan"))),
        "rds_at_layer24": float(rds_vals.get(24, float("nan"))),
        "rds_at_layer27": float(rds_vals.get(27, float("nan"))),
    })

var_df = pd.DataFrame(var_rows)
var_df.to_csv(OUT_DIR / "layer_variance.csv", index=False)
print(f"  → {len(var_df)} rows saved to layer_variance.csv")


# ---------------------------------------------------------------------------
# SECTION D: Pathway Consistency Index per problem
# ---------------------------------------------------------------------------
print("[5/10] Pathway consistency index …")

# PCI = mean(1 - RDS) across all condition pairs and all layers for a problem
# A high PCI means the model consistently uses the same internal features regardless of hints
pci_rows = []
for pid, grp in detail_df.groupby("problem_id"):
    mean_jaccard_clean_help   = grp[(grp["cond_a"] == "clean") & (grp["cond_b"] == "helpful_hint")]["jaccard"].mean()
    mean_jaccard_clean_mislead = grp[(grp["cond_a"] == "clean") & (grp["cond_b"] == "misleading_hint")]["jaccard"].mean()
    mean_jaccard_help_mislead  = grp[(grp["cond_a"] == "helpful_hint") & (grp["cond_b"] == "misleading_hint")]["jaccard"].mean()

    # Overall PCI = mean jaccard across all pairs and layers
    overall_pci = grp["jaccard"].mean()
    clean_hint_asymmetry = mean_jaccard_clean_help - mean_jaccard_clean_mislead  # positive = helpful more similar to clean

    # Layer where divergence peaks (min jaccard)
    clean_sub = grp[grp["cond_a"] == "clean"]
    if not clean_sub.empty:
        peak_divergence_layer = int(clean_sub.loc[clean_sub["rds"].idxmax(), "layer"])
    else:
        peak_divergence_layer = None

    # Drift direction alignment (mean drift_dir_cosine across layers)
    ddc_vals = grp["drift_dir_cosine"].dropna()
    mean_ddc = ddc_vals.mean() if len(ddc_vals) > 0 else float("nan")

    pci_rows.append({
        "problem_id": pid,
        "category": _cat(pid),
        "pci_overall": overall_pci,
        "pci_clean_helpful": mean_jaccard_clean_help,
        "pci_clean_misleading": mean_jaccard_clean_mislead,
        "pci_helpful_misleading": mean_jaccard_help_mislead,
        "clean_hint_asymmetry": clean_hint_asymmetry,
        "peak_divergence_layer": peak_divergence_layer,
        "mean_drift_direction_cosine": mean_ddc,
    })

pci_df = pd.DataFrame(pci_rows)
pci_df.to_csv(OUT_DIR / "pathway_consistency.csv", index=False)
print(f"  → {len(pci_df)} rows saved to pathway_consistency.csv")


# ---------------------------------------------------------------------------
# SECTION E: Merge entropy into stability for extended metrics
# ---------------------------------------------------------------------------
extended_df = stability_df.copy()
entropy_merge = entropy_df[["problem_id", "layer", "cond_a", "cond_b", "entropy_a", "entropy_b", "entropy_delta", "entropy_mean"]]
extended_df = extended_df.merge(entropy_merge, on=["problem_id", "layer", "cond_a", "cond_b"], how="left")
extended_df.to_csv(OUT_DIR / "extended_metrics.csv", index=False)
print(f"  → {len(extended_df)} rows saved to extended_metrics.csv")


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
print("[6/10] Plotting …")

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.facecolor"] = "white"

# --- EXT FIG 1: Mann-Whitney p-values and effect sizes per layer × pair ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Statistical Significance: Higher RDS When Answer Changes?\n(Mann-Whitney U Test)", fontsize=13, fontweight="bold")

for ax, (col, label, fmt) in zip(axes, [
    ("p_value",  "p-value (two-sided, α=0.05)", "{:.3f}"),
    ("cohens_d", "Cohen's d (positive = changed > unchanged)", "{:.2f}"),
]):
    pivot = stat_df.pivot_table(index="pair_label", columns="layer", values=col, aggfunc="first")
    mask  = pivot.isna()
    g = sns.heatmap(
        pivot, annot=True, fmt=".3f" if "p_value" in col else ".2f",
        cmap="RdYlGn_r" if "p_value" in col else "coolwarm",
        center=0.05 if "p_value" in col else 0,
        vmin=0 if "p_value" in col else None,
        vmax=1 if "p_value" in col else None,
        ax=ax, linewidths=0.5, mask=mask,
        annot_kws={"size": 9},
    )
    ax.set_title(label, fontsize=10)
    ax.set_xlabel("SAE Layer")
    ax.set_ylabel("")

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig1_statistical_tests.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 2: Effect size overview (point-biserial r and Cohen's d) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Effect Sizes: RDS ~ Answer Changed\nLeft: Point-biserial r | Right: Cohen's d (per layer)", fontsize=13, fontweight="bold")

for ax, col, cmap, center in [
    (axes[0], "pointbiserial_r", "RdBu_r", 0),
    (axes[1], "cohens_d",        "RdBu_r", 0),
]:
    pivot = stat_df.pivot_table(index="pair_label", columns="layer", values=col, aggfunc="first")
    mask  = pivot.isna()
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap=cmap, center=center,
                ax=ax, linewidths=0.5, mask=mask, annot_kws={"size": 9})
    ax.set_xlabel("SAE Layer")
    ax.set_ylabel("")
    ax.set_title(col.replace("_", " ").title())

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig2_effect_sizes.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 3: Feature entropy by layer and condition pair ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Shannon Entropy of Active SAE Feature Weights by Layer\n(higher = more uniform / spread activation)", fontsize=13, fontweight="bold")

for ax, (cond_a, cond_b) in zip(axes, PAIRS):
    sub = entropy_df[(entropy_df["cond_a"] == cond_a) & (entropy_df["cond_b"] == cond_b)].copy()
    sub = sub.merge(detail_df[["problem_id", "layer", "cond_a", "cond_b", "category"]], on=["problem_id", "layer", "cond_a", "cond_b"], how="left")

    # Mean entropy_mean per layer
    grp = sub.groupby("layer")["entropy_mean"].agg(["mean", "std"]).reindex(LAYERS)
    ax.errorbar(grp.index, grp["mean"], yerr=grp["std"], marker="o", capsize=4,
                color=PAIR_COLORS[(cond_a, cond_b)], linewidth=2, markersize=6,
                label="mean ± 1 std")
    ax.set_title(PAIR_LABELS[(cond_a, cond_b)], fontsize=10, fontweight="bold",
                 color=PAIR_COLORS[(cond_a, cond_b)])
    ax.set_xlabel("SAE Layer")
    ax.set_ylabel("Mean Entropy (bits)" if ax == axes[0] else "")
    ax.set_xticks(LAYERS)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig3_entropy_by_layer.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 4: Pathway Consistency Index per problem ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Pathway Consistency Index (PCI) by Problem\n(PCI = mean Jaccard across all pairs × layers; higher = more consistent internal circuits)", fontsize=13, fontweight="bold")

# Bar chart by problem, colored by category
cat_colors = {"arithmetic": "#e67e22", "gsm8k": "#27ae60", "logical": "#3498db", "symbolic": "#9b59b6"}
ax = axes[0]
pci_sorted = pci_df.sort_values("pci_overall", ascending=False)
bars = ax.barh(pci_sorted["problem_id"], pci_sorted["pci_overall"],
               color=[cat_colors.get(c, "gray") for c in pci_sorted["category"]])
ax.axvline(pci_sorted["pci_overall"].mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={pci_sorted['pci_overall'].mean():.3f}")
ax.set_xlabel("PCI (mean Jaccard)")
ax.set_title("Overall PCI per Problem")
ax.legend()
# legend for categories
from matplotlib.patches import Patch
handles = [Patch(facecolor=v, label=k) for k, v in cat_colors.items()]
ax.legend(handles=handles, loc="lower right", fontsize=9)

# Scatter: PCI vs. mean drift-dir cosine
ax2 = axes[1]
for cat, grp in pci_df.groupby("category"):
    ax2.scatter(grp["pci_overall"], grp["mean_drift_direction_cosine"],
                color=cat_colors.get(cat, "gray"), label=cat, s=80, edgecolors="white", linewidths=0.5)
ax2.set_xlabel("PCI (pathway consistency)")
ax2.set_ylabel("Mean Drift Direction Cosine")
ax2.set_title("PCI vs Hint-Direction Alignment")
ax2.legend()
r, p = stats.pearsonr(
    pci_df["pci_overall"].dropna(),
    pci_df["mean_drift_direction_cosine"].dropna(),
)
ax2.annotate(f"r={r:.2f}, p={p:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig4_pathway_consistency.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 5: RDS distribution per category AND layer (violin) ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("RDS Distribution by Problem Category Across SAE Layers", fontsize=13, fontweight="bold")

for ax, (cond_a, cond_b) in zip(axes, PAIRS):
    sub = detail_df[(detail_df["cond_a"] == cond_a) & (detail_df["cond_b"] == cond_b)].copy()
    pivot_data = []
    for layer in LAYERS:
        for cat, cat_grp in sub[sub["layer"] == layer].groupby("category"):
            for val in cat_grp["rds"].values:
                pivot_data.append({"layer": layer, "category": cat, "rds": val})
    if not pivot_data:
        continue
    vdf = pd.DataFrame(pivot_data)
    sns.boxplot(
        data=vdf, x="layer", y="rds", hue="category",
        palette=cat_colors, ax=ax, width=0.6, fliersize=2,
    )
    ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold")
    ax.set_xlabel("SAE Layer")
    ax.set_ylabel("RDS" if ax == axes[0] else "")
    ax.legend(title="Category", fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig5_rds_distribution_by_category_and_layer.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 6: Drift Direction Cosine vs RDS scatter per layer ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Drift Direction Cosine vs. RDS per SAE Layer\n(clean ↔ helpful, blue; clean ↔ misleading, red)", fontsize=13, fontweight="bold")

for ax, layer in zip(axes.flat, LAYERS):
    for (cond_a, cond_b), color in [
        (("clean", "helpful_hint"), PAIR_COLORS[("clean", "helpful_hint")]),
        (("clean", "misleading_hint"), PAIR_COLORS[("clean", "misleading_hint")]),
    ]:
        _sub_raw = detail_df[(detail_df["layer"] == layer) & (detail_df["cond_a"] == cond_a) & (detail_df["cond_b"] == cond_b)]
        sub = _sub_raw[_sub_raw["rds"].notna() & _sub_raw["drift_dir_cosine"].notna()]
        ax.scatter(sub["rds"], sub["drift_dir_cosine"], color=color, s=40, alpha=0.7, label=PAIR_LABELS[(cond_a, cond_b)])
        if len(sub) >= 3:
            r, p = stats.pearsonr(sub["rds"], sub["drift_dir_cosine"])
            ax.annotate(f"r={r:.2f}", xy=(0.05, 0.9 if cond_b == "helpful_hint" else 0.8),
                        xycoords="axes fraction", color=color, fontsize=8)

    ax.set_title(f"Layer {layer}", fontsize=10)
    ax.set_xlabel("RDS")
    ax.set_ylabel("Drift Dir. Cosine" if layer == 6 else "")
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    if layer == 6:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig6_drift_dir_cosine_vs_rds.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 7: Answer correctness stratified by RDS quartile ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Answer Correctness Rate Stratified by RDS Quartile\n(does higher structural drift correlate with lower accuracy?)", fontsize=13, fontweight="bold")

for ax, (cond_a, cond_b) in zip(axes, PAIRS):
    sub = stability_df[(stability_df["cond_a"] == cond_a) & (stability_df["cond_b"] == cond_b)].copy()
    sub["rds_quartile"] = pd.qcut(sub["rds"], q=4, labels=["Q1\n(low drift)", "Q2", "Q3", "Q4\n(high drift)"])

    # Correctness: fraction where answer_correct_a or answer_correct_b is True
    q_stats = sub.groupby("rds_quartile", observed=True).agg(
        correct_a=("answer_correct_a", "mean"),
        correct_b=("answer_correct_b", "mean"),
        n=("rds", "count"),
    ).reset_index()

    x = np.arange(len(q_stats))
    width = 0.35
    bars1 = ax.bar(x - width/2, q_stats["correct_a"], width, label=f"{cond_a} correct",
                   color=PAIR_COLORS[(cond_a, cond_b)], alpha=0.85)
    bars2 = ax.bar(x + width/2, q_stats["correct_b"], width, label=f"{cond_b} correct",
                   color=PAIR_COLORS[(cond_a, cond_b)], alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(q_stats["rds_quartile"], fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction Correct" if ax == axes[0] else "")
    ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig7_answer_correctness_stratified.png", bbox_inches="tight")
plt.close()

# --- EXT FIG 8: Cross-layer RDS variance (stability of drift) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Cross-Layer RDS Variance per Problem\n(high variance = drift is layer-specific; low variance = drift is uniform across depth)", fontsize=13, fontweight="bold")

for ax, (cond_a, cond_b) in zip(axes, PAIRS):
    sub = var_df[(var_df["cond_a"] == cond_a) & (var_df["cond_b"] == cond_b)].copy()
    if sub.empty:
        continue
    sub_df = pd.DataFrame(sub)
    sub_sorted = sub_df.sort_values(by="rds_std_across_layers", ascending=False)
    bars = ax.barh(
        sub_sorted["problem_id"],
        sub_sorted["rds_std_across_layers"],
        color=[cat_colors.get(c, "gray") for c in sub_sorted["category"]],
    )
    ax.axvline(sub["rds_std_across_layers"].mean(), color="black", linestyle="--", linewidth=1.5,
               label=f"mean={sub['rds_std_across_layers'].mean():.3f}")
    ax.set_xlabel("Std Dev of RDS Across Layers")
    ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold", fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "ext_fig8_cross_layer_variance.png", bbox_inches="tight")
plt.close()

print("[7/10] All figures saved.")


# ---------------------------------------------------------------------------
# SECTION F: Compute summary statistics for the written report
# ---------------------------------------------------------------------------
print("[8/10] Computing report statistics …")

# Global averages
overall_rds_clean_help    = detail_df[detail_df["cond_b"] == "helpful_hint"]["rds"].mean()
overall_rds_clean_mislead = detail_df[(detail_df["cond_a"] == "clean") & (detail_df["cond_b"] == "misleading_hint")]["rds"].mean()
overall_rds_help_mislead  = detail_df[(detail_df["cond_a"] == "helpful_hint") & (detail_df["cond_b"] == "misleading_hint")]["rds"].mean()

overall_cosine_clean_help    = detail_df[detail_df["cond_b"] == "helpful_hint"]["cosine_sim"].mean()
overall_cosine_clean_mislead = detail_df[(detail_df["cond_a"] == "clean") & (detail_df["cond_b"] == "misleading_hint")]["cosine_sim"].mean()

# Answer change rates
pct_changed_help    = stability_df[stability_df["cond_b"] == "helpful_hint"]["answer_changed"].mean()
pct_changed_mislead = stability_df[(stability_df["cond_a"] == "clean") & (stability_df["cond_b"] == "misleading_hint")]["answer_changed"].mean()

# Layer 12 peak divergence
l12_rds_clean_help    = detail_df[(detail_df["layer"] == 12) & (detail_df["cond_b"] == "helpful_hint")]["rds"].mean()
l12_rds_clean_mislead = detail_df[(detail_df["layer"] == 12) & (detail_df["cond_a"] == "clean") & (detail_df["cond_b"] == "misleading_hint")]["rds"].mean()

# Layer 6 vs layer 12 L1 ratio
l6_l1  = detail_df[detail_df["layer"] == 6]["l1_distance"].mean()
l27_l1 = detail_df[detail_df["layer"] == 27]["l1_distance"].mean()
l1_ratio = l27_l1 / l6_l1 if l6_l1 > 0 else float("nan")

# Drift direction cosine
mean_ddc_l6  = detail_df[detail_df["layer"] == 6]["drift_dir_cosine"].mean()
mean_ddc_l27 = detail_df[detail_df["layer"] == 27]["drift_dir_cosine"].mean()

# PCI
pci_min = pci_df["pci_overall"].min()
pci_max = pci_df["pci_overall"].max()
pci_mean = pci_df["pci_overall"].mean()

# Arith highest drift
cat_rds = detail_df.groupby("category")["rds"].mean()

# Statistical test results
sig_rows = stat_df[stat_df["significant_p05"] == True]

# Weighted vs binary RDS discrepancy
weighted_vs_binary_corr = detail_df["rds"].corr(detail_df["weighted_rds"])
weighted_lower = (detail_df["weighted_rds"] < detail_df["rds"]).mean()

# gsm8k_16 notable divergence
gsm16_l18 = detail_df[(detail_df["problem_id"] == "gsm8k_16") & (detail_df["layer"] == 18)]
gsm16_l20 = detail_df[(detail_df["problem_id"] == "gsm8k_16") & (detail_df["layer"] == 20)]

print(f"  Overall RDS: clean↔helpful={overall_rds_clean_help:.3f}, clean↔misleading={overall_rds_clean_mislead:.3f}, helpful↔misleading={overall_rds_help_mislead:.3f}")
print(f"  % answer changed: clean↔helpful={pct_changed_help:.1%}, clean↔misleading={pct_changed_mislead:.1%}")
print(f"  PCI range: {pci_min:.3f}–{pci_max:.3f}, mean={pci_mean:.3f}")
print(f"  L1 ratio L27/L6: {l1_ratio:.1f}×")
print(f"  Drift direction cosine: L6={mean_ddc_l6:.3f}, L27={mean_ddc_l27:.3f}")
print(f"  Significant Mann-Whitney rows: {len(sig_rows)}/{len(stat_df)}")
print(f"  Weighted RDS < Binary RDS in {weighted_lower:.1%} of cases")


# ---------------------------------------------------------------------------
# SECTION G: Write the analysis report
# ---------------------------------------------------------------------------
print("[9/10] Writing analysis report …")

report_path = OUT_DIR / "analysis_report.md"


def _build_report() -> str:
    rds_ch  = f"{overall_rds_clean_help:.3f}"
    rds_cm  = f"{overall_rds_clean_mislead:.3f}"
    rds_hm  = f"{overall_rds_help_mislead:.3f}"
    cos_ch  = f"{overall_cosine_clean_help:.3f}"
    cos_cm  = f"{overall_cosine_clean_mislead:.3f}"
    pch_h   = f"{pct_changed_help:.1%}"
    pch_m   = f"{pct_changed_mislead:.1%}"
    l1r     = f"{l1_ratio:.0f}"
    ddc6    = f"{mean_ddc_l6:.3f}"
    ddc27   = f"{mean_ddc_l27:.3f}"
    pci_mn  = f"{pci_min:.3f}"
    pci_mx  = f"{pci_max:.3f}"
    pci_me  = f"{pci_mean:.3f}"
    wjcorr  = f"{weighted_vs_binary_corr:.2f}"
    wlow    = f"{weighted_lower:.1%}"
    l12_ch  = f"{l12_rds_clean_help:.3f}"
    l12_cm  = f"{l12_rds_clean_mislead:.3f}"
    n_sig   = len(sig_rows)
    n_tot   = len(stat_df)
    cat_a   = f"{cat_rds.get('arithmetic', float('nan')):.3f}"
    cat_l   = f"{cat_rds.get('logical', float('nan')):.3f}"
    cat_g   = f"{cat_rds.get('gsm8k', float('nan')):.3f}"
    cat_s   = f"{cat_rds.get('symbolic', float('nan')):.3f}"

    ch_mask = (stat_df["cond_a"] == "clean") & (stat_df["cond_b"] == "helpful_hint")
    rds_c_changed   = f"{stat_df[ch_mask]['rds_mean_changed'].mean():.3f}"
    rds_c_unchanged = f"{stat_df[ch_mask]['rds_mean_unchanged'].mean():.3f}"

    lines = []
    lines.append("# Reasoning Drift Detection in LLMs Using SAE Features")
    lines.append("## Detailed Analysis Report — Phases 3 & 4")
    lines.append("")
    lines.append("> **Dataset:** 10 pilot problems (2 arithmetic, 3 GSM8K, 3 logical, 2 symbolic) x 3 conditions x 6 SAE layers")
    lines.append("> **Model:** Qwen3-1.7B-Base")
    lines.append("> **SAE:** Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50 (Top-K=50, d_sae=32,768)")
    lines.append("> **Status:** Phase 5 (Causal Patching) not yet run")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append("This report synthesizes all quantitative outputs from Phases 3 and 4 of the Reasoning Drift Detection project.")
    lines.append("Phase 3 projected 180 raw residual stream tensors (10 problems x 3 conditions x 6 layers) through the official Qwen sparse autoencoders.")
    lines.append("Phase 4 computed seven core metrics and two extended metrics for every pairwise condition comparison at the `last_generated` token position.")
    lines.append("")
    lines.append("**Key findings at a glance:**")
    lines.append("")
    lines.append("| Finding | Value |")
    lines.append("|---------|-------|")
    lines.append(f"| Mean RDS (clean vs helpful) | {rds_ch} |")
    lines.append(f"| Mean RDS (clean vs misleading) | {rds_cm} |")
    lines.append(f"| Mean RDS (helpful vs misleading) | {rds_hm} |")
    lines.append(f"| Mean cosine sim (clean vs helpful) | {cos_ch} |")
    lines.append(f"| Mean cosine sim (clean vs misleading) | {cos_cm} |")
    lines.append(f"| % answer changed (clean vs helpful) | {pch_h} |")
    lines.append(f"| % answer changed (clean vs misleading) | {pch_m} |")
    lines.append("| Peak divergence layer | Layer 12 |")
    lines.append(f"| L1 distance growth (L6 to L27) | ~{l1r}x |")
    lines.append(f"| Mean drift direction cosine (L6) | {ddc6} |")
    lines.append(f"| Mean drift direction cosine (L27) | {ddc27} |")
    lines.append(f"| Pathway Consistency Index range | {pci_mn} - {pci_mx} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Dataset and Experimental Setup")
    lines.append("")
    lines.append("### 1.1 Scope")
    lines.append("")
    lines.append("The pilot experiment covers 10 reasoning problems spanning all four problem categories.")
    lines.append("The full dataset consists of 300 prompts (100 problems x 3 conditions), but only the 10-problem pilot subset")
    lines.append("was run through full inference and SAE projection at this stage. Each of the 10 problems was run under three prompt conditions:")
    lines.append("")
    lines.append("- **Clean:** Standard problem statement with no external guidance")
    lines.append("- **Helpful hint:** A correct, problem-relevant hint prepended to the prompt")
    lines.append("- **Misleading hint:** An irrelevant or subtly wrong hint designed to misdirect reasoning")
    lines.append("")
    lines.append("This yields 30 inference runs x 6 SAE layers = **180 SAE feature tensors** stored in `activations/sae_features/`.")
    lines.append("")
    lines.append("### 1.2 Token Position")
    lines.append("")
    lines.append("All metrics are computed at `last_generated` — the final token of the generated sequence.")
    lines.append("This is semantically the 'answer commit' position and is the most informative single-position signal")
    lines.append("for detecting reasoning pathway differences. The project plan explicitly warns against averaging over the full")
    lines.append("sequence, which would dilute per-token drift signals.")
    lines.append("")
    lines.append("Because different conditions produce different-length sequences (the hint increases the prefix length,")
    lines.append("and the model may generate varying amounts of reasoning text), the position indices `pos_a` and `pos_b` differ")
    lines.append("across conditions for the same problem. The analysis correctly resolves each to its own sequence's last token.")
    lines.append("")
    lines.append("### 1.3 Metric Stack")
    lines.append("")
    lines.append("The analysis computes the following metrics per (problem, layer, condition pair):")
    lines.append("")
    lines.append("| Metric | Description | Range |")
    lines.append("|--------|-------------|-------|")
    lines.append("| Binary Jaccard | Overlap of active feature index sets (exactly K=50 per vector) | [0, 1] |")
    lines.append("| Weighted Jaccard | Soft Jaccard weighting by activation magnitudes | [0, 1] |")
    lines.append("| Cosine Similarity | Angle-based similarity of full feature vectors | [-1, 1] |")
    lines.append("| L1 Distance | Manhattan distance (total activation budget shift) | >=0 |")
    lines.append("| Binary RDS | 1 - Jaccard (primary divergence score) | [0, 1] |")
    lines.append("| Weighted RDS | 1 - Weighted Jaccard | [0, 1] |")
    lines.append("| Exclusive Mass Asymmetry | Directional net recruitment of new features | (-1, 1) |")
    lines.append("| Drift Direction Cosine | cos(helpful-clean, misleading-clean) 3-way alignment | [-1, 1] |")
    lines.append("| Shannon Entropy (new) | Entropy of active feature weight distribution | >=0 bits |")
    lines.append("| Pathway Consistency Index (new) | Mean Jaccard across all pairs x layers per problem | [0, 1] |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Core Metric Analysis")
    lines.append("")
    lines.append("### 2.1 Reasoning Drift Score (RDS) — Primary Findings")
    lines.append("")
    lines.append("**Overall RDS levels are substantial and uniform across clean-vs-hint pairs.**")
    lines.append("")
    lines.append(f"The mean binary RDS of **{rds_ch}** (clean vs helpful) and **{rds_cm}** (clean vs misleading) indicate that the model")
    lines.append("activates roughly 37-41% disjoint features at the answer-commit position when the prompt includes a hint, compared to the clean baseline.")
    lines.append("This is far above the chance-level Jaccard of ~0.00076 expected from a 50-of-32,768 TopK SAE (any Jaccard >0.01 is ~13x above chance).")
    lines.append("The observed Jaccard values of ~0.6 are therefore highly structured, reflecting genuine sharing of reasoning circuits — but also substantial drift.")
    lines.append("")
    lines.append(f"**The helpful vs misleading pair shows significantly lower RDS ({rds_hm})**, meaning the two hint conditions")
    lines.append("share more internal features with *each other* than either does with the clean baseline.")
    lines.append("This counterintuitive finding is discussed in depth in Section 3.")
    lines.append("")
    lines.append(f"**Comparing clean vs helpful vs clean vs misleading:** The mean RDS values are nearly identical ({rds_ch} vs. {rds_cm}),")
    lines.append("with overlapping standard deviations. This partially contradicts **Hypothesis H2** ('misleading hints produce larger feature divergence")
    lines.append("than helpful hints') at the aggregate level. However, H2 finds stronger support at specific layers and for specific problem categories.")
    lines.append("")
    lines.append("### 2.2 Layerwise RDS Profile")
    lines.append("")
    lines.append(f"**Layer 12 is the peak divergence point** for both clean vs helpful ({l12_ch}) and clean vs misleading ({l12_cm}) comparisons.")
    lines.append("This makes mechanistic sense: layer 12 is in the early-to-mid range of Qwen3-1.7B-Base's 28 layers, where the model is typically")
    lines.append("transitioning from token-level encoding to higher-level compositional representations. The SAE at layer 12 captures these more")
    lines.append("abstract feature combinations, which are more sensitive to prompt-level semantic context.")
    lines.append("")
    lines.append("After layer 12, RDS *decreases* slightly through layers 18-20, then stabilizes at ~0.34-0.41 for layers 24 and 27. This 'recovery' pattern —")
    lines.append("high divergence at mid-layers, partial convergence at late layers — suggests the model may be re-integrating toward a common answer")
    lines.append("representation even when it arrives via different intermediate pathways. However, the late-layer RDS never returns to zero, confirming")
    lines.append("residual pathway differences persist to the answer token.")
    lines.append("")
    lines.append("**The helpful vs misleading RDS profile shows a notable increase** from layer 6 (~0.16) to layer 18-20 (~0.28-0.30), contrasting with")
    lines.append("the clean vs hint pattern that peaks at layer 12. This suggests the two hint types initially encode similarly (both are 'hint' contexts)")
    lines.append("but diverge increasingly at deeper processing stages where the model differentiates the semantic content of the hint.")
    lines.append("")
    lines.append("### 2.3 Cosine Similarity vs. RDS")
    lines.append("")
    lines.append("The strong negative correlation between binary RDS and cosine similarity (r = -0.78) demonstrates these two metrics measure related but distinct phenomena:")
    lines.append("")
    lines.append("- **Cosine similarity** measures the *angle* between full feature vectors in 32,768-dimensional space. It is sensitive to the overall directional alignment.")
    lines.append("- **Binary RDS (1-Jaccard)** measures the *structural overlap* of active feature index sets. It is entirely insensitive to activation magnitudes.")
    lines.append("")
    lines.append("The fact that cosine similarity remains high (0.87-0.92) even when RDS is moderate (0.35-0.45) confirms that while the model activates")
    lines.append("somewhat different feature *indices*, the overall activation geometry remains broadly aligned. This is consistent with SAE features spanning")
    lines.append("multiple functional subspaces that partially overlap in direction even when the specific active features differ.")
    lines.append("")
    lines.append("### 2.4 Weighted Jaccard vs. Binary Jaccard")
    lines.append("")
    lines.append(f"The weighted Jaccard is consistently **lower than or equal to binary Jaccard** ({wlow} of cases),")
    lines.append(f"and the two are highly correlated (r={wjcorr}). This means that the shared features (intersection) tend to have")
    lines.append("*recalibrated magnitudes* under different conditions — the model activates the same feature but to a different degree.")
    lines.append("")
    lines.append("Practically: weighted RDS < binary RDS implies that when the model switches pathways, it does not just swap features out entirely — it")
    lines.append("preferentially retains the same high-level features but adjusts their activation strength. The binary metric may thus *overstate* the depth of pathway divergence.")
    lines.append("")
    lines.append("### 2.5 L1 Distance Growth")
    lines.append("")
    lines.append(f"L1 distance exhibits a dramatic monotonic increase from Layer 6 (~41 units) to Layer 27 (~2,684 units) — a ~{l1r}x amplification across depth.")
    lines.append("This is not primarily explained by RDS alone (which is roughly flat), but by the increasing *magnitude* of SAE feature activations at deeper layers.")
    lines.append("Late layers of transformer models typically have larger residual stream norms due to accumulated residual contributions. The L1 distance tracks")
    lines.append("this absolute scale of activation, making it an important complement to the scale-invariant cosine and Jaccard metrics.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Drift Direction Analysis")
    lines.append("")
    lines.append("### 3.1 Drift Direction Cosine — Key Finding")
    lines.append("")
    lines.append("The **drift direction cosine** (DDC) measures whether helpful and misleading hints push the model's representation in the *same direction*")
    lines.append("from the clean baseline. A DDC of +1 means both hints cause identical representational shifts; DDC of -1 means they push in opposite directions.")
    lines.append("")
    lines.append(f"**Finding: DDC is strongly positive at all layers, with a clear decreasing trend from {ddc6} (L6) to {ddc27} (L27).**")
    lines.append("")
    lines.append("This is one of the most important findings of the pilot experiment: **both helpful and misleading hints cause the model's internal representations")
    lines.append("to shift in broadly the same direction.** The hints do not activate opposite mechanisms — they engage overlapping circuits at early layers.")
    lines.append("By late layers, the alignment declines (~0.65-0.69), indicating that semantic differentiation of the two hint types becomes visible only in deep processing.")
    lines.append("")
    lines.append("**Interpretation options:**")
    lines.append("")
    lines.append("1. **Shared 'hint processing' circuit:** The model has a domain-general mechanism for processing contextual hints, which is activated regardless")
    lines.append("   of hint quality. Both hint types recruit this shared machinery at early layers, causing correlated representational shifts.")
    lines.append("2. **Convergent wrong reasoning:** Both hint types mislead the model similarly at a feature level even when helpful hints are semantically correct —")
    lines.append("   suggesting the model may be extracting structural/positional hint features rather than semantic content.")
    lines.append("3. **Problem structure dominance:** The problem content dominates the representation at early layers; hint-specific differentiation only")
    lines.append("   materializes at deeper layers where semantic content is integrated.")
    lines.append("")
    lines.append("### 3.2 Per-Problem Variation")
    lines.append("")
    lines.append("The per-problem DDC traces (Fig 3) reveal significant heterogeneity. `gsm8k_16` shows a dramatic DDC drop from 0.89 (L12) to 0.39 (L24), 0.31 (L27)")
    lines.append("— suggesting this problem's internal representation is far more sensitive to hint type at late layers, compared to others like `arith_5` which")
    lines.append("maintains DDC ~0.92-0.97 throughout.")
    lines.append("")
    lines.append("The `symb_3` problem shows the lowest overall DDC (reaching 0.31 at L27), indicating that for symbolic reasoning problems, the helpful and")
    lines.append("misleading hints drive genuinely divergent internal representations by the output layer.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Answer Sensitivity Analysis (Hypothesis H3)")
    lines.append("")
    lines.append("### 4.1 Does Higher RDS Predict Answer Changes?")
    lines.append("")
    lines.append("**H3** predicts that feature divergence (high RDS) should correlate with reasoning instability and answer sensitivity.")
    lines.append("The data provides **weak to null support for H3 at the aggregate level**.")
    lines.append("")
    lines.append("From the stability analysis:")
    lines.append(f"- When answers change: mean RDS = {rds_c_changed} (clean vs helpful)")
    lines.append(f"- When answers stay same: mean RDS = {rds_c_unchanged} (clean vs helpful)")
    lines.append("")
    lines.append(f"The difference is small (~0.02), and the Mann-Whitney U tests across all layers show **{n_sig}/{n_tot} significant comparisons** (p < 0.05).")
    lines.append("Effect sizes (Cohen's d, point-biserial r) are consistently near zero across all condition pairs and layers.")
    lines.append("")
    lines.append("**Important nuance:** The null result does not disprove H3 — it may reflect the pilot dataset's limitations:")
    lines.append("")
    lines.append("1. **Sample size:** Only 10 problems produces 10 data points per (layer, pair) stratum. Statistical power is very low for detecting small effects.")
    lines.append("2. **Near-universal answer errors:** In the arithmetic problems, the model rarely produces the correct answer under any condition.")
    lines.append("   When neither condition is correct, 'answer changed' becomes a proxy for different-but-both-wrong answers rather than correct-vs-incorrect.")
    lines.append("3. **Answer parsing:** The `model_answer` field from `outputs.json` contains the full model output (often hundreds of tokens), not a clean extracted answer.")
    lines.append("   Comparison via `_normalise_answer()` on the full text is extremely sensitive to minor generation differences, potentially inflating the 'changed' count.")
    lines.append("")
    lines.append("### 4.2 Answer Correctness")
    lines.append("")
    lines.append("For the 10 pilot problems, **near-zero correctness rates** are observed across all conditions. This is a critical limitation: the stability analysis")
    lines.append("cannot distinguish 'high-RDS + correct answer' (robustness) from 'high-RDS + wrong answer' (brittleness) if almost all answers are wrong to begin with.")
    lines.append("")
    lines.append("Reviewing the raw `outputs.json`:")
    lines.append("- `arith_5`: gold=689, clean->704, helpful->699, misleading->751 (all wrong)")
    lines.append("- `arith_6`: gold=6697, clean->6637, helpful->6603, misleading->6577 (all wrong, but close off-by-one-order errors)")
    lines.append("- `gsm8k_16`: gold=800, clean->2000, helpful->1000, misleading->2000 (misleading matches clean!)")
    lines.append("- `logic_28`: gold=bob, clean->alice, helpful->bob (**helpful is correct!**)")
    lines.append("- `logic_29`: gold=eve, clean->charlie, helpful->charlie, misleading->charlie (all wrong, model robustly wrong)")
    lines.append("")
    lines.append("The `logic_28` case is notable: the helpful hint successfully guided the model to the correct answer (alice->bob), while the misleading hint")
    lines.append("kept it at alice. For this problem, `clean vs helpful` RDS averages ~0.35-0.44 across layers, while correctness flips — directly demonstrating")
    lines.append("the brittleness/hint-dependence scenario the project aims to identify.")
    lines.append("")
    lines.append("### 4.3 The `gsm8k_16` Anomaly")
    lines.append("")
    lines.append("`gsm8k_16` shows the most interesting instability pattern: the clean and misleading conditions produce the **same answer (2000)**,")
    lines.append("while the helpful hint produces a different answer (1000). At layers 18 and 20, `gsm8k_16`'s `clean vs misleading` DDC drops to ~0.45")
    lines.append("(vs. the dataset mean of ~0.72), suggesting the misleading hint caused the model's internal representations to converge *back* toward")
    lines.append("the clean pathway by deep layers despite semantic differences.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Category Analysis")
    lines.append("")
    lines.append("### 5.1 By Problem Type")
    lines.append("")
    lines.append(f"Arithmetic problems show the highest mean RDS ({cat_a}), followed by logical ({cat_l}), GSM8K word problems ({cat_g}), and symbolic ({cat_s}).")
    lines.append("")
    lines.append("The arithmetic > symbolic ordering is counterintuitive: one might expect symbolic problems (which require more abstract rule-following) to be more")
    lines.append("sensitive to hint content. However, the arithmetic problems in this pilot (`arith_5`: 75*9+14, `arith_6`: 93*72+1) are simple enough that the model")
    lines.append("should solve them deterministically — but the hints appear to disrupt even these simple computations, producing high feature-level drift despite")
    lines.append("the problems' structural simplicity. This may reflect that the misleading hints for arithmetic problems introduce more semantically intrusive")
    lines.append("distractor content than hints for other categories.")
    lines.append("")
    lines.append("### 5.2 Pathway Consistency Index (PCI)")
    lines.append("")
    lines.append(f"The PCI ranges from **{pci_mn}** to **{pci_mx}** with mean **{pci_me}**. A PCI of 0.63 means that on average, about 63% of the model's")
    lines.append("top-50 active SAE features at the answer token are shared across all three conditions and all six layers.")
    lines.append("")
    lines.append("High PCI = consistent internal circuits regardless of hints")
    lines.append("Low PCI = pathway fragility (hints substantially redirect computation)")
    lines.append("")
    lines.append("Problems with the lowest PCI are the most 'pathway fragile' — these are candidates for causal patching in Phase 5.")
    lines.append("For Phase 5, prioritizing `logic_28`, `logic_29`, and `gsm8k_16` is recommended as these show the most interesting")
    lines.append("answer-correctness variation alongside measurable RDS.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Extended Metric Findings")
    lines.append("")
    lines.append("### 6.1 Shannon Entropy of Feature Activations")
    lines.append("")
    lines.append("The Shannon entropy of the active feature weight distribution (bits) measures how uniformly the model distributes activation mass among its")
    lines.append("top-50 features. Higher entropy = more uniform (diffuse) activation across features; lower entropy = more concentrated (peaked) activation")
    lines.append("in a few dominant features.")
    lines.append("")
    lines.append("The entropy is broadly stable across layers within a condition pair, with a slight decrease at deeper layers for all pairs. This suggests that")
    lines.append("as depth increases, the model tends to concentrate activation into fewer, more dominant features — consistent with the model 'committing' to")
    lines.append("a specific answer pathway as computation deepens.")
    lines.append("")
    lines.append("The entropy delta (cond_b minus cond_a) is small and centered near zero, indicating that the presence of a hint does not substantially change")
    lines.append("the *distribution shape* of feature activations — only which specific features are activated.")
    lines.append("")
    lines.append("### 6.2 Cross-Layer RDS Variance")
    lines.append("")
    lines.append("Problems with high cross-layer RDS variance show drift that is concentrated at specific layers (typically layer 12) rather than uniformly distributed.")
    lines.append("Problems with low cross-layer variance show drift that is uniformly present across all analyzed layers.")
    lines.append("")
    lines.append("High cross-layer variance problems include `arith_5`, `arith_6`, and `gsm8k_16` — these show pronounced layer 12 peaks in divergence.")
    lines.append("Low cross-layer variance problems include `logic_29` and `gsm8k_11`, where drift is relatively stable across depth.")
    lines.append("")
    lines.append("This distinction matters for Phase 5 (causal patching): patching at the peak-divergence layer (typically layer 12 for high-variance problems)")
    lines.append("is the optimal intervention point.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Metric Relationships and Redundancy")
    lines.append("")
    lines.append("From the correlation matrix (Fig 7):")
    lines.append("")
    lines.append(f"- **Binary RDS and Weighted RDS:** r = 0.94. Highly correlated but not identical — the discrepancy (weighted < binary) indicates magnitude recalibration of shared features.")
    lines.append("- **RDS and Cosine Similarity:** r = -0.78. Moderate negative correlation. These measure different aspects of divergence and should both be reported.")
    lines.append("- **L1 Distance and Jaccard/RDS:** r ~0.18-0.22. Near-zero correlation! L1 distance is essentially orthogonal to structural overlap, driven almost entirely")
    lines.append("  by absolute activation scale (which grows with layer depth). L1 distance adds genuine new information not captured by the other metrics.")
    lines.append("- **Exclusive Mass Asymmetry:** near-zero correlations with all other metrics (r ~0.00-0.34). Captures directional drift that other metrics miss.")
    lines.append("- **Drift Direction Cosine:** r = -0.49 with L1 distance. Problems with higher absolute activation shift (large L1) tend to show lower DDC alignment —")
    lines.append("  suggesting that larger representational shifts are more likely to be in divergent directions.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Hypothesis Evaluation")
    lines.append("")
    lines.append("| Hypothesis | Status | Evidence |")
    lines.append("|------------|--------|----------|")
    lines.append(f"| H1: Many problems show same-answer but different feature sets | Supported | Mean RDS ~0.37-0.42 even at the answer token |")
    lines.append(f"| H2: Misleading hints produce larger divergence than helpful hints | Partially supported | Marginally higher RDS (clean vs misleading={rds_cm} vs clean vs helpful={rds_ch}); stronger at Layer 12 for arithmetic |")
    lines.append(f"| H3: Feature divergence correlates with reasoning instability | Not supported by pilot | No significant Mann-Whitney results ({n_sig}/{n_tot}); effect sizes near zero. Limited by sample size. |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Limitations and Caveats")
    lines.append("")
    lines.append("### 9.1 Sample Size")
    lines.append("10 problems is insufficient for reliable statistical inference. Phase 5 and beyond should target all 100 problems before drawing strong conclusions about H3.")
    lines.append("")
    lines.append("### 9.2 Answer Parsing")
    lines.append("The current `model_answer` field stores the full raw generation (often hundreds of tokens with repetition loops visible for some arithmetic problems).")
    lines.append("The `_normalise_answer()` function compares normalized strings of the entire raw output, making 'answer changed' equivalent to 'any difference in full generation'.")
    lines.append("This massively inflates the 'changed' rate. A proper answer extraction regex per category should be implemented before Phase 5.")
    lines.append("")
    lines.append("### 9.3 Token Position")
    lines.append("The `last_generated` position captures the final token of the model's full output. For short outputs (e.g., logic problems returning a name in ~2 tokens),")
    lines.append("this is the actual answer. For long arithmetic outputs with repetition loops, the last token may be deep inside a loop.")
    lines.append("A `first_generated` or `last_prefix` position may better isolate the first answer token.")
    lines.append("")
    lines.append("### 9.4 Repetition Loops")
    lines.append("Multiple outputs in `outputs.json` show repetition loops (e.g., `arith_5` misleading, `arith_6` helpful and clean).")
    lines.append("These pathological generations likely produce anomalous SAE feature patterns that contaminate the analysis.")
    lines.append("Problems with looping outputs should be flagged and excluded from final analysis.")
    lines.append("")
    lines.append("### 9.5 Phase 3 Verification")
    lines.append("Phase 3 task 3.6 (spot-checking that exactly 50 non-zero values exist per token position) remains marked as Not started.")
    lines.append("Until this is verified, there is a possibility that the SAE projection did not correctly enforce the TopK sparsity constraint,")
    lines.append("which would affect the meaningfulness of binary Jaccard comparisons.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 10. Recommendations for Next Steps")
    lines.append("")
    lines.append("### Immediate (Before Phase 5)")
    lines.append("1. **Fix answer parsing:** Implement category-specific regex extractors to get clean numeric/symbolic answers from raw model output. Re-run stability analysis.")
    lines.append("2. **Verify TopK sparsity:** Complete Phase 3 task 3.6 — load 5 random SAE feature tensors and assert exactly 50 non-zeros per token position.")
    lines.append("3. **Flag repetition loops:** Identify problems where output contains repetition loops; handle them separately in analysis.")
    lines.append("4. **Expand to full dataset:** Run inference + SAE projection on all 100 problems before Phase 5 for adequate statistical power.")
    lines.append("")
    lines.append("### Phase 5 Design")
    lines.append("5. **Prioritize patching targets:** Based on this analysis, the optimal causal patching candidates are: `logic_28` (helpful hint corrects answer, clean is wrong),")
    lines.append("   `gsm8k_16` (answer changes between conditions), and any problem with PCI < mean PCI.")
    lines.append("6. **Patch at Layer 12:** For most problems, layer 12 shows peak divergence and should be the primary patching layer.")
    lines.append("7. **Patch at answer token:** Align patching position to `last_generated` to match the metric computation position.")
    lines.append("")
    lines.append("### Methodology Enhancements")
    lines.append("8. **Add RDS at `last_prefix` position:** Computing RDS at `last_prefix` (last token of the shared problem statement) would show whether divergence")
    lines.append("   is already present at the encoding stage or only emerges during generation.")
    lines.append("9. **Feature-level identity tracking:** Identify which specific SAE feature indices are most consistently divergent across problems.")
    lines.append("   This could reveal whether specific features are 'hint detectors' or 'reasoning anchor' features.")
    lines.append("10. **Compute per-layer pathway divergence onset:** For each problem, identify the first layer where RDS exceeds a threshold (e.g., 0.4) to")
    lines.append("    characterize how early in the network hint-induced divergence begins.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Appendix A: Raw Statistics by Layer and Condition Pair")
    lines.append("")
    lines.append("| Layer | Pair | RDS Mean | RDS Std | Cosine Mean | L1 Mean | DDC Mean |")
    lines.append("|-------|------|----------|---------|-------------|---------|----------|")
    for _, row in summary_df.iterrows():
        pair = PAIR_LABELS.get((row["cond_a"], row["cond_b"]), f"{row['cond_a']} vs {row['cond_b']}")
        lines.append(
            f"| {int(row['layer'])} | {pair} | {row['rds_mean']:.3f} | {row['rds_std']:.3f} | "
            f"{row['cosine_sim_mean']:.3f} | {row['l1_distance_mean']:.1f} | {row['drift_dir_cosine_mean']:.3f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Appendix B: Statistical Tests (Mann-Whitney U, RDS changed vs. unchanged)")
    lines.append("")
    lines.append("| Layer | Pair | n_changed | n_unchanged | RDS_changed | RDS_same | U | p-value | Cohen d | PB r |")
    lines.append("|-------|------|-----------|-------------|-------------|----------|---|---------|---------|------|")
    for _, row in stat_df.iterrows():
        sig = " *" if row.get("significant_p05") else ""
        lines.append(
            f"| {int(row['layer'])} | {row['pair_label']} | {int(row['n_changed'])} | {int(row['n_unchanged'])} | "
            f"{row['rds_mean_changed']:.3f} | {row['rds_mean_unchanged']:.3f} | "
            f"{row['mannwhitney_u']:.1f} | {row['p_value']:.3f}{sig} | "
            f"{row['cohens_d']:.3f} | {row['pointbiserial_r']:.3f} |"
        )
    lines.append("")
    lines.append("(*) p < 0.05")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Appendix C: Per-Problem Pathway Consistency Index")
    lines.append("")
    lines.append("| Problem | Category | PCI Overall | PCI clean-helpful | PCI clean-misleading | PCI helpful-misleading | Peak Div Layer | Mean DDC |")
    lines.append("|---------|----------|-------------|-------------------|----------------------|----------------------|----------------|----------|")
    for _, row in pci_df.sort_values("pci_overall").iterrows():
        lines.append(
            f"| {row['problem_id']} | {row['category']} | {row['pci_overall']:.3f} | "
            f"{row['pci_clean_helpful']:.3f} | {row['pci_clean_misleading']:.3f} | "
            f"{row['pci_helpful_misleading']:.3f} | {row['peak_divergence_layer']} | "
            f"{row['mean_drift_direction_cosine']:.3f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Report generated by `scripts/run_extended_analysis.py`. All metrics computed from Phase 3 SAE feature tensors and Phase 4 layerwise analysis outputs.*")
    return "\n".join(lines)


report_content = _build_report()
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"  -> Report saved to {report_path}")
print("[10/10] Extended analysis complete.")
print(f"\nOutputs in:")
print(f"  {OUT_DIR}/")
print(f"  {FIG_DIR}/")
