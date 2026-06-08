from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

from src.visualization.style import PAIR_COLORS, PAIR_LABELS, CAT_COLORS, LAYERS

def plot_drift_direction(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 3: Drift direction cosine (helpful vs misleading alignment)"""
    sub = detail[(detail["cond_a"] == "clean") & (detail["cond_b"] == "helpful_hint")].copy()
    problems = sorted(sub["problem_id"].unique())
    from src.visualization.style import add_category
    cat_map = add_category(sub)[["problem_id", "category"]].drop_duplicates().set_index("problem_id")["category"].to_dict()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        "Drift Direction Cosine: cos( helpful_hint − clean,  misleading_hint − clean )\n"
        "+1 = both hints activate same features  |  −1 = opposite directions",
        fontsize=12, fontweight="bold",
    )

    for pid in problems:
        p_sub = sub[sub["problem_id"] == pid].sort_values("layer")
        cat = cat_map.get(pid, "arithmetic")
        ax1.plot(
            p_sub["layer"], p_sub["drift_dir_cosine"], color=CAT_COLORS.get(cat, "gray"),
            alpha=0.7, linewidth=1.5, marker="o", markersize=4, label=f"{pid} ({cat})",
        )

    mean_ddc = sub.groupby("layer")["drift_dir_cosine"].mean()
    ax1.plot(mean_ddc.index, mean_ddc.values, color="black", linewidth=3, linestyle="--", label="mean", zorder=10)
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(4, 29)
    ax1.set_xticks(LAYERS)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("SAE Layer", fontsize=11)
    ax1.set_ylabel("Drift Direction Cosine", fontsize=11)
    ax1.set_title("Per-Problem Traces", fontsize=11)
    
    # Custom legend for categories
    from matplotlib.lines import Line2D
    cat_lines = [Line2D([0], [0], color=c, lw=2) for c in CAT_COLORS.values()][:4]
    ax1.legend(cat_lines, list(CAT_COLORS.keys())[:4], fontsize=7, ncol=2, framealpha=0.7, loc="lower left")

    pivot = sub.pivot_table(index="problem_id", columns="layer", values="drift_dir_cosine")
    pivot = pivot.reindex(problems)
    sns.heatmap(
        pivot, ax=ax2, cmap="RdYlGn", vmin=-1.0, vmax=1.0, annot=True,
        fmt=".2f", linewidths=0.5, cbar_kws={"label": "Drift Direction Cosine", "shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax2.set_title("Heatmap (green = same direction, red = opposite)", fontsize=11)
    ax2.set_xlabel("SAE Layer", fontsize=10)
    ax2.set_ylabel("Problem", fontsize=10)

    path = out / f"fig3_drift_direction.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_metric_relationships(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 7: Key metric scatter plots + correlation heatmap"""
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)
    fig.suptitle("Metric Relationships & Correlations", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ca, cb in PAIR_LABELS.keys():
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        ax1.scatter(
            sub["cosine_sim"], sub["rds"], color=PAIR_COLORS[(ca, cb)],
            label=PAIR_LABELS[(ca, cb)], alpha=0.55, s=35, edgecolors="white", linewidth=0.3,
        )

    all_cos = detail["cosine_sim"].dropna()
    all_rds = detail["rds"].dropna()
    if len(all_cos) > 0 and len(all_rds) > 0:
        z = np.polyfit(all_cos, all_rds, 1)
        p = np.poly1d(z)
        xs = np.linspace(all_cos.min(), all_cos.max(), 100)
        ax1.plot(
            xs, p(xs), color="black", linewidth=1.5, linestyle="--", alpha=0.7,
            label=f"trend (r={np.corrcoef(all_cos, all_rds)[0, 1]:.2f})",
        )

    ax1.set_xlabel("Cosine Similarity", fontsize=10)
    ax1.set_ylabel("Binary RDS", fontsize=10)
    ax1.set_title("A) RDS vs Cosine Similarity", fontsize=11)
    ax1.legend(fontsize=7, framealpha=0.7)

    for ca, cb in PAIR_LABELS.keys():
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        ax2.scatter(
            sub["rds"], sub["weighted_rds"], color=PAIR_COLORS[(ca, cb)],
            label=PAIR_LABELS[(ca, cb)], alpha=0.55, s=35, edgecolors="white", linewidth=0.3,
        )
    ax2.plot([0, 1], [0, 1], color="gray", linewidth=1, linestyle="--", label="y = x (identical)")
    ax2.set_xlabel("Binary RDS (1 − Jaccard)", fontsize=10)
    ax2.set_ylabel("Weighted RDS (1 − WJaccard)", fontsize=10)
    ax2.set_title("B) Binary vs Weighted RDS", fontsize=10)
    ax2.legend(fontsize=7, framealpha=0.7)

    metric_cols = ["jaccard", "weighted_jaccard", "cosine_sim", "rds", "weighted_rds", "l1_distance", "exclusive_mass_asymmetry", "drift_dir_cosine"]
    valid_cols = [c for c in metric_cols if c in detail.columns]
    
    corr = detail[valid_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, ax=ax3, cmap="RdBu_r", vmin=-1, vmax=1, annot=True,
        fmt=".2f", annot_kws={"size": 8}, linewidths=0.5, mask=mask,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax3.set_title("C) Metric Correlation Matrix", fontsize=11)
    ax3.tick_params(axis="both", labelsize=8)

    path = out / f"fig7_metric_relationships.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_pathway_consistency(pci_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 4: Pathway Consistency Index per problem"""
    if pci_df.empty: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Pathway Consistency Index (PCI) by Problem\n(PCI = mean Jaccard across all pairs × layers; higher = more consistent internal circuits)", fontsize=13, fontweight="bold")

    ax = axes[0]
    pci_sorted = pci_df.sort_values("pci_overall", ascending=False)
    ax.barh(pci_sorted["problem_id"], pci_sorted["pci_overall"],
            color=[CAT_COLORS.get(c, "gray") for c in pci_sorted["category"]])
    ax.axvline(pci_sorted["pci_overall"].mean(), color="black", linestyle="--", linewidth=1.5, label=f"mean={pci_sorted['pci_overall'].mean():.3f}")
    ax.set_xlabel("PCI (mean Jaccard)")
    ax.set_title("Overall PCI per Problem")
    
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=v, label=k) for k, v in CAT_COLORS.items() if k in pci_sorted["category"].values]
    ax.legend(handles=handles, loc="lower right", fontsize=9)

    ax2 = axes[1]
    for cat, grp in pci_df.groupby("category"):
        ax2.scatter(grp["pci_overall"], grp["mean_drift_direction_cosine"],
                    color=CAT_COLORS.get(cat, "gray"), label=cat, s=80, edgecolors="white", linewidths=0.5)
    ax2.set_xlabel("PCI (pathway consistency)")
    ax2.set_ylabel("Mean Drift Direction Cosine")
    ax2.set_title("PCI vs Hint-Direction Alignment")
    ax2.legend()
    
    valid = pci_df.dropna(subset=["pci_overall", "mean_drift_direction_cosine"])
    if len(valid) >= 3:
        r, p = stats.pearsonr(valid["pci_overall"], valid["mean_drift_direction_cosine"])
        ax2.annotate(f"r={r:.2f}, p={p:.3f}", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=10)

    plt.tight_layout()
    path = out / f"ext_fig4_pathway_consistency.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_drift_dir_vs_rds(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 6: Drift Direction Cosine vs RDS scatter per layer"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Drift Direction Cosine vs. RDS per SAE Layer\n(clean ↔ helpful, blue; clean ↔ misleading, red)", fontsize=13, fontweight="bold")

    for ax, layer in zip(axes.flat, LAYERS):
        for (cond_a, cond_b), color in [
            (("clean", "helpful_hint"), PAIR_COLORS[("clean", "helpful_hint")]),
            (("clean", "misleading_hint"), PAIR_COLORS[("clean", "misleading_hint")]),
        ]:
            _sub_raw = detail[(detail["layer"] == layer) & (detail["cond_a"] == cond_a) & (detail["cond_b"] == cond_b)]
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
    path = out / f"ext_fig6_drift_dir_cosine_vs_rds.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
