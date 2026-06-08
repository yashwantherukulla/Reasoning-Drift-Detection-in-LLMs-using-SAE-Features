from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
import numpy as np

from src.visualization.style import PAIR_COLORS, PAIR_LABELS, PAIR_MARKERS, CAT_COLORS, LAYERS

def plot_layerwise_profiles(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 2: Layerwise metric profiles across layers"""
    metrics = [
        ("rds", "RDS (1 − Jaccard)", [0.0, 1.0]),
        ("weighted_rds", "Weighted RDS (1 − WJaccard)", [0.0, 1.0]),
        ("cosine_sim", "Cosine Similarity", [0.0, 1.0]),
        ("l1_distance", "L1 Distance (log scale)", None),
    ]
    pairs = list(PAIR_LABELS.keys())

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    fig.suptitle(
        "Metric Profiles Across SAE Layers\n(mean ± 1 std across problems)",
        fontsize=13, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for ax, (metric, ylabel, ylim) in zip(axes_flat, metrics):
        for ca, cb in pairs:
            sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
            stats = sub.groupby("layer")[metric].agg(["mean", "std"]).reindex(LAYERS)
            means = stats["mean"].values
            stds = stats["std"].values
            color = PAIR_COLORS[(ca, cb)]
            label = PAIR_LABELS[(ca, cb)]
            marker = PAIR_MARKERS[(ca, cb)]

            if metric == "l1_distance":
                ax.semilogy(
                    LAYERS, means, color=color, marker=marker,
                    linewidth=2, markersize=6, label=label,
                )
                ax.fill_between(
                    LAYERS, np.maximum(means - stds, 1e-3), means + stds,
                    color=color, alpha=0.12,
                )
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))
            else:
                ax.plot(
                    LAYERS, means, color=color, marker=marker,
                    linewidth=2, markersize=6, label=label,
                )
                ax.fill_between(LAYERS, means - stds, means + stds, color=color, alpha=0.12)

        ax.set_xticks(LAYERS)
        ax.set_xlabel("SAE Layer", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3, linestyle="--")

    path = out / f"fig2_layerwise_profiles.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_problem_profiles(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 6: Small-multiple line plots of RDS across layers for each problem."""
    problems = sorted(detail["problem_id"].unique())
    n = len(problems)
    ncols = 5
    nrows = max(1, (n + ncols - 1) // ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, max(4, 4 * nrows)), constrained_layout=True, sharey=True
    )
    fig.suptitle(
        "Per-Problem RDS Profile Across SAE Layers\n(all 3 condition pairs)",
        fontsize=13, fontweight="bold",
    )

    # In case there's only 1 row, axes might be 1D array
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax_idx, pid in enumerate(problems):
        ax = axes_flat[ax_idx]
        sub = detail[detail["problem_id"] == pid]
        cat = pid.split("_")[0].replace("gsm8k", "gsm8k").replace("gsm", "gsm8k")

        for ca, cb in PAIR_LABELS.keys():
            pair_sub = sub[(sub["cond_a"] == ca) & (sub["cond_b"] == cb)].sort_values("layer")
            ax.plot(
                pair_sub["layer"], pair_sub["rds"], color=PAIR_COLORS[(ca, cb)],
                marker=PAIR_MARKERS[(ca, cb)], linewidth=2, markersize=5,
                label=PAIR_LABELS[(ca, cb)],
            )

        ax.set_title(f"{pid}", fontsize=10, fontweight="bold", color=CAT_COLORS.get(cat, "black"))
        ax.set_xticks(LAYERS)
        ax.set_xticklabels(LAYERS, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Layer" if ax_idx >= (nrows - 1) * ncols else "", fontsize=8)
        ax.set_ylabel("RDS" if ax_idx % ncols == 0 else "", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02), fontsize=9, framealpha=0.8,
    )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    path = out / f"fig6_problem_profiles.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_cross_layer_variance(var_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 8: Cross-Layer RDS Variance per Problem"""
    from src.visualization.style import CAT_COLORS
    
    if var_df.empty:
        return
        
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cross-Layer RDS Variance per Problem\n(high variance = drift is layer-specific; low variance = drift is uniform across depth)", fontsize=13, fontweight="bold")

    pairs = list(PAIR_LABELS.keys())

    for ax, (cond_a, cond_b) in zip(axes, pairs):
        sub = var_df[(var_df["cond_a"] == cond_a) & (var_df["cond_b"] == cond_b)].copy()
        if sub.empty: continue
        
        sub_sorted = sub.sort_values(by="rds_std_across_layers", ascending=False)
        ax.barh(
            sub_sorted["problem_id"], sub_sorted["rds_std_across_layers"],
            color=[CAT_COLORS.get(c, "gray") for c in sub_sorted["category"]],
        )
        ax.axvline(sub["rds_std_across_layers"].mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={sub['rds_std_across_layers'].mean():.3f}")
        ax.set_xlabel("Std Dev of RDS Across Layers")
        ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold", fontsize=10)
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = out / f"ext_fig8_cross_layer_variance.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
