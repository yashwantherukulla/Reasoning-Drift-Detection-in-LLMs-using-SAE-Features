from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.visualization.style import PAIR_COLORS, PAIR_LABELS

def plot_rds_histograms(stability: pd.DataFrame, out: Path, fmt: str) -> None:
    """Task 6.3: RDS distribution histograms (overall and stratified by correct/incorrect answer)"""
    pairs = list(PAIR_LABELS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle("RDS Distributions: Answer Changed vs Same", fontsize=13, fontweight="bold")

    for col_idx, (ca, cb) in enumerate(pairs):
        sub = stability[(stability["cond_a"] == ca) & (stability["cond_b"] == cb)].copy()
        sub["Answer"] = sub["answer_changed"].map({True: "Changed", False: "Same"})
        color = PAIR_COLORS[(ca, cb)]
        palette = {"Changed": color, "Same": "#BDBDBD"}

        # Row 0: Violin plot (like original fig4)
        ax = axes[0][col_idx]
        sns.violinplot(
            data=sub, x="Answer", y="rds", hue="Answer",
            palette=palette, ax=ax, inner=None, cut=0, alpha=0.7,
            order=["Changed", "Same"], legend=False
        )
        sns.stripplot(
            data=sub, x="Answer", y="rds", hue="Answer",
            palette=palette, ax=ax, size=4, alpha=0.6, jitter=True,
            order=["Changed", "Same"], legend=False
        )
        
        for i, ans in enumerate(["Changed", "Same"]):
            m = sub[sub["Answer"] == ans]["rds"].mean()
            ax.hlines(m, i - 0.3, i + 0.3, colors="black", linewidth=2.5, zorder=5)
            ax.text(i, m + 0.015, f"{m:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("")
        ax.set_ylabel("RDS" if col_idx == 0 else "", fontsize=10)
        ax.set_title(PAIR_LABELS[(ca, cb)], fontsize=11, fontweight="bold", color=color)
        
        # Row 1: Histogram representation (the exact Task 6.3 request)
        ax2 = axes[1][col_idx]
        sns.histplot(
            data=sub, x="rds", hue="Answer", palette=palette, 
            multiple="layer", bins=15, ax=ax2, alpha=0.5
        )
        ax2.set_xlabel("RDS")
        ax2.set_ylabel("Count" if col_idx == 0 else "")
        
        # Add n counts
        n_changed = (sub["answer_changed"]).sum()
        n_same = (~sub["answer_changed"]).sum()
        ax2.set_title(f"Changed n={n_changed}  |  Same n={n_same}", fontsize=9)

    path = out / f"fig3_rds_histograms.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_category_breakdown(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 5: Bar chart comparing RDS across problem categories and condition pairs."""
    from src.visualization.style import add_category
    df = add_category(detail)
    cats = ["arithmetic", "gsm8k", "logical", "symbolic"]
    pairs = list(PAIR_LABELS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    fig.suptitle("Divergence by Problem Category", fontsize=13, fontweight="bold")

    for ax, metric, ylabel in zip(
        axes,
        ["rds", "weighted_rds"],
        ["Binary RDS (1 − Jaccard)", "Weighted RDS (1 − Weighted Jaccard)"],
    ):
        x = np.arange(len(cats))
        bar_width = 0.25

        for i, (ca, cb) in enumerate(pairs):
            sub = df[(df["cond_a"] == ca) & (df["cond_b"] == cb)]
            means = [sub[sub["category"] == cat][metric].mean() for cat in cats]
            stds = [sub[sub["category"] == cat][metric].std() for cat in cats]
            offset = (i - 1) * bar_width

            bars = ax.bar(
                x + offset, means, bar_width, color=PAIR_COLORS[(ca, cb)],
                label=PAIR_LABELS[(ca, cb)], alpha=0.85, edgecolor="white", linewidth=0.5
            )
            ax.errorbar(x + offset, means, yerr=stds, fmt="none", color="black", capsize=3, linewidth=1)

            for bar, m in zip(bars, means):
                if not np.isnan(m):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{m:.2f}", ha="center", va="bottom", fontsize=7
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cats], fontsize=11)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Problem Category", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)

    path = out / f"fig5_category_breakdown.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
