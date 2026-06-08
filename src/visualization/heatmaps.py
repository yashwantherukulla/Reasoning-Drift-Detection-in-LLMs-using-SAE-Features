from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.visualization.style import PAIR_COLORS, PAIR_LABELS

def plot_rds_heatmaps(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 1: Layerwise overlap heatmaps (x=layer, y=problem, color=RDS)"""
    pairs = [
        ("clean", "helpful_hint"),
        ("clean", "misleading_hint"),
        ("helpful_hint", "misleading_hint"),
    ]
    problems = sorted(detail["problem_id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "Reasoning Divergence Score (RDS) — Binary Jaccard\nby Problem × Layer",
        fontsize=14, fontweight="bold", y=1.02
    )

    for ax, (ca, cb) in zip(axes, pairs):
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        pivot = sub.pivot_table(index="problem_id", columns="layer", values="rds")
        pivot = pivot.reindex(problems)

        sns.heatmap(
            pivot, ax=ax, cmap="YlOrRd", vmin=0.0, vmax=1.0, annot=True,
            fmt=".2f", linewidths=0.5, cbar_kws={"label": "RDS", "shrink": 0.8},
            annot_kws={"size": 8}
        )
        ax.set_title(PAIR_LABELS[(ca, cb)], fontsize=12, fontweight="bold", color=PAIR_COLORS[(ca, cb)])
        ax.set_xlabel("SAE Layer", fontsize=10)
        ax.set_ylabel("Problem" if ax == axes[0] else "", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    path = out / f"fig1_rds_heatmaps.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_statistical_tests(stat_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 1 & 2: Statistical tests and effect sizes (Mann-Whitney U, Cohen's d)"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Statistical Significance & Effect Sizes: Higher RDS When Answer Changes?\n(Mann-Whitney U Test)", fontsize=13, fontweight="bold")

    for ax, (col, label, cmap, vmin, vmax, center, str_fmt) in zip(axes, [
        ("p_value", "p-value (two-sided, α=0.05)", "RdYlGn_r", 0, 1, 0.05, ".3f"),
        ("cohens_d", "Cohen's d (positive = changed > unchanged)", "coolwarm", None, None, 0, ".2f"),
    ]):
        pivot = stat_df.pivot_table(index="pair_label", columns="layer", values=col, aggfunc="first")
        mask = pivot.isna()
        sns.heatmap(
            pivot, annot=True, fmt=str_fmt, cmap=cmap, center=center,
            vmin=vmin, vmax=vmax, ax=ax, linewidths=0.5, mask=mask, annot_kws={"size": 9}
        )
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("SAE Layer")
        ax.set_ylabel("")

    plt.tight_layout()
    path = out / f"ext_fig1_statistical_tests.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
