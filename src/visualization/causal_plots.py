from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.visualization.style import PAIR_COLORS, PAIR_LABELS

def plot_causal_intervention(patching_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Task 6.4: Causal intervention results: bar chart of % answer changed by n_features_patched and layer."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if patching_df.empty:
        ax.text(0.5, 0.5, "No patching data available", ha="center", va="center", transform=ax.transAxes)
    else:
        # Group by layer and n_features_patched to get the % answer changed
        grp = patching_df.groupby(["layer_patched", "n_features_patched"])["answer_changed"].mean().reset_index()
        
        sns.barplot(
            data=grp, x="layer_patched", y="answer_changed", 
            hue="n_features_patched", ax=ax, palette="viridis"
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Fraction Answer Changed")
        ax.set_xlabel("Layer Patched")
        
        # Add labels on top of bars
        for p in ax.patches:
            h = p.get_height()
            if not np.isnan(h):
                ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h),
                            ha='center', va='center', xytext=(0, 5), textcoords='offset points', fontsize=8)

    ax.set_title("Causal Intervention: Answer Change by Patched Features", fontweight="bold")
    
    path = out / f"fig4_causal_intervention.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_answer_correctness_stratified(stability: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 7: Answer Correctness Rate Stratified by RDS Quartile"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Answer Correctness Rate Stratified by RDS Quartile\n(does higher structural drift correlate with lower accuracy?)", fontsize=13, fontweight="bold")

    pairs = list(PAIR_LABELS.keys())

    for ax, (cond_a, cond_b) in zip(axes, pairs):
        sub = stability[(stability["cond_a"] == cond_a) & (stability["cond_b"] == cond_b)].copy()
        
        # Only process if we have enough data to do quartiles
        if len(sub) >= 4:
            try:
                sub["rds_quartile"] = pd.qcut(sub["rds"], q=4, labels=["Q1\n(low drift)", "Q2", "Q3", "Q4\n(high drift)"], duplicates='drop')
                
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
            except ValueError:
                ax.text(0.5, 0.5, "Insufficient variance for quartiles", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
            
        ax.set_ylim(0, 1)
        ax.set_ylabel("Fraction Correct" if ax == axes[0] else "")
        ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold")
        ax.legend(fontsize=8)

    path = out / f"ext_fig7_answer_correctness_stratified.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
