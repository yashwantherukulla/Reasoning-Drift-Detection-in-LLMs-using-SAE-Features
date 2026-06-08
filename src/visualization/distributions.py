from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import numpy as np

from src.visualization.style import PAIR_COLORS, PAIR_LABELS, CAT_COLORS, LAYERS

def plot_activation_distributions(detail_df: pd.DataFrame, sae_dir: Path, out: Path, fmt: str) -> None:
    """Task 6.2: SAE feature activation strength distributions across the 3 prompt conditions per layer."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    fig.suptitle("SAE Feature Activation Strengths (Non-zero values)\nAcross conditions per layer", fontsize=14, fontweight="bold")
    
    conditions = ["clean", "helpful_hint", "misleading_hint"]
    cond_colors = {"clean": "#757575", "helpful_hint": "#4CAF50", "misleading_hint": "#F44336"}
    
    for ax, layer in zip(axes.flat, LAYERS):
        layer_data = []
        # Filter for this layer, we can just take the clean vs helpful subset to get the pairs 
        # (each row has pos_a, pos_b, but actually it's easier to just iterate over the unique problem/conditions)
        sub = detail_df[detail_df["layer"] == layer].drop_duplicates(subset=["problem_id"])
        
        for _, row in sub.iterrows():
            pid = row["problem_id"]
            # We assume pos_a for clean is row['pos_a'] when cond_a is clean
            # We can just load the tensors and take the last position (or all positions?)
            # The prompt asks for distribution of feature strengths. We will take the last token.
            pos_dict = {}
            # Re-extract positions from detail_df
            for cond in conditions:
                # Find a row where this condition is cond_a or cond_b to get its pos
                r_a = detail_df[(detail_df["problem_id"] == pid) & (detail_df["layer"] == layer) & (detail_df["cond_a"] == cond)]
                r_b = detail_df[(detail_df["problem_id"] == pid) & (detail_df["layer"] == layer) & (detail_df["cond_b"] == cond)]
                
                pos = None
                if not r_a.empty:
                    pos = int(r_a.iloc[0]["pos_a"])
                elif not r_b.empty:
                    pos = int(r_b.iloc[0]["pos_b"])
                    
                if pos is not None:
                    path = sae_dir / f"{pid}_{cond}_{layer}.pt"
                    if path.exists():
                        try:
                            feat = torch.load(path, map_location="cpu", weights_only=True)
                            if feat is not None and feat.numel() > 0:
                                pos = min(pos, feat.shape[0] - 1)
                                vec = feat[pos].float()
                                nz = vec[vec > 0].numpy()
                                for v in nz:
                                    layer_data.append({"Condition": cond, "Activation": v})
                        except Exception:
                            pass
                            
        if layer_data:
            df_plot = pd.DataFrame(layer_data)
            sns.violinplot(
                data=df_plot, x="Condition", y="Activation", 
                palette=cond_colors, ax=ax, inner="quartile", cut=0
            )
            ax.set_title(f"Layer {layer}", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("Activation Magnitude" if layer in [6, 20] else "")
            
    path = out / f"fig2_activation_distributions.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_rds_distribution_by_category(detail_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 5: RDS Distribution by Problem Category Across SAE Layers"""
    from src.visualization.style import add_category
    df = add_category(detail_df)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    fig.suptitle("RDS Distribution by Problem Category Across SAE Layers", fontsize=13, fontweight="bold")

    pairs = [
        ("clean", "helpful_hint"),
        ("clean", "misleading_hint"),
        ("helpful_hint", "misleading_hint"),
    ]

    for ax, (cond_a, cond_b) in zip(axes, pairs):
        sub = df[(df["cond_a"] == cond_a) & (df["cond_b"] == cond_b)].copy()
        if sub.empty: continue
        
        sns.boxplot(
            data=sub, x="layer", y="rds", hue="category",
            palette=CAT_COLORS, ax=ax, width=0.6, fliersize=2,
        )
        ax.set_title(PAIR_LABELS[(cond_a, cond_b)], color=PAIR_COLORS[(cond_a, cond_b)], fontweight="bold")
        ax.set_xlabel("SAE Layer")
        ax.set_ylabel("RDS" if ax == axes[0] else "")
        ax.legend(title="Category", fontsize=8, title_fontsize=8)

    path = out / f"ext_fig5_rds_distribution_by_category_and_layer.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def plot_entropy_by_layer(entropy_df: pd.DataFrame, detail_df: pd.DataFrame, out: Path, fmt: str) -> None:
    """Extended Figure 3: Shannon Entropy of Active SAE Feature Weights by Layer"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    fig.suptitle("Shannon Entropy of Active SAE Feature Weights by Layer\n(higher = more uniform / spread activation)", fontsize=13, fontweight="bold")

    pairs = [
        ("clean", "helpful_hint"),
        ("clean", "misleading_hint"),
        ("helpful_hint", "misleading_hint"),
    ]

    for ax, (cond_a, cond_b) in zip(axes, pairs):
        sub = entropy_df[(entropy_df["cond_a"] == cond_a) & (entropy_df["cond_b"] == cond_b)].copy()
        if sub.empty: continue
        
        grp = sub.groupby("layer")["entropy_mean"].agg(["mean", "std"]).reindex(LAYERS)
        ax.errorbar(
            grp.index, grp["mean"], yerr=grp["std"], marker="o", capsize=4,
            color=PAIR_COLORS[(cond_a, cond_b)], linewidth=2, markersize=6,
            label="mean ± 1 std"
        )
        ax.set_title(PAIR_LABELS[(cond_a, cond_b)], fontsize=10, fontweight="bold",
                     color=PAIR_COLORS[(cond_a, cond_b)])
        ax.set_xlabel("SAE Layer")
        ax.set_ylabel("Mean Entropy (bits)" if ax == axes[0] else "")
        ax.set_xticks(LAYERS)

    path = out / f"ext_fig3_entropy_by_layer.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
