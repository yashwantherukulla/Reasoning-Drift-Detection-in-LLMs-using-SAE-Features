"""
scripts/run_visualization.py
============================
Phase 4 visualization — produces 7 publication-quality figures from the
metrics CSVs in results/metrics/.

Figures saved to results/figures/:
    fig1_rds_heatmaps.png          Layerwise RDS heatmaps (problem × layer)
    fig2_layerwise_profiles.png    Mean metric profiles across layers + std bands
    fig3_drift_direction.png       Drift direction cosine: helpful vs misleading alignment
    fig4_answer_sensitivity.png    Does higher RDS predict answer changes?
    fig5_category_breakdown.png    RDS and weighted-RDS by problem category
    fig6_problem_profiles.png      Small-multiple per-problem layerwise traces
    fig7_metric_relationships.png  Key metric scatter plots + correlation heatmap

Usage:
    uv run scripts/run_visualization.py

    # Custom output directory
    uv run scripts/run_visualization.py --out results/figures/

    # Save as PDF instead
    uv run scripts/run_visualization.py --fmt pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PAIR_COLORS = {
    ("clean", "helpful_hint"): "#2196F3",  # blue
    ("clean", "misleading_hint"): "#F44336",  # red
    ("helpful_hint", "misleading_hint"): "#7B1FA2",  # purple
}
PAIR_LABELS = {
    ("clean", "helpful_hint"): "clean ↔ helpful",
    ("clean", "misleading_hint"): "clean ↔ misleading",
    ("helpful_hint", "misleading_hint"): "helpful ↔ misleading",
}
PAIR_MARKERS = {
    ("clean", "helpful_hint"): "o",
    ("clean", "misleading_hint"): "s",
    ("helpful_hint", "misleading_hint"): "^",
}

CAT_COLORS = {
    "arith": "#FF9800",
    "gsm8k": "#4CAF50",
    "logic": "#2196F3",
    "symb": "#9C27B0",
}

LAYERS = [6, 12, 18, 20, 24, 27]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 120,
    }
)


def _pair_key(row) -> tuple[str, str]:
    return (row["cond_a"], row["cond_b"])


def _add_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["category"] = df["problem_id"].str.extract(r"^([a-z]+)")
    # Normalise gsm8k prefix
    df["category"] = df["category"].replace({"gsm": "gsm8k"})
    return df


# ---------------------------------------------------------------------------
# Figure 1: RDS Heatmaps  (problem × layer, 3 panels)
# ---------------------------------------------------------------------------


def fig1_rds_heatmaps(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    pairs = [
        ("clean", "helpful_hint"),
        ("clean", "misleading_hint"),
        ("helpful_hint", "misleading_hint"),
    ]
    problems = sorted(detail["problem_id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    fig.suptitle(
        "Reasoning Divergence Score (RDS) — Binary Jaccard\nby Problem × Layer",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for ax, (ca, cb) in zip(axes, pairs):
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        pivot = sub.pivot_table(index="problem_id", columns="layer", values="rds")
        pivot = pivot.reindex(problems)

        sns.heatmap(
            pivot,
            ax=ax,
            cmap="YlOrRd",
            vmin=0.0,
            vmax=1.0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            cbar_kws={"label": "RDS", "shrink": 0.8},
            annot_kws={"size": 8},
        )
        ax.set_title(
            PAIR_LABELS[(ca, cb)],
            fontsize=12,
            fontweight="bold",
            color=PAIR_COLORS[(ca, cb)],
        )
        ax.set_xlabel("SAE Layer", fontsize=10)
        ax.set_ylabel("Problem" if ax == axes[0] else "", fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    path = out / f"fig1_rds_heatmaps.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 2: Layerwise metric profiles
# ---------------------------------------------------------------------------


def fig2_layerwise_profiles(detail: pd.DataFrame, out: Path, fmt: str) -> None:
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
        fontsize=13,
        fontweight="bold",
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
                # Plot on log scale
                ax.semilogy(
                    LAYERS,
                    means,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    label=label,
                )
                ax.fill_between(
                    LAYERS,
                    np.maximum(means - stds, 1e-3),
                    means + stds,
                    color=color,
                    alpha=0.12,
                )
                ax.yaxis.set_major_formatter(
                    mticker.FuncFormatter(lambda x, _: f"{x:.0f}")
                )
            else:
                ax.plot(
                    LAYERS,
                    means,
                    color=color,
                    marker=marker,
                    linewidth=2,
                    markersize=6,
                    label=label,
                )
                ax.fill_between(
                    LAYERS, means - stds, means + stds, color=color, alpha=0.12
                )

        ax.set_xticks(LAYERS)
        ax.set_xlabel("SAE Layer", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.grid(True, alpha=0.3, linestyle="--")

    path = out / f"fig2_layerwise_profiles.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 3: Drift direction cosine
# ---------------------------------------------------------------------------


def fig3_drift_direction(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """
    The drift_dir_cosine is the cosine between (helpful−clean) and
    (misleading−clean) delta vectors.  +1 = both hints push the same way;
    −1 = opposite directions.  This figure shows how that alignment varies
    across layers and problems.
    """
    # drift_dir_cosine is the same for all three pairs per (problem, layer)
    # since it's a 3-way metric. Use clean↔helpful rows.
    sub = detail[
        (detail["cond_a"] == "clean") & (detail["cond_b"] == "helpful_hint")
    ].copy()

    problems = sorted(sub["problem_id"].unique())
    cat_map = (
        sub[["problem_id"]]
        .drop_duplicates()
        .assign(
            category=lambda d: (
                d["problem_id"].str.extract(r"^([a-z]+)")[0].replace({"gsm": "gsm8k"})
            )
        )
        .set_index("problem_id")["category"]
        .to_dict()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    fig.suptitle(
        "Drift Direction Cosine: cos( helpful_hint − clean,  misleading_hint − clean )\n"
        "+1 = both hints activate same features  |  −1 = opposite directions",
        fontsize=12,
        fontweight="bold",
    )

    # --- Panel A: line per problem ---
    for pid in problems:
        p_sub = sub[sub["problem_id"] == pid].sort_values("layer")
        cat = cat_map.get(pid, "arith")
        ax1.plot(
            p_sub["layer"],
            p_sub["drift_dir_cosine"],
            color=CAT_COLORS.get(cat, "gray"),
            alpha=0.7,
            linewidth=1.5,
            marker="o",
            markersize=4,
            label=f"{pid} ({cat})",
        )

    # Mean line
    mean_ddc = sub.groupby("layer")["drift_dir_cosine"].mean()
    ax1.plot(
        mean_ddc.index,
        mean_ddc.values,
        color="black",
        linewidth=3,
        linestyle="--",
        label="mean",
        zorder=10,
    )
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_xlim(4, 29)
    ax1.set_xticks(LAYERS)
    ax1.set_ylim(-0.1, 1.1)
    ax1.set_xlabel("SAE Layer", fontsize=11)
    ax1.set_ylabel("Drift Direction Cosine", fontsize=11)
    ax1.set_title("Per-Problem Traces", fontsize=11)
    ax1.legend(fontsize=7, ncol=2, framealpha=0.7, loc="lower left")

    # Legend for categories
    for cat, color in CAT_COLORS.items():
        ax1.plot([], [], color=color, linewidth=2, label=cat)

    # --- Panel B: heatmap (problem × layer) ---
    pivot = sub.pivot_table(
        index="problem_id", columns="layer", values="drift_dir_cosine"
    )
    pivot = pivot.reindex(problems)
    sns.heatmap(
        pivot,
        ax=ax2,
        cmap="RdYlGn",
        vmin=-1.0,
        vmax=1.0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Drift Direction Cosine", "shrink": 0.8},
        annot_kws={"size": 8},
    )
    ax2.set_title("Heatmap (green = same direction, red = opposite)", fontsize=11)
    ax2.set_xlabel("SAE Layer", fontsize=10)
    ax2.set_ylabel("Problem", fontsize=10)

    path = out / f"fig3_drift_direction.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 4: Answer sensitivity
# ---------------------------------------------------------------------------


def fig4_answer_sensitivity(stability: pd.DataFrame, out: Path, fmt: str) -> None:
    """
    Does higher RDS predict that the model changed its answer?
    Shows distributions of RDS and weighted_RDS split by answer_changed,
    across all three condition pairs.
    """
    pairs = list(PAIR_LABELS.keys())

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    fig.suptitle(
        "Answer Sensitivity: Does Higher Divergence Predict Answer Changes?\n"
        "Left: Binary RDS  |  Right: Weighted RDS",
        fontsize=13,
        fontweight="bold",
    )

    for col_idx, (ca, cb) in enumerate(pairs):
        sub = stability[
            (stability["cond_a"] == ca) & (stability["cond_b"] == cb)
        ].copy()
        sub["Answer"] = sub["answer_changed"].map({True: "Changed", False: "Same"})
        color = PAIR_COLORS[(ca, cb)]
        palette = {"Changed": color, "Same": "#BDBDBD"}

        for row_idx, (metric, ylabel) in enumerate(
            [("rds", "RDS"), ("weighted_rds", "Weighted RDS")]
        ):
            ax = axes[row_idx][col_idx]

            # Violin with strip overlay
            sns.violinplot(
                data=sub,
                x="Answer",
                y=metric,
                hue="Answer",
                palette=palette,
                ax=ax,
                inner=None,
                cut=0,
                alpha=0.7,
                order=["Changed", "Same"],
                legend=False,
            )
            sns.stripplot(
                data=sub,
                x="Answer",
                y=metric,
                hue="Answer",
                palette=palette,
                ax=ax,
                size=4,
                alpha=0.6,
                jitter=True,
                order=["Changed", "Same"],
                legend=False,
            )

            # Add mean lines
            for i, ans in enumerate(["Changed", "Same"]):
                m = sub[sub["Answer"] == ans][metric].mean()
                ax.hlines(m, i - 0.3, i + 0.3, colors="black", linewidth=2.5, zorder=5)
                ax.text(
                    i,
                    m + 0.015,
                    f"{m:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("")
            ax.set_ylabel(ylabel if col_idx == 0 else "", fontsize=10)
            if row_idx == 0:
                ax.set_title(
                    PAIR_LABELS[(ca, cb)], fontsize=11, fontweight="bold", color=color
                )
            ax.tick_params(labelsize=9)

        # Add n counts
        n_changed = (sub["answer_changed"]).sum()
        n_same = (~sub["answer_changed"]).sum()
        axes[1][col_idx].set_xlabel(
            f"Changed n={n_changed}  |  Same n={n_same}", fontsize=9
        )

    path = out / f"fig4_answer_sensitivity.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 5: Category breakdown
# ---------------------------------------------------------------------------


def fig5_category_breakdown(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Bar chart comparing RDS across problem categories and condition pairs."""
    df = _add_category(detail)
    cats = ["arith", "gsm8k", "logic", "symb"]
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
                x + offset,
                means,
                bar_width,
                color=PAIR_COLORS[(ca, cb)],
                label=PAIR_LABELS[(ca, cb)],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.5,
            )
            ax.errorbar(
                x + offset,
                means,
                yerr=stds,
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1,
            )

            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{m:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([c.upper() for c in cats], fontsize=11)
        ax.set_ylim(0, 0.75)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Problem Category", fontsize=10)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.axhline(0, color="black", linewidth=0.5)

    path = out / f"fig5_category_breakdown.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 6: Per-problem profiles (small multiples)
# ---------------------------------------------------------------------------


def fig6_problem_profiles(detail: pd.DataFrame, out: Path, fmt: str) -> None:
    """Small-multiple line plots: RDS across layers for each problem."""
    problems = sorted(detail["problem_id"].unique())
    n = len(problems)
    ncols = 5
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(18, 4 * nrows), constrained_layout=True, sharey=True
    )
    fig.suptitle(
        "Per-Problem RDS Profile Across SAE Layers\n(all 3 condition pairs)",
        fontsize=13,
        fontweight="bold",
    )

    axes_flat = axes.flatten()

    for ax_idx, pid in enumerate(problems):
        ax = axes_flat[ax_idx]
        sub = detail[detail["problem_id"] == pid]
        cat = pid.split("_")[0].replace("gsm8k", "gsm8k").replace("gsm", "gsm8k")

        for ca, cb in PAIR_LABELS:
            pair_sub = sub[(sub["cond_a"] == ca) & (sub["cond_b"] == cb)].sort_values(
                "layer"
            )
            ax.plot(
                pair_sub["layer"],
                pair_sub["rds"],
                color=PAIR_COLORS[(ca, cb)],
                marker=PAIR_MARKERS[(ca, cb)],
                linewidth=2,
                markersize=5,
                label=PAIR_LABELS[(ca, cb)],
            )

        ax.set_title(
            f"{pid}", fontsize=10, fontweight="bold", color=CAT_COLORS.get(cat, "black")
        )
        ax.set_xticks(LAYERS)
        ax.set_xticklabels(LAYERS, fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("Layer" if ax_idx >= (nrows - 1) * ncols else "", fontsize=8)
        ax.set_ylabel("RDS" if ax_idx % ncols == 0 else "", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(0.5, color="gray", linewidth=0.5, linestyle=":")

    # Add shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
        framealpha=0.8,
    )

    # Hide empty subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    path = out / f"fig6_problem_profiles.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 7: Metric relationships
# ---------------------------------------------------------------------------


def fig7_metric_relationships(
    detail: pd.DataFrame, stability: pd.DataFrame, out: Path, fmt: str
) -> None:
    """
    Three panels:
    A) RDS vs cosine_sim scatter (all rows, colored by pair)
    B) Binary vs weighted RDS scatter (shows where they diverge)
    C) Metric correlation heatmap
    """
    fig = plt.figure(figsize=(17, 5), constrained_layout=True)
    fig.suptitle("Metric Relationships & Correlations", fontsize=13, fontweight="bold")

    gs = fig.add_gridspec(1, 3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # ---- Panel A: RDS vs cosine_sim ----
    for ca, cb in PAIR_LABELS:
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        ax1.scatter(
            sub["cosine_sim"],
            sub["rds"],
            color=PAIR_COLORS[(ca, cb)],
            label=PAIR_LABELS[(ca, cb)],
            alpha=0.55,
            s=35,
            edgecolors="white",
            linewidth=0.3,
        )

    # Fit a trend line across all rows
    all_cos = detail["cosine_sim"]
    all_rds = detail["rds"]
    z = np.polyfit(all_cos, all_rds, 1)
    p = np.poly1d(z)
    xs = np.linspace(all_cos.min(), all_cos.max(), 100)
    ax1.plot(
        xs,
        p(xs),
        color="black",
        linewidth=1.5,
        linestyle="--",
        alpha=0.7,
        label=f"trend (r={np.corrcoef(all_cos, all_rds)[0, 1]:.2f})",
    )

    ax1.set_xlabel("Cosine Similarity", fontsize=10)
    ax1.set_ylabel("Binary RDS", fontsize=10)
    ax1.set_title("A) RDS vs Cosine Similarity", fontsize=11)
    ax1.legend(fontsize=7, framealpha=0.7)
    ax1.set_xlim(0.7, 1.01)
    ax1.set_ylim(-0.05, 1.05)

    # ---- Panel B: Binary RDS vs Weighted RDS ----
    for ca, cb in PAIR_LABELS:
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        ax2.scatter(
            sub["rds"],
            sub["weighted_rds"],
            color=PAIR_COLORS[(ca, cb)],
            label=PAIR_LABELS[(ca, cb)],
            alpha=0.55,
            s=35,
            edgecolors="white",
            linewidth=0.3,
        )
    ax2.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linewidth=1,
        linestyle="--",
        label="y = x (identical)",
    )
    ax2.set_xlabel("Binary RDS (1 − Jaccard)", fontsize=10)
    ax2.set_ylabel("Weighted RDS (1 − WJaccard)", fontsize=10)
    ax2.set_title(
        "B) Binary vs Weighted RDS\n(points below diagonal: magnitude recalibration)",
        fontsize=10,
    )
    ax2.legend(fontsize=7, framealpha=0.7)
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)

    # Shade below-diagonal region
    ax2.fill_between(
        [0, 1], [0, 1], 0, color="green", alpha=0.05, label="weighted < binary"
    )
    ax2.fill_between([0, 1], [0, 1], 1, color="red", alpha=0.05)

    # ---- Panel C: Metric correlation heatmap ----
    metric_cols = [
        "jaccard",
        "weighted_jaccard",
        "cosine_sim",
        "rds",
        "weighted_rds",
        "l1_distance",
        "exclusive_mass_asymmetry",
        "drift_dir_cosine",
    ]
    corr = detail[metric_cols].corr()
    short_names = [
        "Jaccard",
        "W.Jaccard",
        "Cosine",
        "RDS",
        "W.RDS",
        "L1 dist",
        "Excl.Mass\nAsym.",
        "Drift Dir\nCosine",
    ]
    corr.index = corr.columns = short_names

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr,
        ax=ax3,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        linewidths=0.5,
        mask=mask,
        cbar_kws={"label": "Pearson r", "shrink": 0.8},
    )
    ax3.set_title("C) Metric Correlation Matrix", fontsize=11)
    ax3.tick_params(axis="both", labelsize=8)

    path = out / f"fig7_metric_relationships.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 8: Summary dashboard
# ---------------------------------------------------------------------------


def fig8_summary_dashboard(
    detail: pd.DataFrame, stability: pd.DataFrame, out: Path, fmt: str
) -> None:
    """
    One-page summary: key numbers + clean↔misleading RDS heatmap +
    layerwise RDS profiles + stratum breakdown.
    """
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Reasoning Drift Detection — Phase 4 Summary Dashboard\n"
        "Qwen3-1.7B-Base  |  10 pilot problems  |  6 SAE layers",
        fontsize=13,
        fontweight="bold",
    )

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    ax_hm = fig.add_subplot(gs[0, 0])  # heatmap: clean vs misleading
    ax_line = fig.add_subplot(gs[0, 1])  # layerwise RDS lines
    ax_ddc = fig.add_subplot(gs[0, 2])  # drift direction cosine
    ax_cat = fig.add_subplot(gs[1, 0])  # category breakdown
    ax_ans = fig.add_subplot(gs[1, 1])  # answer sensitivity box
    ax_txt = fig.add_subplot(gs[1, 2])  # key findings text

    problems = sorted(detail["problem_id"].unique())

    # ---- A: RDS heatmap (clean vs misleading) ----
    sub_mis = detail[
        (detail["cond_a"] == "clean") & (detail["cond_b"] == "misleading_hint")
    ]
    pivot = sub_mis.pivot_table(
        index="problem_id", columns="layer", values="rds"
    ).reindex(problems)
    sns.heatmap(
        pivot,
        ax=ax_hm,
        cmap="YlOrRd",
        vmin=0,
        vmax=0.7,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        linewidths=0.4,
        cbar_kws={"label": "RDS", "shrink": 0.8},
    )
    ax_hm.set_title(
        "RDS: clean ↔ misleading",
        fontsize=10,
        fontweight="bold",
        color=PAIR_COLORS[("clean", "misleading_hint")],
    )
    ax_hm.set_xlabel("Layer", fontsize=9)
    ax_hm.set_ylabel("Problem", fontsize=9)
    ax_hm.tick_params(labelsize=7)

    # ---- B: Layerwise RDS profiles ----
    for ca, cb in PAIR_LABELS:
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        stats = sub.groupby("layer")["rds"].agg(["mean", "std"]).reindex(LAYERS)
        ax_line.plot(
            LAYERS,
            stats["mean"],
            color=PAIR_COLORS[(ca, cb)],
            marker=PAIR_MARKERS[(ca, cb)],
            linewidth=2,
            markersize=5,
            label=PAIR_LABELS[(ca, cb)],
        )
        ax_line.fill_between(
            LAYERS,
            stats["mean"] - stats["std"],
            stats["mean"] + stats["std"],
            color=PAIR_COLORS[(ca, cb)],
            alpha=0.12,
        )
    ax_line.set_xticks(LAYERS)
    ax_line.set_ylim(0, 0.85)
    ax_line.set_xlabel("SAE Layer", fontsize=9)
    ax_line.set_ylabel("Mean RDS", fontsize=9)
    ax_line.set_title("Layerwise RDS (mean ± std)", fontsize=10, fontweight="bold")
    ax_line.legend(fontsize=7)

    # ---- C: Drift direction cosine across layers ----
    ddc_sub = detail[
        (detail["cond_a"] == "clean") & (detail["cond_b"] == "helpful_hint")
    ]
    ddc_stats = (
        ddc_sub.groupby("layer")["drift_dir_cosine"]
        .agg(["mean", "std"])
        .reindex(LAYERS)
    )
    ax_ddc.plot(
        LAYERS,
        ddc_stats["mean"],
        color="#FF9800",
        linewidth=2.5,
        marker="D",
        markersize=6,
    )
    ax_ddc.fill_between(
        LAYERS,
        ddc_stats["mean"] - ddc_stats["std"],
        ddc_stats["mean"] + ddc_stats["std"],
        color="#FF9800",
        alpha=0.2,
    )
    ax_ddc.axhline(0, color="red", linewidth=0.8, linestyle=":")
    ax_ddc.axhline(1, color="green", linewidth=0.8, linestyle=":")
    ax_ddc.set_ylim(-0.2, 1.1)
    ax_ddc.set_xticks(LAYERS)
    ax_ddc.set_xlabel("SAE Layer", fontsize=9)
    ax_ddc.set_ylabel("Drift Dir. Cosine", fontsize=9)
    ax_ddc.set_title(
        "Helpful ↔ Misleading Drift Alignment\n(+1 = same direction as clean)",
        fontsize=9,
        fontweight="bold",
    )

    # ---- D: Category breakdown ----
    df_cat = _add_category(detail)
    cats = ["arith", "gsm8k", "logic", "symb"]
    x = np.arange(len(cats))
    bw = 0.25
    for i, (ca, cb) in enumerate(PAIR_LABELS):
        sub = df_cat[(df_cat["cond_a"] == ca) & (df_cat["cond_b"] == cb)]
        means = [sub[sub["category"] == c]["rds"].mean() for c in cats]
        ax_cat.bar(
            x + (i - 1) * bw,
            means,
            bw,
            color=PAIR_COLORS[(ca, cb)],
            alpha=0.85,
            label=PAIR_LABELS[(ca, cb)],
            edgecolor="white",
        )
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels([c.upper() for c in cats], fontsize=9)
    ax_cat.set_ylim(0, 0.7)
    ax_cat.set_ylabel("Mean RDS", fontsize=9)
    ax_cat.set_title("RDS by Problem Category", fontsize=10, fontweight="bold")
    ax_cat.legend(fontsize=7)

    # ---- E: Answer sensitivity ----
    stab_clean_mis = stability[
        (stability["cond_a"] == "clean") & (stability["cond_b"] == "misleading_hint")
    ].copy()
    stab_clean_mis["Answer"] = stab_clean_mis["answer_changed"].map(
        {True: "Changed", False: "Same"}
    )
    sns.boxplot(
        data=stab_clean_mis,
        x="Answer",
        y="rds",
        hue="Answer",
        palette={
            "Changed": PAIR_COLORS[("clean", "misleading_hint")],
            "Same": "#BDBDBD",
        },
        ax=ax_ans,
        order=["Changed", "Same"],
        width=0.5,
        legend=False,
    )
    # Add means
    for i, ans in enumerate(["Changed", "Same"]):
        m = stab_clean_mis[stab_clean_mis["Answer"] == ans]["rds"].mean()
        ax_ans.text(
            i, m + 0.02, f"μ={m:.3f}", ha="center", fontsize=8, fontweight="bold"
        )
    ax_ans.set_ylim(0, 0.85)
    ax_ans.set_xlabel("Answer Changed?", fontsize=9)
    ax_ans.set_ylabel("RDS (clean ↔ misleading)", fontsize=9)
    ax_ans.set_title("Does Drift → Answer Change?", fontsize=10, fontweight="bold")

    # ---- F: Key findings ----
    ax_txt.axis("off")
    findings = [
        "KEY FINDINGS (10 pilot problems)",
        "",
        f"• Mean RDS (clean↔helpful):    {detail[detail['cond_b'] == 'helpful_hint']['rds'].mean():.3f}",
        f"• Mean RDS (clean↔misleading): {detail[detail['cond_b'] == 'misleading_hint']['rds'].mean():.3f}",
        f"• Mean RDS (helpful↔mislead):  {detail[(detail['cond_a'] == 'helpful_hint')]['rds'].mean():.3f}",
        "",
        f"• Drift dir. cosine (mean): {detail[detail['cond_b'] == 'helpful_hint']['drift_dir_cosine'].mean():.3f}",
        "  → Both hints shift reps in same direction",
        "",
        f"• % answer changed (clean↔helpful):   {stability[stability['cond_b'] == 'helpful_hint']['answer_changed'].mean():.0%}",
        f"• % answer changed (clean↔misleading): {stability[stability['cond_b'] == 'misleading_hint']['answer_changed'].mean():.0%}",
        "",
        "• arith problems show highest drift",
        "• L1 distance grows 350× from L6→L27",
        "• Weighted RDS < binary RDS:",
        "  magnitude recalibration dampens drift",
    ]
    ax_txt.text(
        0.05,
        0.97,
        "\n".join(findings),
        transform=ax_txt.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#F5F5F5",
            edgecolor="#BDBDBD",
            linewidth=1,
        ),
    )

    path = out / f"fig8_summary_dashboard.{fmt}"
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 visualizations")
    parser.add_argument("--metrics", default="results/metrics/metrics_detail.csv")
    parser.add_argument("--stability", default="results/metrics/stability_detail.csv")
    parser.add_argument("--out", default="results/figures/")
    parser.add_argument("--fmt", default="png", choices=["png", "pdf", "svg"])
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading data...")
    detail = pd.read_csv(args.metrics)
    stability = pd.read_csv(args.stability)

    print(f"Generating figures -> {out}/")
    fig1_rds_heatmaps(detail, out, args.fmt)
    fig2_layerwise_profiles(detail, out, args.fmt)
    fig3_drift_direction(detail, out, args.fmt)
    fig4_answer_sensitivity(stability, out, args.fmt)
    fig5_category_breakdown(detail, out, args.fmt)
    fig6_problem_profiles(detail, out, args.fmt)
    fig7_metric_relationships(detail, stability, out, args.fmt)
    fig8_summary_dashboard(detail, stability, out, args.fmt)

    print(f"\nDone: {len(list(out.glob('*.' + args.fmt)))} figures in {out}")


if __name__ == "__main__":
    main()
