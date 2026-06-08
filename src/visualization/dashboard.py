from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from src.visualization.style import PAIR_COLORS, PAIR_LABELS, PAIR_MARKERS, CAT_COLORS, LAYERS

def plot_summary_dashboard(detail: pd.DataFrame, stability: pd.DataFrame, out: Path, fmt: str) -> None:
    """Figure 8: One-page summary dashboard"""
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    fig.suptitle(
        "Reasoning Drift Detection — Phase 4 Summary Dashboard\n"
        "Qwen3-1.7B-Base  |  Pilot problems  |  6 SAE layers",
        fontsize=13, fontweight="bold",
    )

    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)
    ax_hm = fig.add_subplot(gs[0, 0])
    ax_line = fig.add_subplot(gs[0, 1])
    ax_ddc = fig.add_subplot(gs[0, 2])
    ax_cat = fig.add_subplot(gs[1, 0])
    ax_ans = fig.add_subplot(gs[1, 1])
    ax_txt = fig.add_subplot(gs[1, 2])

    problems = sorted(detail["problem_id"].unique())

    # A: RDS heatmap (clean vs misleading)
    sub_mis = detail[(detail["cond_a"] == "clean") & (detail["cond_b"] == "misleading_hint")]
    pivot = sub_mis.pivot_table(index="problem_id", columns="layer", values="rds").reindex(problems)
    sns.heatmap(
        pivot, ax=ax_hm, cmap="YlOrRd", vmin=0, vmax=0.7, annot=True,
        fmt=".2f", annot_kws={"size": 7}, linewidths=0.4, cbar_kws={"label": "RDS", "shrink": 0.8},
    )
    ax_hm.set_title("RDS: clean ↔ misleading", fontsize=10, fontweight="bold", color=PAIR_COLORS[("clean", "misleading_hint")])
    ax_hm.set_xlabel("Layer", fontsize=9)
    ax_hm.set_ylabel("Problem", fontsize=9)
    ax_hm.tick_params(labelsize=7)

    # B: Layerwise RDS profiles
    for ca, cb in PAIR_LABELS.keys():
        sub = detail[(detail["cond_a"] == ca) & (detail["cond_b"] == cb)]
        stats = sub.groupby("layer")["rds"].agg(["mean", "std"]).reindex(LAYERS)
        ax_line.plot(
            LAYERS, stats["mean"], color=PAIR_COLORS[(ca, cb)], marker=PAIR_MARKERS[(ca, cb)],
            linewidth=2, markersize=5, label=PAIR_LABELS[(ca, cb)],
        )
        ax_line.fill_between(LAYERS, stats["mean"] - stats["std"], stats["mean"] + stats["std"], color=PAIR_COLORS[(ca, cb)], alpha=0.12)
    ax_line.set_xticks(LAYERS)
    ax_line.set_ylim(0, 0.85)
    ax_line.set_xlabel("SAE Layer", fontsize=9)
    ax_line.set_ylabel("Mean RDS", fontsize=9)
    ax_line.set_title("Layerwise RDS (mean ± std)", fontsize=10, fontweight="bold")
    ax_line.legend(fontsize=7)

    # C: Drift direction cosine across layers
    ddc_sub = detail[(detail["cond_a"] == "clean") & (detail["cond_b"] == "helpful_hint")]
    ddc_stats = ddc_sub.groupby("layer")["drift_dir_cosine"].agg(["mean", "std"]).reindex(LAYERS)
    ax_ddc.plot(LAYERS, ddc_stats["mean"], color="#FF9800", linewidth=2.5, marker="D", markersize=6)
    ax_ddc.fill_between(LAYERS, ddc_stats["mean"] - ddc_stats["std"], ddc_stats["mean"] + ddc_stats["std"], color="#FF9800", alpha=0.2)
    ax_ddc.axhline(0, color="red", linewidth=0.8, linestyle=":")
    ax_ddc.axhline(1, color="green", linewidth=0.8, linestyle=":")
    ax_ddc.set_ylim(-0.2, 1.1)
    ax_ddc.set_xticks(LAYERS)
    ax_ddc.set_xlabel("SAE Layer", fontsize=9)
    ax_ddc.set_ylabel("Drift Dir. Cosine", fontsize=9)
    ax_ddc.set_title("Helpful ↔ Misleading Drift Alignment\n(+1 = same direction as clean)", fontsize=9, fontweight="bold")

    # D: Category breakdown
    from src.visualization.style import add_category
    df_cat = add_category(detail)
    cats = ["arithmetic", "gsm8k", "logical", "symbolic"]
    x = np.arange(len(cats))
    bw = 0.25
    for i, (ca, cb) in enumerate(PAIR_LABELS.keys()):
        sub = df_cat[(df_cat["cond_a"] == ca) & (df_cat["cond_b"] == cb)]
        means = [sub[sub["category"] == c]["rds"].mean() for c in cats]
        ax_cat.bar(x + (i - 1) * bw, means, bw, color=PAIR_COLORS[(ca, cb)], alpha=0.85, label=PAIR_LABELS[(ca, cb)], edgecolor="white")
    ax_cat.set_xticks(x)
    ax_cat.set_xticklabels([c.upper() for c in cats], fontsize=9)
    ax_cat.set_ylim(0, 0.7)
    ax_cat.set_ylabel("Mean RDS", fontsize=9)
    ax_cat.set_title("RDS by Problem Category", fontsize=10, fontweight="bold")
    ax_cat.legend(fontsize=7)

    # E: Answer sensitivity
    stab_clean_mis = stability[(stability["cond_a"] == "clean") & (stability["cond_b"] == "misleading_hint")].copy()
    stab_clean_mis["Answer"] = stab_clean_mis["answer_changed"].map({True: "Changed", False: "Same"})
    sns.boxplot(
        data=stab_clean_mis, x="Answer", y="rds", hue="Answer",
        palette={"Changed": PAIR_COLORS[("clean", "misleading_hint")], "Same": "#BDBDBD"},
        ax=ax_ans, order=["Changed", "Same"], width=0.5, legend=False,
    )
    for i, ans in enumerate(["Changed", "Same"]):
        m = stab_clean_mis[stab_clean_mis["Answer"] == ans]["rds"].mean()
        if not np.isnan(m):
            ax_ans.text(i, m + 0.02, f"μ={m:.3f}", ha="center", fontsize=8, fontweight="bold")
    ax_ans.set_ylim(0, 0.85)
    ax_ans.set_xlabel("Answer Changed?", fontsize=9)
    ax_ans.set_ylabel("RDS (clean ↔ misleading)", fontsize=9)
    ax_ans.set_title("Does Drift → Answer Change?", fontsize=10, fontweight="bold")

    # F: Key findings
    ax_txt.axis("off")
    findings = [
        "KEY FINDINGS",
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
        "• Weighted RDS < binary RDS:",
        "  magnitude recalibration dampens drift",
    ]
    ax_txt.text(
        0.05, 0.97, "\n".join(findings), transform=ax_txt.transAxes, fontsize=9,
        verticalalignment="top", fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5", edgecolor="#BDBDBD", linewidth=1),
    )

    path = out / f"fig8_summary_dashboard.{fmt}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")
