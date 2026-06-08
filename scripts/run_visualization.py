"""
scripts/run_visualization.py
============================
Phase 6 visualization — produces all figures by merging standard and extended analysis.

Usage:
    uv run scripts/run_visualization.py
    # Override formats/outputs via Hydra
    uv run scripts/run_visualization.py visualization.figure_format=pdf
"""

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig
import pandas as pd

from src.config import register_configs

# Register hydra config
register_configs()

from src.visualization import (
    set_global_style,
    plot_rds_heatmaps,
    plot_statistical_tests,
    plot_activation_distributions,
    plot_rds_distribution_by_category,
    plot_entropy_by_layer,
    plot_rds_histograms,
    plot_category_breakdown,
    plot_causal_intervention,
    plot_answer_correctness_stratified,
    plot_layerwise_profiles,
    plot_problem_profiles,
    plot_cross_layer_variance,
    plot_drift_direction,
    plot_metric_relationships,
    plot_pathway_consistency,
    plot_drift_dir_vs_rds,
    plot_summary_dashboard,
)

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    out = Path(cfg.visualization.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = cfg.visualization.figure_format
    dpi = cfg.visualization.dpi

    set_global_style(dpi=dpi)

    print(f"Loading data...")
    # Standard metrics
    metrics_path = Path(cfg.analysis.results_dir) / "metrics_detail.csv"
    stability_path = Path(cfg.analysis.results_dir) / "stability_detail.csv"
    patching_path = Path(cfg.patching.results_dir) / "patching_results.csv"
    sae_dir = Path(cfg.activations.sae_dir)
    
    # Extended metrics
    ext_dir = Path("results/analysis")
    stat_path = ext_dir / "statistical_tests.csv"
    entropy_path = ext_dir / "feature_entropy.csv"
    var_path = ext_dir / "layer_variance.csv"
    pci_path = ext_dir / "pathway_consistency.csv"

    # We use empty dataframes if some files don't exist yet so it doesn't crash
    detail = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    stability = pd.read_csv(stability_path) if stability_path.exists() else pd.DataFrame()
    patching = pd.read_csv(patching_path) if patching_path.exists() else pd.DataFrame()

    stat_df = pd.read_csv(stat_path) if stat_path.exists() else pd.DataFrame()
    entropy_df = pd.read_csv(entropy_path) if entropy_path.exists() else pd.DataFrame()
    var_df = pd.read_csv(var_path) if var_path.exists() else pd.DataFrame()
    pci_df = pd.read_csv(pci_path) if pci_path.exists() else pd.DataFrame()

    print(f"Generating figures -> {out}/")

    if not detail.empty:
        plot_rds_heatmaps(detail, out, fmt)
        plot_layerwise_profiles(detail, out, fmt)
        plot_problem_profiles(detail, out, fmt)
        plot_category_breakdown(detail, out, fmt)
        plot_drift_direction(detail, out, fmt)
        plot_drift_dir_vs_rds(detail, out, fmt)
        plot_metric_relationships(detail, out, fmt)
        plot_activation_distributions(detail, sae_dir, out, fmt)
        plot_rds_distribution_by_category(detail, out, fmt)
        
    if not stability.empty:
        plot_rds_histograms(stability, out, fmt)
        plot_answer_correctness_stratified(stability, out, fmt)

    if not patching.empty:
        plot_causal_intervention(patching, out, fmt)

    if not stat_df.empty:
        plot_statistical_tests(stat_df, out, fmt)

    if not entropy_df.empty:
        plot_entropy_by_layer(entropy_df, detail, out, fmt)

    if not var_df.empty:
        plot_cross_layer_variance(var_df, out, fmt)

    if not pci_df.empty:
        plot_pathway_consistency(pci_df, out, fmt)

    if not detail.empty and not stability.empty:
        plot_summary_dashboard(detail, stability, out, fmt)

    print(f"\nDone: {len(list(out.glob('*.' + fmt)))} figures in {out}")

if __name__ == "__main__":
    main()
