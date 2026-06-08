"""
main.py
=======
Top-level Hydra entry point for the Reasoning Drift Detection pipeline.

Usage:
    uv run main.py                          # default — prints resolved config
    uv run main.py --cfg job               # inspect resolved config without running
    uv run main.py model.device=cpu        # CLI override
    uv run main.py model=qwen3_1_7b        # swap model config group

Individual pipeline stages each have their own script in scripts/:
    uv run scripts/curate_problems.py
    uv run scripts/build_dataset.py
    uv run scripts/run_inference.py
    uv run scripts/run_sae_projection.py
    uv run scripts/run_analysis.py
    uv run scripts/run_patching.py
    uv run scripts/run_visualization.py
"""

import hydra
from omegaconf import DictConfig, OmegaConf

from src.config import register_configs

register_configs()


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Reasoning Drift Detection in LLMs using SAE Features")
    print("======================================================")
    print("Resolved config:")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
