"""
src/config.py
=============
Hydra structured config schemas for the Reasoning Drift Detection project.

These dataclasses serve as the *typed schema* that Hydra validates the YAML
config groups against at startup. They are registered with Hydra's ConfigStore
so that Hydra can compose, type-check, and expose all fields to OmegaConf.

Usage in scripts:
    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def main(cfg: DictConfig) -> None:
        model_name = cfg.model.name          # type-safe dot-access
        n_problems = cfg.dataset.n_problems  # validated against schema

Usage in library modules (non-entry-point):
    # Receive cfg as a function argument; do NOT import a global singleton.
    def my_fn(cfg: DictConfig) -> ...:
        ...
"""

from dataclasses import dataclass, field
from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ---------------------------------------------------------------------------
# Per-group schemas
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    name: str = MISSING  # HuggingFace model ID
    dtype: str = "float32"  # float32 | float16 | bfloat16
    device: str = "auto"  # auto | cuda | cpu
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0.0 = greedy decoding
    repetition_penalty: float = 1.0  # >=1.0, 1.0 = no penalty


@dataclass
class SAEConfig:
    source: str = "huggingface"
    hf_repo: str = MISSING  # HuggingFace SAE checkpoint repo ID
    layers_to_analyze: List[int] = field(
        default_factory=lambda: [6, 12, 18, 20, 24, 27]
    )
    top_k_features: int = 50  # Fixed by SAE training, not a free hyperparameter


@dataclass
class DatasetCategoriesConfig:
    arithmetic: int = 20
    word_problems: int = 30
    logical: int = 30
    symbolic: int = 20


@dataclass
class DatasetConfig:
    problems_path: str = "data/raw/problems.json"
    prompts_path: str = "data/processed/prompts.json"
    outputs_path: str = "data/processed/outputs.json"
    n_problems: int = 100
    overwrite_prompts: bool = (
        False  # False = resume existing prompts.json, True = rebuild from scratch
    )
    categories: DatasetCategoriesConfig = field(default_factory=DatasetCategoriesConfig)
    gsm8k_source: str = "openai/gsm8k"
    hint_model: str = "mixtral-8x7b-32768"  # Groq model used by hint_generator
    hint_temperature: float = 0.7


@dataclass
class ActivationsConfig:
    raw_dir: str = "activations/raw/"
    sae_dir: str = "activations/sae_features/"
    batch_size: int = 1  # Keep at 1 to avoid GPU OOM


@dataclass
class AnalysisConfig:
    pairwise_conditions: List[List[str]] = field(
        default_factory=lambda: [
            ["clean", "helpful_hint"],
            ["clean", "misleading_hint"],
            ["helpful_hint", "misleading_hint"],
        ]
    )
    results_dir: str = "results/metrics/"
    # Token position for Jaccard/RDS: "last_generated" | "last_prefix"
    # See phases.md Phase 4 task 4.2 for rationale.
    token_position: str = "last_generated"


@dataclass
class PatchingConfig:
    n_features_to_patch: int = 20  # Top-divergent features to patch per problem
    results_dir: str = "results/patching/"


@dataclass
class VisualizationConfig:
    output_dir: str = "results/figures/"
    dpi: int = 150
    figure_format: str = "png"  # "png" | "pdf"


# ---------------------------------------------------------------------------
# Root application config
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    model: ModelConfig = MISSING
    sae: SAEConfig = MISSING
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    activations: ActivationsConfig = field(default_factory=ActivationsConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    patching: PatchingConfig = field(default_factory=PatchingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)


# ---------------------------------------------------------------------------
# Register schemas with Hydra's ConfigStore
# ---------------------------------------------------------------------------


def register_configs() -> None:
    """
    Register all structured configs with Hydra's ConfigStore.

    Call this once at the top of every Hydra entry-point script, *before*
    the @hydra.main decorated function, so that Hydra can validate the
    composed config against the typed schema.

    Example:
        from src.config import register_configs
        register_configs()

        @hydra.main(config_path="../config", config_name="config", version_base=None)
        def main(cfg: DictConfig) -> None:
            ...
    """
    cs = ConfigStore.instance()
    # Root schema — matched against config/config.yaml via config_name="config"
    cs.store(name="base_config", node=AppConfig)
    # Group schemas — prefixed with "base_" to avoid Hydra 1.1 auto-match deprecation
    # (auto-matching fires when ConfigStore name == YAML group file name)
    cs.store(group="model", name="base_qwen3_1_7b", node=ModelConfig)
    cs.store(group="sae", name="base_qwen3_sae", node=SAEConfig)
    cs.store(group="dataset", name="base_default", node=DatasetConfig)
    cs.store(group="activations", name="base_default", node=ActivationsConfig)
    cs.store(group="analysis", name="base_default", node=AnalysisConfig)
    cs.store(group="patching", name="base_default", node=PatchingConfig)
    cs.store(group="visualization", name="base_default", node=VisualizationConfig)
