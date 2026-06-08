# Reasoning Drift Detection in LLMs using SAE Features

This repository implements a framework to evaluate how helpful versus misleading hints alter the internal reasoning pathways of Large Language Models (LLMs). Rather than just looking at the final token accuracy, this pipeline utilizes **Sparse Autoencoder (SAE)** features to track the divergence of internal representations across network depth.

By comparing the activation states of the model when prompted with a `clean` (no hint), `helpful_hint`, and `misleading_hint` prompt, we calculate the **Reasoning Divergence Score (RDS)** to determine where and how reasoning drift occurs.

---

## 🚀 Setup & Installation

This project uses `uv` for lightning-fast dependency management.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/Reasoning-Drift-Detection-in-LLMs-using-SAE-Features.git
   cd Reasoning-Drift-Detection-in-LLMs-using-SAE-Features
   ```

2. **Install dependencies:**
   ```bash
   uv sync
   ```

3. **Configure Environment:**
   You will need API keys for Groq (used for hint generation) and HuggingFace (to access gated models/SAEs). Add these to a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key
   HF_TOKEN=your_huggingface_token
   ```

---

## 🧠 Pipeline Overview

The framework consists of 6 sequential phases, configured centrally via `Hydra` in `config/config.yaml`:

1. **Dataset Construction (`scripts/build_dataset.py`)**
   Generates synthetic logic/math problems and uses Mixtral-8x7B (via Groq) to generate semantically relevant helpful and misleading hints.
2. **Inference & Metric Logging (`scripts/run_inference.py`)**
   Runs the base LLM (e.g., `Qwen/Qwen2.5-1.5B`) on all prompts and saves raw hidden states at target layers.
3. **SAE Projection (`scripts/run_sae_projection.py`)**
   Projects the dense hidden states into a sparse feature space using pre-trained SAEs, drastically reducing memory overhead while isolating interpretable features.
4. **Analysis & Metrics (`scripts/run_analysis.py` & `scripts/run_extended_analysis.py`)**
   Computes RDS (1 - Jaccard overlap), L1 distances, feature entropy, Pathway Consistency Index (PCI), and identifies divergent features across prompt conditions.
5. **Causal Patching (`scripts/run_patching.py`)**
   Performs activation patching interventions at peak divergence layers to determine if substituting SAE features can causally force the model's answer to change.
6. **Visualization (`scripts/run_visualization.py`)**
   Generates publication-ready figures (heatmaps, distributions, and summary dashboards) from the computed metrics.

---

## 🏃 How to Reproduce Results

We provide a single unified orchestrator script that automatically runs all 6 stages sequentially.

### Option 1: Python Orchestrator (Windows / Linux / Mac)
You can run the python orchestrator directly using `uv`:

To run a fast **10-problem pilot run** (Sanity check):
```bash
uv run scripts/run_all.py --pilot
```

To run the **full 100-problem evaluation**:
```bash
uv run scripts/run_all.py --full
```

### Option 2: Bash Wrapper (Linux / Mac / Git Bash)
If you prefer standard shell scripts:
```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh --pilot
./scripts/run_all.sh --full
```

*Note: All generated datasets, metrics, and visualization plots will be output into the `data/` and `results/` directories.*

---

## 📊 Key Findings from Pilot Analysis

Based on an initial 10-problem pilot run on the `Qwen3-1.7B-Base` model:

1. **Semantic Bypass (Convergent Wrong Reasoning):** Both helpful and misleading hints successfully misdirected the model's feature-level representations. The model tended to extract structural/positional "hint" features rather than semantic content, often yielding identical internal divergence regardless of whether the hint was actually helpful or misleading.
2. **L1 Distance Decoupling:** While structural overlap (RDS / Jaccard similarity) stabilized at deeper layers, the raw activation magnitude (L1 distance) exploded exponentially. The divergence between models scaling is largely driven by activation scale recalibration, not just the identity of the features activated.
3. **Pathway Fragility:** The baseline models exhibit significant "pathway fragility" on simple arithmetic problems. Misleading hints disrupted even the simplest rule-following loops, injecting noise that overpowered structural problem constraints.
4. **Drift Direction Alignment:** The `Drift Direction Cosine` metric revealed that helpful and misleading hints often push the representations in the *same* vector direction relative to the clean baseline, confirming that the network represents "hint presence" more strongly than "hint correctness."

---

## 📓 Interactive Exploration

Two interactive Jupyter notebooks utilizing Plotly are provided to explore the granular results:
- `notebooks/03_feature_analysis.ipynb`: Slice and dice metric trends, layer profiles, and RDS heatmaps.
- `notebooks/04_causal_validation.ipynb`: Explore the raw causal patching outputs.

Run them via standard Jupyter commands:
```bash
uv run jupyter lab
```
