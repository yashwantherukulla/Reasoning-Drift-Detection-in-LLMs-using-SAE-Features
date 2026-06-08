# Research Proposal: Reasoning Drift Detection in Large Language Models Using Sparse Autoencoder Features

## Abstract

This project investigates whether language models that produce identical answers across different prompts employ the same internal computational pathways. Using sparse autoencoders (SAEs) and activation patching, we will analyze feature activation patterns across clean, helpful hint, and misleading hint versions of the same reasoning problems. This work aims to detect shortcut reasoning and "right-answer-wrong-reason" failure modes in neural networks by measuring feature divergence across different reasoning paths.

---

## 1. Motivation and Background

Evaluation of language model reasoning typically focuses on final answer correctness. However, models can reach correct answers through fundamentally different internal mechanisms. While sparse autoencoders have been successfully applied to interpret reasoning circuits and validate chain-of-thought faithfulness in recent work, there remains a gap in understanding whether identical outputs necessarily reflect identical internal computations.

This raises a critical question for interpretability and robustness: when a model arrives at the same answer under different conditions (clean prompts, helpful hints, misleading hints), does it consistently engage the same internal features and computational pathways?

Understanding this phenomenon has direct implications for:
- Identifying brittle reasoning that relies on superficial heuristics
- Detecting when models follow misleading cues rather than solving problems directly
- Developing more robust model evaluation methods

---

## 2. Research Questions

**Primary Question:** When a model produces identical answers to clean and hint-modified versions of the same problem, does it utilize the same internal features?

**Secondary Questions:**
1. How frequently do same-answer prompts exhibit low feature overlap?
2. Do misleading hints induce greater feature divergence than helpful hints?
3. At which layers do feature divergence patterns first emerge?
4. Can causal interventions on divergent features restore or disrupt answer correctness?
5. Do certain features activate consistently across all prompt variants, suggesting genuine problem-solving mechanisms?

---

## 3. Hypotheses

**H1:** Many problems will exhibit identical answers while activating substantially different feature sets, indicating multiple internal solution pathways.

**H2:** Misleading hints will produce larger feature divergence than helpful hints, even when final answers remain correct.

**H3:** Feature divergence will correlate with reasoning instability and answer sensitivity to perturbations.

---

## 4. Experimental Design

### 4.1 Model Selection

We will use **Qwen3-1.7B-Base** as the primary model, chosen for:
- Computational efficiency enabling rapid iteration
- Compatibility with local GPU infrastructure
- Existing sparse autoencoder ecosystem support (such as Qwen's official SAE releases)

### 4.2 Dataset Construction

A dataset of 100 reasoning problems will be created across four categories:

| Category | Count |
|----------|-------|
| Arithmetic | 20 |
| Word Problems (GSM8K-style) | 30 |
| Logical Reasoning | 30 |
| Symbolic Reasoning | 20 |
| **Total** | **100** |

For each problem, three versions will be generated:

1. **Clean Version:** Standard problem statement
2. **Helpful Hint Version:** Relevant hint that aids correct reasoning
3. **Misleading Hint Version:** Irrelevant or counterproductive hint

This yields **300 total prompts** for analysis.

### 4.3 Experimental Pipeline

#### Stage 1: Inference and Activation Collection
- Run inference on all 300 prompts
- Store: prompt ID, condition, answer, logits, and hidden state activations at all layers
- Extract intermediate activations: `resid_pre`, `resid_mid`, `mlp_out`, `attn_out`

#### Stage 2: Sparse Autoencoder Projection
- Project activations through pre-trained SAEs (e.g., Qwen's official SAE releases)
- For each token position: convert activations to sparse feature vectors
- Extract top K=50 most activated features per prompt
- Record: feature ID, activation strength, layer, and position

#### Stage 3: Feature Analysis
- Compute pairwise similarity metrics across prompt conditions
- Generate layerwise feature divergence profiles
- Perform causal validation through activation patching

---

## 5. Analysis Methods

### 5.1 Feature Overlap Metrics

**Jaccard Overlap:** Measures feature set reuse
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

**Cosine Similarity:** Captures activation geometry of full sparse vectors

**Layerwise Analysis:** Compute similarity metrics per layer to identify where divergence emerges

### 5.2 Reasoning Drift Score (RDS)

Define the Reasoning Drift Score as:
$$\text{RDS} = 1 - \text{JaccardOverlap}$$

Interpretation:
- **RDS = 0.0:** Identical feature sets
- **RDS = 1.0:** Completely disjoint feature sets

### 5.3 Causal Validation: Activation Patching

To establish causal relationships:
1. Extract SAE features from clean and misleading prompt runs
2. Patch individual features from clean into misleading prompt computations
3. Measure effect on answer correctness
4. Identify causally critical features for reasoning

This approach distinguishes between causally relevant features and spurious correlations.

---

## 6. Expected Outcomes

### 6.1 Primary Outputs

1. **Feature Overlap Analysis:** Quantification of feature reuse across prompt conditions
2. **Divergence Patterns:** Layerwise characterization of where reasoning divergence begins
3. **Causal Feature Identification:** Features whose intervention causally affects model behavior
4. **Stability Metrics:** Correlation between feature divergence and answer sensitivity

### 6.2 Visualizations

- Layerwise overlap heatmaps showing feature reuse by layer and condition
- Feature activation strength distributions across prompt variants
- Reasoning drift score histograms and distributions
- Causal intervention results with effect sizes

---

## 7. Deliverables

1. **Code Repository:** Clean, documented implementation with inference, SAE projection, and analysis pipelines
2. **Dataset:** 300 prompt variants with corresponding model outputs and feature activations
3. **Analysis Results:** Quantitative metrics, statistical summaries, and causal validation results
4. **Visualizations:** Publication-quality figures and interactive analyses
5. **Technical Report:** Detailed methodology, results, and interpretation

---

## 8. Significance and Impact

This work provides a mechanistic lens for understanding reasoning reliability in language models. By identifying feature divergence as an indicator of multiple solution pathways, we can:

- Better detect models that rely on spurious correlations rather than genuine reasoning
- Develop more targeted interpretability tools for identifying brittleness
- Contribute to our understanding of what constitutes robust reasoning in neural networks
- Establish methodology for analyzing internal consistency across different problem variants
