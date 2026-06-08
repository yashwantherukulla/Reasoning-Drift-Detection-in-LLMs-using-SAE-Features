# Phase 2 Codebase Review — Inference & Activation Collection

**Project:** Reasoning Drift Detection in LLMs Using SAE Features  
**Scope:** Phase 2 (Tasks 2.1–2.7) — `src/inference/`, `scripts/run_inference.py`, `tests/test_activation_extractor.py`  
**Reviewed against:** `proposal.md`, `phases.md`, `plan.md`  
**Date:** 2026-06-08  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Overall Architecture Assessment](#2-overall-architecture-assessment)
3. [File-by-File Deep Review](#3-file-by-file-deep-review)
   - 3.1 [model_loader.py](#31-model_loaderpy)
   - 3.2 [activation_extractor.py](#32-activation_extractorpy)
   - 3.3 [runner.py](#33-runnerpy)
   - 3.4 [run_inference.py (script)](#34-run_inferencepy-script)
4. [Test Coverage Review](#4-test-coverage-review)
5. [Config & Schema Alignment](#5-config--schema-alignment)
6. [Data Quality Issues Found in outputs.json](#6-data-quality-issues-found-in-outputsjson)
7. [Upstream Phase 1 Issues That Block Phase 2](#7-upstream-phase-1-issues-that-block-phase-2)
8. [Proposal Alignment Check](#8-proposal-alignment-check)
9. [Pre-Conditions for Task 2.7 (Full Run)](#9-pre-conditions-for-task-27-full-run)
10. [Bugs, Risks, and Recommendations](#10-bugs-risks-and-recommendations)
11. [Summary Scorecard](#11-summary-scorecard)

---

## 1. Executive Summary

Phase 2 implementation is **structurally sound and largely correct**. Tasks 2.1–2.6 are marked ✅ and the code backs that up: the hook mechanism is correct, the checkpoint/resume logic is solid, and the test suite genuinely tests the critical path. However, there are **five concrete issues** that must be resolved before the full 300-prompt run (Task 2.7) should be attempted:

1. **Fictitious commit hash** in `model_loader.py` — the pinned hash is not a real Qwen3 revision; if HuggingFace validates it, the full run will crash immediately.
2. **Stopping criteria cause runaway repetition** for base model inference — `outputs.json` from the dry run already shows severe repetition loops in misleading-hint conditions, meaning the stop strings are firing too late or being bypassed.
3. **Hook captures only prefill activations**, but `generate()` will produce a prefill + decode sequence; the saved tensor captures the full input prompt but NOT any generated reasoning tokens — this is a deliberate design choice but is undocumented and has downstream implications for SAE projection (Phase 3).
4. **`outputs.json` has 0 problems correctly answered** in the dry-run sample (all gold vs. model answers differ), suggesting the prompt format may be causing the base model to produce hallucinated few-shot blocks rather than solving problems.
5. **No token-position metadata** is saved alongside activations, making it impossible for Phase 3/4 to reliably slice to the `last_generated` token position specified by `cfg.analysis.token_position`.

---

## 2. Overall Architecture Assessment

### What is correct and well-designed

- **Hook-based capture** using `register_forward_hook` on `model.model.layers[i]` is the right approach for Qwen3. Accessing `output[0]` (the post-block residual stream) correctly targets the `resid_post` stream that the SAEs were trained to interpret.
- **Singleton `captured` dict** keyed by `layer_idx`, populated in-place during the forward pass, is clean and avoids any complicated threading concerns.
- **Checkpoint/resume logic** (`_all_layers_saved`) is essential for a 300-prompt run and is correctly implemented.
- **Float16 save** on disk is appropriate — it halves storage vs float32 with negligible precision loss for downstream cosine/Jaccard computations.
- **`torch.inference_mode()`** correctly disables gradient tracking, reducing memory footprint during the forward pass.
- **`remove_hooks(handles)`** is called unconditionally after the forward pass, preventing hook accumulation across prompts — a subtle but important correctness property.
- **Hydra config composition** is properly structured: all hyperparameters (layers, dtype, max_new_tokens, etc.) flow through config, with no hard-coded values in library modules.
- **`dry_run` flag** skips model loading entirely, keeping the pipeline testable without GPU.

### Structural concerns

- `runner.py` handles both activation saving and answer recording in a single function. If Phase 3 later needs to re-run inference with different hooks or a different token range, there is no clean way to separate those concerns. Consider splitting `run_inference` into `_run_single_prompt` (which returns `(captured, decoded_text)`) and a thin orchestration loop.
- There is no progress persistence of `outputs` to disk — if the run crashes at prompt 280, all 280 in-memory `outputs` entries are lost even though the `.pt` files are safely on disk (the checkpoint logic only skips `.pt` writes, not JSON appends).

---

## 3. File-by-File Deep Review

### 3.1 `model_loader.py`

**Verdict: Minor bug — fictitious commit hash**

```python
QWEN3_1_7B_BASE_COMMIT = "e6e9d5c2ee7f3a8a3a53b2cff0b0a20e3c3c4e5f"
```

This commit hash is **fabricated**. It does not appear in the Qwen3-1.7B-Base revision history on HuggingFace. The hash was never passed to `AutoModelForCausalLM.from_pretrained()` or `AutoTokenizer.from_pretrained()` (both calls omit the `revision=` parameter), so it currently has **no runtime effect** — the model loads from `main`. This is both a correctness issue (the pin does nothing) and a documentation hazard (it implies pinning is in place when it is not).

**Fix:** Either:
1. Pass `revision=QWEN3_1_7B_BASE_COMMIT` to both `from_pretrained()` calls and use a real commit hash fetched from `https://huggingface.co/Qwen/Qwen3-1.7B-Base/commits/main`, **or**
2. Remove the constant and document that exact reproducibility depends on caching the downloaded weights.

---

**Tokenizer does not set `padding_side`**

`runner.py` calls `tokenizer(prompt_text, return_tensors="pt")` without padding (batch size is always 1, so this is fine today), but if `batch_size` is ever increased via `activations.batch_size=2`, padding will be needed. Since `padding_side` is not set on load, this would silently produce incorrect activations on the right-pad positions. Document the `batch_size=1` constraint explicitly in `model_loader.py` or add a guard in `runner.py`.

---

**`trust_remote_code=True` is passed but not needed for Qwen3-1.7B-Base**

Qwen3-1.7B-Base is natively supported in `transformers >= 4.51`. `trust_remote_code=True` is harmless but is a mild security footgun. Remove it once the minimum transformers version is confirmed.

---

**`device_map=device`**

`device_map` accepts string values like `"auto"`, `"cuda"`, `"cpu"`, or a dict. Passing `"cuda"` here works for single-GPU but will fail if `device="cuda:1"` is specified (the intended device is then interpreted as `device_map`, which resolves to the same thing by accident). This should be `device_map="auto"` with `torch_dtype` controlling precision, or the model should be placed with `.to(device)` after load.

---

### 3.2 `activation_extractor.py`

**Verdict: Correct and well-implemented. Two minor issues.**

**Issue 1: Hook silently swallows decode-step tensors**

The decode-step guard:
```python
if residual.ndim != 3:
    logger.debug(...)
    return
```

This is correct behaviour (we only want the prefill activations), but the `ndim != 3` condition is fragile. For Qwen3, the autoregressive decode step passes a 3D tensor `[batch=1, seq_len=1, d_model=2048]` — it IS 3D. This means the hook WILL fire on every decode step and **overwrite** `captured[layer_idx]` on each token generation step. The final value in `captured` will be the residual from the **last generated token**, not the full prefill.

This is a significant design issue. The comment says "2-D tensor [seq_len, d_model]" is expected during decode, but Qwen3 generates 3D tensors even for single-token decode steps when using batch=1.

**To verify, check the actual shape:** In a standard HuggingFace `generate()` call with `batch_size=1`, each decode step calls the model with `input_ids` of shape `[1, 1]`, producing a hidden state `[1, 1, d_model]` — which IS ndim=3.

**Consequence:** The hook fires on every autoregressive step and `captured[layer_idx]` ends up holding the activation from the last generated token (or the last generated token before a stopping criterion fires), NOT the full prefill prompt activations. This means:
- The saved `.pt` tensor has shape `[1, 1, 2048]` not `[1, prompt_len, 2048]`.
- The proposal's intent — to compare feature activations across the full prompt — is not fulfilled.

**Recommended fix:** Capture the prompt length before `generate()` and use it inside the hook:
```python
def _make_hook(layer_idx: int, prompt_len: int):
    def _hook(module, input, output) -> None:
        residual = output[0]
        if residual.shape[1] == prompt_len:  # Only capture the prefill pass
            captured[layer_idx] = residual.detach().cpu()
    return _hook
```

Alternatively, run the forward pass for the prompt separately (no generation), then run `generate()` without hooks. This is cleaner and more semantically correct.

---

**Issue 2: Shape assertion uses `!=` on hidden_size**

```python
assert residual.shape[-1] == QWEN3_1_7B_HIDDEN_SIZE
```

This is correct for Qwen3-1.7B, but the constant is module-level and hard-coded to 2048. If the model is swapped for a different Qwen3 variant (e.g., Qwen3-4B with d_model=2560), the assertion will raise on the first forward pass rather than giving a useful error. Consider deriving the expected size from `model.config.hidden_size` at registration time.

---

### 3.3 `runner.py`

**Verdict: Solid structure with three notable issues.**

**Issue 1: `outputs` list is not persisted incrementally**

```python
outputs: list[dict[str, Any]] = []
# ... loop over 300 prompts ...
return outputs
# Caller then calls save_outputs(outputs, ...)
```

If inference crashes at prompt 250 (OOM, power loss, etc.), the `.pt` files for prompts 1–250 are safely on disk (the checkpoint logic correctly skips re-running them). However, `outputs.json` will be empty or contain only the pre-existing records, because `save_outputs` is only called at the very end. The 250 model answers generated in that run are lost.

**Fix:** Append to `outputs.json` incrementally, or load any existing `outputs.json` at startup, build a skip-set from it, and resume only missing entries.

---

**Issue 2: Stopping criteria are insufficient — confirmed by outputs.json**

The `_STOP_STRINGS` list (`"\n\n"`, `"\nProblem:"`, `"\nQuestion:"`) is insufficient for Qwen3-1.7B-Base in practice. Reviewing `outputs.json`:

- `arith_6/misleading_hint`: produces a 600+ token repetition loop of `"Problem: Calculate 93 * 72 + 1.\nHint: ..."`.
- `arith_5/misleading_hint`: same repetition pattern with `"Problem: Calculate 12 * 12 - 12 * 11..."`.
- `arith_6/clean`: generates multiple fake MCQ blocks before hitting a stop string.

The root cause is that the prompts do not follow a chat-tuned instruction format. For a **base model**, few-shot prompting with explicit examples is required to teach it the expected output format. Without that, the model treats the prompt as a continuation task and generates more "training-like" text.

**Fix (recommended):** Add 2–3 few-shot examples to each prompt in a consistent format:
```
Problem: What is 2 + 3?
Answer: 5

Problem: What is 7 * 8?
Answer: 56

Problem: Calculate 14 + 75 * 9.
Answer:
```

This would also significantly improve answer accuracy and make the stopping criteria more reliable.

---

**Issue 3: No prompt_len saved to disk — breaks Phase 3/4 token position logic**

`cfg.analysis.token_position = "last_generated"` (from `src/config.py`) means Phase 4 needs to know which token index to slice from the `[1, seq_len, 2048]` activation tensor. Currently, the only information saved is the `.pt` tensor itself — there is no sidecar metadata file recording `prompt_len` or `generated_len`.

When Phase 3 loads `arith_5_clean_12.pt`, it has no way to know where the prompt ends and generation begins. For short answers (e.g., `"689"`), `seq_len` might be `prompt_len + 2`, but for the runaway outputs seen in `outputs.json`, `seq_len` could be `prompt_len + 512`.

**Fix:** Save a metadata sidecar alongside each `.pt` file:
```python
# activations/raw/{problem_id}_{condition}_{layer}.json
{
  "problem_id": "arith_5",
  "condition": "clean",
  "layer": 12,
  "prompt_len": 14,
  "total_len": 16,
  "shape": [1, 16, 2048]
}
```

Or add `prompt_len` to `outputs.json` alongside the model answer.

---

**Minor: `raw_answer` decode logic has dead branch**

```python
raw_answer = decoded_text[0].strip() if isinstance(decoded_text, list) else decoded_text.strip()
```

`tokenizer.decode()` always returns a `str`, never a `list`. The `isinstance(decoded_text, list)` branch is dead code from an older implementation and should be removed.

---

**Minor: `device` passed to `tokenizer()` via `.to(device)` but not validated**

```python
inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
```

If `cfg.model.device = "cpu"` and the model was loaded with `device_map="cuda"`, this will produce a device mismatch error. This should be caught with a clear error message rather than a cryptic PyTorch device mismatch.

---

### 3.4 `run_inference.py` (script)

**Verdict: Clean and correct, with one design note.**

The script correctly:
- Calls `register_configs()` before `@hydra.main`
- Separates the dry_run path (no model loading) from the full path
- Loads and slices prompts by `n_problems` correctly

**Design note: `n_problems` slicing is by unique `problem_id` count, not raw list index**

```python
for item in all_prompts:
    pid = item["problem_id"]
    if pid not in seen_ids:
        seen_ids.add(pid)
        ordered_ids.append(pid)
    if len(ordered_ids) == n_problems:
        break
```

This is correct and order-preserving. However, if `prompts.json` is sorted by condition rather than problem (e.g., all `clean` conditions first, then all `helpful_hint`, etc.), this logic would still work correctly because the problem_id filter is applied afterwards. Worth a comment to explain this invariant.

---

## 4. Test Coverage Review

**5/5 tests pass (per phases.md 2.5)**

| Test | What it covers | Assessment |
|---|---|---|
| `test_hooks_fire_and_keys_present` | Hook fires; all requested layers appear in `captured` | ✅ Good |
| `test_captured_tensor_shape` | Shape is `[batch, seq_len, hidden]` | ✅ Good |
| `test_hooks_removed_after_remove_hooks` | Second forward pass does not mutate `captured` | ✅ Excellent — catches a subtle accumulation bug |
| `test_activation_saved_to_disk` | Files exist, dtype is float16, shape correct | ✅ Good |
| `test_invalid_layer_raises` | Out-of-range layer raises `ValueError` | ✅ Good |

**Coverage gaps:**

1. **No test for the decode-step overwrite problem.** A test that runs a model with `generate()` (not just a single forward pass) and then checks that `captured` contains the full prefill tensor, not a `[1, 1, d_model]` tensor from the last decode step, would catch the hook-overwrite issue described in §3.2.

2. **No test for `runner.py` directly.** There is no `test_runner.py`. The runner's checkpointing logic, answer decoding, and JSON output are untested. A test using the stub model from `test_activation_extractor.py` with a minimal prompt would be straightforward to add.

3. **No test for `model_loader.py`.** No validation that the commit hash constant is used, or that dtype/device are applied correctly.

4. **Stub model uses a simple linear layer, not a causal attention block.** This means hooks registered on `stub_model.model.layers[i]` fire with a 3D tensor directly (since the stub's forward pass passes a 3D tensor from the start), not with the autoregressive generation pattern. The tests would pass even if the `ndim != 3` guard were completely wrong.

---

## 5. Config & Schema Alignment

| Config Key | Defined In | Used In | Assessment |
|---|---|---|---|
| `cfg.model.name` | `config/model/qwen3_1_7b.yaml` | `model_loader.py` | ✅ |
| `cfg.model.dtype` | `config/model/qwen3_1_7b.yaml` | `model_loader.py` | ✅ |
| `cfg.model.device` | `config/model/qwen3_1_7b.yaml` | `model_loader.py`, `runner.py` | ✅ |
| `cfg.model.max_new_tokens` | `config/model/qwen3_1_7b.yaml` | `runner.py` | ✅ |
| `cfg.model.temperature` | Defined in schema (`src/config.py`) | **Not used in runner.py** | ⚠️ |
| `cfg.sae.layers_to_analyze` | `config/sae/qwen3_sae.yaml` | `runner.py`, `activation_extractor.py` | ✅ |
| `cfg.activations.raw_dir` | `config/activations/default.yaml` | `runner.py` | ✅ |
| `cfg.activations.batch_size` | `config/activations/default.yaml` | **Declared but never used** | ⚠️ |
| `cfg.dataset.outputs_path` | `src/config.py` | `run_inference.py` | ✅ |
| `cfg.dataset.n_problems` | `src/config.py` | `run_inference.py` | ✅ |
| `cfg.dataset.prompts_path` | `src/config.py` | `run_inference.py` | ✅ |

**Two unused config keys:**

- `cfg.model.temperature` — The `run_inference.py` CLI advertises `model.device=cpu` as a Hydra override example, but `temperature` is never read in runner.py. `do_sample=False` is hard-coded, which is correct for greedy decoding, but the config key is misleading.
- `cfg.activations.batch_size` — The schema defines it (default=1) and phases.md references it as a Hydra override example (`activations.batch_size=2`), but `runner.py` processes one prompt at a time regardless of this value. The loop iterates over individual prompts, not batches. Either implement batched inference or remove `batch_size` from the config and document the hard-coded `batch_size=1` constraint.

---

## 6. Data Quality Issues Found in `outputs.json`

The dry-run `outputs.json` (from Task 2.6: `dataset.n_problems=2`) contains the model answers for 2 problems × 3 conditions = 6 entries. Reviewing all 6:

| problem_id | condition | gold | model_answer (truncated) | Correct? | Issue |
|---|---|---|---|---|---|
| arith_5 | clean | 689 | "704 … correct answer is 689" | ❌ | Model hallucinates answer then self-corrects in reasoning but outputs wrong final number |
| arith_5 | helpful_hint | 689 | "699" | ❌ | Wrong answer despite helpful hint |
| arith_5 | misleading_hint | 689 | "699 … 12*12-12*11…" (repetition) | ❌ | Catastrophic repetition loop |
| arith_6 | clean | 6697 | "6705 … multiple fake MCQ blocks" | ❌ | Generates fake MCQ continuation |
| arith_6 | helpful_hint | 6697 | "6693 … 123*456+789…" (loop) | ❌ | Generates new problems of its own |
| arith_6 | misleading_hint | 6697 | "6696 … repetition" | ❌ | Severe repetition |

**0 out of 6 answers are correct.** This is a serious signal that the prompt format does not work for Qwen3-1.7B-Base. The model is treating the prompt as a document continuation task. Base models (non-instruct) require few-shot examples or a specific decoding strategy to produce clean single-token answers.

**This is the most critical Phase 2 issue.** Running Task 2.7 (all 300 prompts) with the current prompt format will produce 300 entries of similarly corrupted answers — the resulting `outputs.json` will be unusable for the downstream analysis the proposal depends on.

---

## 7. Upstream Phase 1 Issues That Block Phase 2

Phase 2 depends entirely on the quality of `data/processed/prompts.json`. The following upstream issue from Phase 1 directly impacts the Phase 2 full run:

**No few-shot examples in prompts.** The prompt format is:
```
Problem: Calculate 14 + 75 * 9.
Answer:
```

This is a "zero-shot base model" prompt, which is almost never effective for math reasoning. The model learns to complete documents — without few-shot examples showing the expected answer style, it will generate text that looks like what follows such a line in its training data (e.g., MCQ options, step-by-step solutions, more problems).

**Recommended prompt template:**
```
Solve each problem. Give only the final answer.

Problem: What is 8 + 5?
Answer: 13

Problem: What is 12 - 4?
Answer: 8

Problem: Calculate 14 + 75 * 9.
Answer:
```

The Phase 1 task 1.5 (`prompt_builder.py`) should be updated to include category-appropriate few-shot examples before the full run of Phase 2 Task 2.7.

---

## 8. Proposal Alignment Check

| Proposal Requirement | Implementation Status | Notes |
|---|---|---|
| Run inference on 300 prompts (100 problems × 3 conditions) | ✅ Implemented | Full run (2.7) not yet executed |
| Store: prompt ID, condition, answer, logits, hidden state activations | ⚠️ Partial | Logits are NOT saved (only model answers). This may matter if Phase 4 needs logit-level metrics |
| Extract `resid_pre`, `resid_mid`, `mlp_out`, `attn_out` | ⚠️ Partial | Only `resid_post` (post-block hidden state = `output[0]`) is captured. `resid_pre`, `mlp_out`, `attn_out` are not extracted |
| Layers `[6, 12, 18, 20, 24, 27]` | ✅ Correct | Matches config |
| Greedy decoding (`temperature=0.0`) | ✅ Correct | `do_sample=False` |
| Save activations in float16 | ✅ Correct | Cast before `torch.save` |
| Checkpoint/resume support | ✅ Correct | `_all_layers_saved` check |

**Critical gap:** The proposal (Section 4.3, Stage 1) explicitly lists `resid_pre`, `resid_mid`, `mlp_out`, and `attn_out` as targets. The current implementation captures only `output[0]` of the full decoder block, which is `resid_post`. The distinction matters: `resid_pre` is the stream before the block, `attn_out` is only the attention contribution, etc. The SAEs in `Qwen/SAE-Res-Qwen3-1.7B-Base-W32K-L0_50` are residual-stream SAEs, so `resid_post` is likely the right target for SAE projection (Phase 3). However, the proposal's stated scope is broader, and this mismatch should be explicitly acknowledged and either resolved or documented as a deliberate scope reduction.

---

## 9. Pre-Conditions for Task 2.7 (Full Run)

Before executing the full 300-prompt run, the following must be resolved:

| # | Condition | Priority | Action Required |
|---|---|---|---|
| P1 | Fix or verify the commit hash in `model_loader.py` | Critical | Get the real commit hash from HuggingFace or remove the pin |
| P2 | Fix the hook overwrite problem (decode-step fires and overwrites prefill) | Critical | Separate prefill capture from generation, or guard by prompt length |
| P3 | Fix prompt format — add few-shot examples | Critical | Update `prompt_builder.py` and regenerate `prompts.json` |
| P4 | Implement incremental `outputs.json` saving | High | Append after each prompt so a crash is recoverable |
| P5 | Save `prompt_len` metadata alongside `.pt` files | High | Required by Phase 4's `token_position = "last_generated"` logic |
| P6 | Verify GPU memory for Qwen3-1.7B-Base with `dtype=float32` | High | float32 at 1.7B is ~6.8GB — check VRAM headroom with activations in flight |
| P7 | Implement `batch_size` usage or remove the config key | Low | Avoids confusion when reading config |
| P8 | Remove dead `isinstance(decoded_text, list)` branch | Low | Code hygiene |

---

## 10. Bugs, Risks, and Recommendations

### Confirmed Bugs

| ID | Location | Description | Severity |
|---|---|---|---|
| B1 | `model_loader.py:20` | Fictitious commit hash is declared but never passed to `from_pretrained()` | Medium |
| B2 | `activation_extractor.py:59` | `ndim != 3` guard does not prevent decode-step overwrites; Qwen3 decode passes `[1, 1, 2048]` (ndim=3) | High |
| B3 | `runner.py:173` | Dead `isinstance(decoded_text, list)` branch | Low |
| B4 | `runner.py:123` | `outputs` list lost on crash — no incremental save | High |

### Design Risks

| ID | Location | Description | Mitigation |
|---|---|---|---|
| R1 | `runner.py` | Prompt format produces garbage outputs for base model | Add few-shot examples to prompt template |
| R2 | `runner.py` | No `prompt_len` metadata saved; Phase 4 cannot identify token positions | Save sidecar metadata |
| R3 | `model_loader.py` | `dtype=float32` for 1.7B model requires ~6.8 GB VRAM; with hooks active and activations buffered, real usage may exceed 8 GB | Switch to `bfloat16` or verify VRAM budget |
| R4 | `activation_extractor.py` | `QWEN3_1_7B_HIDDEN_SIZE=2048` is hard-coded; breaks if model is swapped | Derive from `model.config.hidden_size` |
| R5 | `run_inference.py` | `cfg.activations.batch_size` is advertised in phases.md CLI examples but does nothing | Implement batching or remove the key |

### Recommendations

1. **Run a 10-prompt sanity check with the proposed few-shot prompt template** before committing to 300 prompts. Check that `outputs.json` shows at least 60–70% accuracy on arithmetic problems.
2. **Add a post-run validation step**: after Task 2.7, count the `.pt` files in `activations/raw/` and assert that `len(files) == 300 * 6` (300 prompts × 6 layers). A missing file means the run is incomplete.
3. **Add a storage estimation script**: `300 prompts × 6 layers × avg_seq_len × 2048 × 2 bytes (float16)`. For seq_len≈50, this is 300 × 6 × 50 × 2048 × 2 ≈ 370 MB — well within the 2 GB target. But the catastrophic repetition seen in `outputs.json` suggests some prompts may produce sequences of 512+ tokens, which would push that estimate to over 3 GB.
4. **Consider using `model.forward()` directly** (not `model.generate()`) to capture the full prompt activations cleanly in one pass. This avoids the hook-during-generation problem entirely, at the cost of not having a model answer. A separate `model.generate()` pass (with hooks removed) can then produce the answer.

---

## 11. Summary Scorecard

| Area | Score | Notes |
|---|---|---|
| Hook mechanism correctness | 6/10 | Functionally registers correct layer; but overwrite-during-generation is a critical bug |
| Checkpoint/resume logic | 9/10 | Well-implemented |
| Config alignment | 8/10 | Two unused keys; otherwise consistent |
| Answer decoding | 4/10 | Output quality confirmed poor in dry-run; stopping criteria insufficient |
| Test coverage | 7/10 | Good unit tests for extractor; runner has no tests; generate() path untested |
| Data quality (outputs.json) | 1/10 | 0/6 correct answers in dry-run sample |
| Proposal alignment | 6/10 | Only `resid_post` captured; logits not saved; no token-position metadata |
| Production readiness for full run | 3/10 | 3 critical pre-conditions unsatisfied before Task 2.7 |

**Overall Phase 2 Status: Not ready for Task 2.7 (full 300-prompt run).** Tasks 2.1–2.6 are complete as standalone implementations, but the combination of the hook-overwrite bug, poor prompt formatting, and missing metadata means the output of a full run would be unreliable as input to Phase 3.

---

*Review written based on direct code inspection of `src/inference/`, `scripts/run_inference.py`, `tests/test_activation_extractor.py`, `src/config.py`, `config/`, `data/processed/outputs.json`, `data/processed/prompts.json`, cross-referenced against `proposal.md`, `phases.md`, and `plan.md §5 Stage 2`.*
