import hydra
from omegaconf import DictConfig
import pandas as pd
import torch
from pathlib import Path
from loguru import logger
import json

from src.inference.model_loader import load_model_and_tokenizer
from src.sae.sae_loader import load_saes_for_layers
from src.patching.patch_runner import get_divergent_feature_indices, build_patched_feature_vector, make_patch_hook
from src.patching.effect_measurer import measure_patching_effect

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    patching_cfg = cfg.patching
    n_features = patching_cfg.n_features_to_patch
    results_dir = Path(patching_cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Analysis report suggests layer 12
    sae_layer = 12 
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(cfg)
    
    logger.info(f"Loading SAE for layer {sae_layer}...")
    saes = load_saes_for_layers(cfg.sae.hf_repo, [sae_layer])
    sae = saes[sae_layer]
    # W_dec and b_dec are handled by sae.decode()
    
    logger.info("Loading dataset and outputs...")
    with open(cfg.dataset.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompt_map = { (p["problem_id"], p["condition"]): p for p in prompts }
    
    outputs_path = cfg.dataset.outputs_path
    if not Path(outputs_path).exists():
        # Fallback to sanity check outputs if main is empty
        outputs_path = "data/processed/outputs_10_sanity.json"
        
    with open(outputs_path, "r", encoding="utf-8") as f:
        outputs = json.load(f)
    output_map = { (o["problem_id"], o["condition"]): o for o in outputs }
    
    test_problems = list(set([o["problem_id"] for o in outputs]))
    logger.info(f"Found {len(test_problems)} problems to test.")
    
    results = []
    
    for pid in test_problems:
        cond_clean = "clean"
        cond_corrupt = "misleading_hint"
        
        if (pid, cond_clean) not in prompt_map or (pid, cond_corrupt) not in prompt_map:
            continue
            
        p_clean = prompt_map[(pid, cond_clean)]
        p_corrupt = prompt_map[(pid, cond_corrupt)]
        o_corrupt = output_map.get((pid, cond_corrupt))
        if not o_corrupt:
            continue
            
        # Load pre-computed sparse feature vectors
        feat_clean_path = Path(cfg.activations.sae_dir) / f"{pid}_{cond_clean}_{sae_layer}.pt"
        feat_corrupt_path = Path(cfg.activations.sae_dir) / f"{pid}_{cond_corrupt}_{sae_layer}.pt"
        
        if not feat_clean_path.exists() or not feat_corrupt_path.exists():
            logger.warning(f"Missing features for {pid}, skipping...")
            continue
            
        feat_clean = torch.load(feat_clean_path, map_location="cpu")
        feat_corrupt = torch.load(feat_corrupt_path, map_location="cpu")
        
        # Token position = last token of prompt (last_prefix)
        # feat shape is [seq_len, d_sae] or [1, seq_len, d_sae]
        if feat_clean.ndim == 3:
            pos_clean = feat_clean.shape[1] - 1
            pos_corrupt = feat_corrupt.shape[1] - 1
            fc = feat_clean[0, pos_clean, :]
            fcorr = feat_corrupt[0, pos_corrupt, :]
        else:
            pos_clean = feat_clean.shape[0] - 1
            pos_corrupt = feat_corrupt.shape[0] - 1
            fc = feat_clean[pos_clean, :]
            fcorr = feat_corrupt[pos_corrupt, :]
        
        div_idx = get_divergent_feature_indices(fc, fcorr, n_features)
        feat_patched = build_patched_feature_vector(fc, fcorr, div_idx)
        
        # Decode back to residual space
        feat_patched = feat_patched.to(sae.device).to(torch.float32)
        resid_patched = sae.decode(feat_patched)
        resid_patched = resid_patched.unsqueeze(0).unsqueeze(0) # [1, 1, d_model]
        
        hook_fn = make_patch_hook(resid_patched, pos_corrupt)
        
        effect = measure_patching_effect(
            model=model,
            prompt_corrupt=p_corrupt["prompt"],
            hook_fn=hook_fn,
            layer_to_patch=sae_layer,
            gold_answer=p_corrupt["answer"],
            original_corrupt_answer=o_corrupt["model_answer"],
            problem_id=pid,
            max_new_tokens=cfg.model.max_new_tokens,
            repetition_penalty=float(getattr(cfg.model, "repetition_penalty", 1.2))
        )
        effect["n_features_patched"] = n_features
        results.append(effect)
        
    df = pd.DataFrame(results)
    output_file = results_dir / "patching_results.csv"
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(results)} patching results to {output_file}")
    
if __name__ == "__main__":
    main()
