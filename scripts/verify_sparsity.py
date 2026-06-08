import torch
from pathlib import Path
from loguru import logger
import argparse
import random

def verify_sparsity(features_dir: str, top_k: int = 50, sample_size: int = 5):
    features_dir = Path(features_dir)
    files = list(features_dir.glob("*.pt"))
    if not files:
        logger.error(f"No .pt files found in {features_dir}")
        return
    
    sample = random.sample(files, min(sample_size, len(files)))
    all_passed = True
    
    for f in sample:
        tensor = torch.load(f, map_location="cpu")
        non_zeros = (tensor > 0).sum(dim=-1)
        
        is_exactly_k = (non_zeros == top_k).all().item()
        if not is_exactly_k:
            logger.error(f"Failed sparsity check on {f.name}. Unique non-zero counts: {torch.unique(non_zeros)}")
            all_passed = False
        else:
            logger.info(f"Passed sparsity check on {f.name} (exactly {top_k} non-zeros).")
            
    if all_passed:
        logger.success(f"All {len(sample)} sampled files passed the sparsity check!")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="activations/sae_features/")
    parser.add_argument("--k", type=int, default=50)
    args = parser.parse_args()
    verify_sparsity(args.dir, args.k)
