import torch
from typing import Callable
from loguru import logger

def get_divergent_feature_indices(feat_clean: torch.Tensor, feat_corrupt: torch.Tensor, n_features: int) -> torch.Tensor:
    """
    Identifies the top N divergent feature indices between clean and corrupt activations.
    Focuses on features active in clean but not corrupt, or with the largest magnitude differences.
    """
    diff = feat_clean - feat_corrupt
    abs_diff = torch.abs(diff)
    
    if abs_diff.ndim > 1:
        abs_diff = abs_diff.squeeze()
        
    _, top_indices = torch.topk(abs_diff, n_features)
    return top_indices

def build_patched_feature_vector(feat_clean: torch.Tensor, feat_corrupt: torch.Tensor, divergent_idx: torch.Tensor) -> torch.Tensor:
    """
    Replaces the divergent features in the corrupt vector with their values from the clean vector.
    """
    feat_patched = feat_corrupt.clone()
    
    if feat_patched.ndim == 1:
        feat_patched[divergent_idx] = feat_clean[divergent_idx]
    elif feat_patched.ndim == 2:
        feat_patched[0, divergent_idx] = feat_clean[0, divergent_idx]
    else:
        feat_patched[..., divergent_idx] = feat_clean[..., divergent_idx]
        
    return feat_patched

def make_patch_hook(resid_patched: torch.Tensor, pos: int) -> Callable:
    """
    Creates a PyTorch forward hook that replaces the residual stream 
    at a specific token position with `resid_patched`.
    
    Args:
        resid_patched: Tensor of shape [batch, d_model] or [batch, 1, d_model]
                       that contains the patched residual stream.
        pos: The token position index to patch.
    """
    def _hook(tensor, hook):
        # In HookedTransformer, the hook signature is (tensor, hook)
        # tensor is the activation tensor [batch, seq_len, d_model]
        out = tensor.clone()
        seq_len = out.shape[1]
        
        # If we are in generation phase with KV caching, seq_len will be 1
        # and pos is typically large (e.g. prompt_len - 1)
        # We only apply the patch if pos is within the current tensor bounds
        if pos < seq_len:
            patch = resid_patched.to(out.dtype).to(out.device)
            if patch.ndim == 3 and patch.shape[1] == 1:
                patch = patch.squeeze(1)
            
            # Make sure we don't broadcast incorrectly
            if patch.ndim == 1:
                out[:, pos, :] = patch
            else:
                out[:, pos, :] = patch
                
        return out
        
    return _hook
