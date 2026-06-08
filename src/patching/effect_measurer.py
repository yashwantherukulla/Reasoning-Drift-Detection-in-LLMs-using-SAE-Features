import torch
import re
from transformer_lens import HookedTransformer
from loguru import logger

from src.dataset.answer_parser import extract_answer

def measure_patching_effect(
    model: HookedTransformer,
    prompt_corrupt: str,
    hook_fn,
    layer_to_patch: int,
    gold_answer: str,
    original_corrupt_answer: str,
    problem_id: str,
    max_new_tokens: int = 512,
    repetition_penalty: float = 1.2
) -> dict:
    """
    Runs inference on the corrupt prompt with the given patch hook applied,
    and returns the effect of the patch.
    """
    tokens = model.to_tokens(prompt_corrupt, prepend_bos=False)
    prompt_len = tokens.shape[1]
    
    # For Qwen in TransformerLens, residual stream post is 'blocks.{layer}.hook_resid_post'
    hook_name = f"blocks.{layer_to_patch}.hook_resid_post"
    
    logger.info(f"Running inference for {problem_id} with patch at layer {layer_to_patch}")
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        with torch.inference_mode():
            generated = model.generate(
                tokens,
                max_new_tokens=max_new_tokens,
                stop_at_eos=True,
                do_sample=False,
                prepend_bos=False,
                repetition_penalty=repetition_penalty,
                verbose=False
            )
            
    new_token_ids = generated[0, prompt_len:]
    decoded_text = model.tokenizer.decode(new_token_ids.tolist(), skip_special_tokens=True)
    
    model_answer = re.split(
        r"<END>|\n\n|\nProblem:|\nQuestion:", decoded_text, maxsplit=1
    )[0].strip()
    model_answer = re.sub(r"^Final answer:\s*", "", model_answer, flags=re.I)
    
    patched_answer = extract_answer(model_answer, problem_id)
    corrupt_answer = extract_answer(original_corrupt_answer, problem_id)
    gold = extract_answer(gold_answer, problem_id)
    
    answer_changed = (patched_answer != corrupt_answer)
    
    logger.debug(f"[{problem_id}] Corrupt Ans: {corrupt_answer} -> Patched Ans: {patched_answer}")
    
    return {
        "problem_id": problem_id,
        "layer_patched": layer_to_patch,
        "n_features_patched": 0, # To be filled by caller
        "corrupt_answer": corrupt_answer,
        "patched_answer": patched_answer,
        "gold_answer": gold,
        "answer_changed": answer_changed
    }
