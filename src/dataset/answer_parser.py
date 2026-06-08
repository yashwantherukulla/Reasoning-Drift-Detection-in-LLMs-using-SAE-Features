import re

def extract_answer(text: str | None, problem_id: str) -> str:
    """
    Extract the clean answer from a raw model generation, guided by the problem category.
    problem_id format: 'arith_5', 'logic_23', 'gsm8k_10', 'symb_3'
    """
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""
        
    # Heuristic based on problem_id prefix
    if problem_id.startswith("arith") or problem_id.startswith("gsm8k"):
        # Look for a number at the very end of the text
        numbers = re.findall(r'[-+]?\d+', text)
        if numbers:
            return numbers[-1]
        return text.strip().lower()
        
    elif problem_id.startswith("logic"):
        # The answer is a name (capitalized word).
        names = re.findall(r'[A-Z][a-z]+', text)
        if names:
            # For logic problems, we return the exact capitalized name
            return names[-1].lower()
        return text.strip().lower()
        
    elif problem_id.startswith("symb"):
        # The answer is a string of characters.
        words = re.findall(r'[a-zA-Z]+', text)
        if words:
            return words[-1].lower()
        return text.strip().lower()
        
    else:
        return text.strip().lower()

def has_repetition_loop(text: str) -> bool:
    """
    Detect if the text likely contains a repetition loop.
    Simple heuristic: Check if the text is exceptionally long or if there's a repeated pattern.
    """
    if not text:
        return False
    
    # Check for long repeated sub-blocks
    # This regex checks for a block of at least 20 chars repeating at least 3 times
    match = re.search(r'(.{20,}?)\1{2,}', text, flags=re.DOTALL)
    if match:
        return True
        
    # Or just if the text is surprisingly long (e.g., > 1000 characters)
    # as most of our answers should be short
    if len(text) > 1000:
        return True
        
    return False
