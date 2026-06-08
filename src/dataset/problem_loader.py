import json
from pydantic import BaseModel, ValidationError

class Problem(BaseModel):
    id: str
    category: str
    question: str
    answer: str

def load_problems(filepath: str) -> list[Problem]:
    """
    Loads and validates problems from a JSON file.
    
    Args:
        filepath: Path to the problems.json file.
        
    Returns:
        A list of validated Problem objects.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    problems = []
    for p in data:
        try:
            problems.append(Problem(**p))
        except ValidationError as e:
            print(f"Validation error for problem {p.get('id', 'UNKNOWN')}: {e}")
            raise
    
    return problems

def save_problems(problems: list[Problem], filepath: str):
    """
    Saves a list of problems to a JSON file.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump([p.model_dump() for p in problems], f, indent=2)
