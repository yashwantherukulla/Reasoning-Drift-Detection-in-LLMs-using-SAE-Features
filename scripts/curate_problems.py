"""
scripts/curate_problems.py
==========================
CLI entry point for Phase 1 dataset curation.

Usage:
    uv run scripts/curate_problems.py

Hydra CLI overrides (no YAML editing needed):
    uv run scripts/curate_problems.py dataset.n_problems=50
    uv run scripts/curate_problems.py dataset.categories.arithmetic=10
    uv run scripts/curate_problems.py model.device=cpu
    uv run scripts/curate_problems.py --cfg job          # print resolved config and exit
    uv run scripts/curate_problems.py --multirun dataset.n_problems=50,100
"""

import random
import hydra
from omegaconf import DictConfig
from datasets import load_dataset

from src.config import register_configs
from src.dataset.problem_loader import Problem, save_problems
from typing import Any, cast

register_configs()


def generate_arithmetic(n: int) -> list[Problem]:
    problems = []
    for i in range(n):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        c = random.randint(1, 9)
        op = random.choice(["+", "-", "*"])
        if op == "+":
            ans = a + b * c
            q = f"Calculate {a} + {b} * {c}."
        elif op == "-":
            ans = a - b * c
            q = f"Calculate {a} - {b} * {c}."
        else:
            ans = a * b + c
            q = f"Calculate {a} * {b} + {c}."

        problems.append(Problem(
            id=f"arith_{i}",
            category="arithmetic",
            question=q,
            answer=str(ans)
        ))
    return problems


def generate_logical(n: int) -> list[Problem]:
    problems = []
    names = ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]
    for i in range(n):
        sel = random.sample(names, 4)
        q = (
            f"{sel[0]} is taller than {sel[1]}. "
            f"{sel[1]} is taller than {sel[2]}. "
            f"{sel[3]} is shorter than {sel[2]}. "
            f"Who is the tallest among them?"
        )
        problems.append(Problem(
            id=f"logic_{i}",
            category="logical",
            question=q,
            answer=sel[0]
        ))
    return problems


def generate_symbolic(n: int) -> list[Problem]:
    problems = []
    for i in range(n):
        word = "".join(random.choices("abcdefghijklmnopqrstuvwxyz", k=5))
        q = f"Reverse the string '{word}' and concatenate it with the original string '{word}'."
        ans = word[::-1] + word
        problems.append(Problem(
            id=f"symb_{i}",
            category="symbolic",
            question=q,
            answer=ans
        ))
    return problems


def get_gsm8k(n: int, source: str) -> list[Problem]:
    print(f"Downloading {n} problems from {source}...")
    dataset = load_dataset(source, "main", split="train")
    subset = dataset.select(range(n))
    problems = []
    for i, item in enumerate(subset):
        row = cast(dict[str, Any], item)
        ans_parts = row["answer"].split("####")
        final_answer = ans_parts[-1].strip() if len(ans_parts) > 1 else row["answer"].strip()
        problems.append(Problem(
            id=f"gsm8k_{i}",
            category="word_problems",
            question=row["question"],
            answer=final_answer
        ))
    return problems


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print("Curating reasoning problems...")
    cats = cfg.dataset.categories

    arith_probs  = generate_arithmetic(cats.arithmetic)
    logic_probs  = generate_logical(cats.logical)
    symb_probs   = generate_symbolic(cats.symbolic)
    gsm8k_probs  = get_gsm8k(cats.word_problems, cfg.dataset.gsm8k_source)

    all_problems = arith_probs + logic_probs + symb_probs + gsm8k_probs
    random.shuffle(all_problems)

    print(f"Total problems collected: {len(all_problems)}")

    out_path = cfg.dataset.problems_path
    save_problems(all_problems, out_path)
    print(f"Saved problems to {out_path}")


if __name__ == "__main__":
    main()
