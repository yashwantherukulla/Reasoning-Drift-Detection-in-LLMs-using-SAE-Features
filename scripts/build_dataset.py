import json
import os
import time

from dotenv import load_dotenv
from tqdm import tqdm

# Explicitly load .env from project root.
# Use utf-8-sig so a BOM-prefixed .env still parses keys correctly
# (e.g. "GROQ_API_KEY" instead of "\ufeffGROQ_API_KEY").
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(root_dir, ".env"), encoding="utf-8-sig")

import hydra
from omegaconf import DictConfig

from src.config import register_configs
from src.dataset.hint_generator import generate_hints
from src.dataset.problem_loader import load_problems
from src.dataset.prompt_builder import build_prompts

register_configs()


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Load problems
    problems = load_problems(cfg.dataset.problems_path)

    # Restrict to n_problems from config
    n_problems = cfg.dataset.n_problems
    overwrite_prompts = bool(cfg.dataset.overwrite_prompts)
    problems_to_process = problems[:n_problems]

    all_prompts = []
    processed_problem_ids = set()

    # Checkpoint recovery: resume only when overwrite_prompts is False.
    if overwrite_prompts:
        print(
            f"overwrite_prompts=True — rebuilding {cfg.dataset.prompts_path} from scratch."
        )
    elif os.path.exists(cfg.dataset.prompts_path):
        try:
            with open(cfg.dataset.prompts_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    all_prompts.append(item)
                    processed_problem_ids.add(item["problem_id"])
        except json.JSONDecodeError:
            pass  # File corrupted or empty, start from scratch

    # Filter problems that haven't been processed yet
    remaining_problems = [
        p for p in problems_to_process if p.id not in processed_problem_ids
    ]

    mode = "Overwriting" if overwrite_prompts else "Resuming"
    print(
        f"{mode} dataset build: {len(processed_problem_ids)} processed problems found, "
        f"{len(remaining_problems)} remaining."
    )

    # Set up tqdm with correct initial count so numbering continues correctly
    with tqdm(
        total=len(problems_to_process),
        initial=len(processed_problem_ids),
        desc="Generating Prompts",
    ) as pbar:
        for p in remaining_problems:
            try:
                # Generate hints via LLM API
                hints = generate_hints(
                    problem=p.question,
                    answer=p.answer,
                    model_name=cfg.dataset.hint_model,
                    temperature=cfg.dataset.hint_temperature,
                )

                # Combine into prompts
                prompts = build_prompts(
                    problem_id=p.id,
                    category=p.category,
                    question=p.question,
                    answer=p.answer,
                    helpful_hint=hints.helpful_hint,
                    misleading_hint=hints.misleading_hint,
                )

                # We extend with dictionaries since we are going to json.dump them
                all_prompts.extend([pr.model_dump() for pr in prompts])

                # Checkpoint: Save the generated prompts incrementally
                os.makedirs(os.path.dirname(cfg.dataset.prompts_path), exist_ok=True)
                with open(cfg.dataset.prompts_path, "w", encoding="utf-8") as f:
                    json.dump(all_prompts, f, indent=2)

                # Proactively sleep to respect the 6000 Tokens Per Minute (TPM) limit
                # Assuming ~300 tokens per request: 60s / 15 requests = 4s sleep
                time.sleep(4.0)

                pbar.update(1)
            except Exception as e:
                print(f"\nError processing problem {p.id}: {e}")
                raise

    print(
        f"\nSuccessfully generated {len(all_prompts)} prompts and saved to {cfg.dataset.prompts_path}"
    )


if __name__ == "__main__":
    main()
