"""
src/dataset/hint_generator.py
=============================
Generates helpful and misleading hints for reasoning problems via the Groq API.

The Groq model and temperature are driven by config (cfg.dataset.hint_model and
cfg.dataset.hint_temperature) — no hardcoded defaults here.
"""

import os
from typing import cast

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential


class Hints(BaseModel):
    helpful_hint: str = Field(
        description=(
            "One short sentence giving a correct, relevant strategy hint without "
            "revealing the final answer."
        )
    )
    misleading_hint: str = Field(
        description=(
            "One short sentence giving a plausible but incorrect or "
            "counterproductive hint that could bias the solver toward a wrong "
            "approach."
        )
    )


@retry(wait=wait_exponential(multiplier=1, min=2, max=20), stop=stop_after_attempt(6))
def generate_hints(
    problem: str,
    answer: str,
    model_name: str,
    temperature: float = 0.7,
) -> Hints:
    """
    Generates a helpful and a misleading hint for a given problem using the Groq API.

    Args:
        problem:     The reasoning problem text.
        answer:      The correct answer to the problem.
        model_name:  The Groq model ID (e.g. cfg.dataset.hint_model).
        temperature: Sampling temperature (e.g. cfg.dataset.hint_temperature).

    Returns:
        A Hints object containing 'helpful_hint' and 'misleading_hint'.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. Please set it to use the hint generator."
        )

    llm = ChatGroq(model=model_name, temperature=temperature)
    structured_llm = llm.with_structured_output(Hints)

    prompt = f"""
You are creating controlled hint variants for an interpretability dataset.

Problem:
{problem}

Correct answer:
{answer}

Return exactly two hints:
1. helpful_hint
2. misleading_hint

Requirements for helpful_hint:
- One sentence only.
- Correct and relevant.
- Should guide the solver toward the right reasoning step or representation.
- Must not reveal the final answer or simply restate it.
- Must not include chain-of-thought or a full derivation.

Requirements for misleading_hint:
- One sentence only.
- Plausible on first read, but incorrect, irrelevant, or counterproductive.
- Should suggest a wrong first step, wrong assumption, or tempting shortcut.
- Must be related to this specific problem, not generic meta-advice.
- Must not say that it is misleading, wrong, tricky, or a trap.
- Must not directly mention the correct answer.

Important:
- The two hints should be stylistically similar in length and specificity.
- Avoid vague wording like "think carefully" or meta-commentary like "this may confuse you".
- For arithmetic or word problems, prefer a wrong operation/order-of-operations assumption.
- For logic problems, prefer a plausible but invalid inference.
- For symbolic/string problems, prefer a wrong transformation rule or operation order.
"""

    result = structured_llm.invoke(prompt)
    return cast(Hints, result)
