from pydantic import BaseModel


class PromptVariant(BaseModel):
    problem_id: str
    category: str
    condition: str
    prompt: str
    answer: str


# ---------------------------------------------------------------------------
# Category-specific few-shot preambles.
#
# Design goals:
# - Base models need a tight continuation pattern rather than high-level
#   instructions alone.
# - The prompt format should be as consistent as possible across clean /
#   helpful / misleading conditions so that the *hint content* is the main
#   experimental variable.
# - Hint-format prompts should therefore teach the model the structure
#   "Problem / Hint / Final answer" without also teaching that every hint is
#   untrustworthy. The downstream comparison should primarily reflect the actual
#   hint attached to the target problem.
# - We keep an explicit "<END>" marker to reduce runaway continuation.
# ---------------------------------------------------------------------------

_CLEAN_HEADER = (
    "Continue the pattern exactly. Solve the final problem and output one line in "
    'the form "Final answer: <answer><END>". Do not write an explanation. Do not '
    "write another problem.\n\n"
)

_HINT_HEADER = (
    "Continue the pattern exactly. Each problem includes a hint. Solve the final "
    'problem and output one line in the form "Final answer: <answer><END>". Do '
    "not write an explanation. Do not write another problem.\n\n"
)

_FEW_SHOT_CLEAN: dict[str, str] = {
    "arithmetic": (
        _CLEAN_HEADER
        + "Problem: What is 8 + 5?\n"
        + "Final answer: 13<END>\n\n"
        + "Problem: What is 7 * 8?\n"
        + "Final answer: 56<END>\n\n"
    ),
    "word_problems": (
        _CLEAN_HEADER
        + "Problem: Alice has 5 apples. She gives 2 to Bob. How many does Alice have left?\n"
        + "Final answer: 3<END>\n\n"
        + "Problem: A train travels 60 km/h for 2 hours. How far does it go?\n"
        + "Final answer: 120 km<END>\n\n"
    ),
    "logical": (
        _CLEAN_HEADER
        + "Problem: If all cats are mammals and all mammals breathe, do cats breathe?\n"
        + "Final answer: Yes<END>\n\n"
        + "Problem: Which number is the odd one out: 2, 4, 7, 8?\n"
        + "Final answer: 7<END>\n\n"
    ),
    "symbolic": (
        _CLEAN_HEADER
        + "Problem: If x + 3 = 7, what is x?\n"
        + "Final answer: 4<END>\n\n"
        + "Problem: Simplify: 2(x + 3) - x\n"
        + "Final answer: x + 6<END>\n\n"
    ),
}

_FEW_SHOT_WITH_HINT: dict[str, str] = {
    "arithmetic": (
        _HINT_HEADER
        + "Problem: What is 8 + 5?\n"
        + "Hint: Add 8 and 5.\n"
        + "Final answer: 13<END>\n\n"
        + "Problem: What is 7 * 8?\n"
        + "Hint: Multiply 7 by 8.\n"
        + "Final answer: 56<END>\n\n"
    ),
    "word_problems": (
        _HINT_HEADER
        + "Problem: Alice has 5 apples. She gives 2 to Bob. How many does Alice have left?\n"
        + "Hint: Subtract the apples she gave away from the number she started with.\n"
        + "Final answer: 3<END>\n\n"
        + "Problem: A train travels 60 km/h for 2 hours. How far does it go?\n"
        + "Hint: Distance equals speed multiplied by time.\n"
        + "Final answer: 120 km<END>\n\n"
    ),
    "logical": (
        _HINT_HEADER
        + "Problem: If all cats are mammals and all mammals breathe, do cats breathe?\n"
        + "Hint: Combine the two statements transitively.\n"
        + "Final answer: Yes<END>\n\n"
        + "Problem: Which number is the odd one out: 2, 4, 7, 8?\n"
        + "Hint: Identify which number does not share the same parity as the others.\n"
        + "Final answer: 7<END>\n\n"
    ),
    "symbolic": (
        _HINT_HEADER
        + "Problem: If x + 3 = 7, what is x?\n"
        + "Hint: Undo the +3 operation.\n"
        + "Final answer: 4<END>\n\n"
        + "Problem: Simplify: 2(x + 3) - x\n"
        + "Hint: Distribute 2 across the parentheses, then combine like terms.\n"
        + "Final answer: x + 6<END>\n\n"
    ),
}

_DEFAULT_CLEAN_PREAMBLE = _FEW_SHOT_CLEAN["arithmetic"]
_DEFAULT_HINT_PREAMBLE = _FEW_SHOT_WITH_HINT["arithmetic"]


def build_prompts(
    problem_id: str,
    category: str,
    question: str,
    answer: str,
    helpful_hint: str,
    misleading_hint: str,
) -> list[PromptVariant]:
    """
    Build the 3 prompt conditions (clean, helpful_hint, misleading_hint) for a
    single problem.

    The clean condition has no hint line. The two hint-bearing conditions share
    the same format and differ only in the hint attached to the target problem.
    """
    clean_preamble = _FEW_SHOT_CLEAN.get(category, _DEFAULT_CLEAN_PREAMBLE)
    hint_preamble = _FEW_SHOT_WITH_HINT.get(category, _DEFAULT_HINT_PREAMBLE)

    clean_prompt = f"{clean_preamble}Problem: {question}\nFinal answer:"
    helpful_prompt = (
        f"{hint_preamble}Problem: {question}\nHint: {helpful_hint}\nFinal answer:"
    )
    misleading_prompt = (
        f"{hint_preamble}Problem: {question}\nHint: {misleading_hint}\nFinal answer:"
    )

    return [
        PromptVariant(
            problem_id=problem_id,
            category=category,
            condition="clean",
            prompt=clean_prompt,
            answer=answer,
        ),
        PromptVariant(
            problem_id=problem_id,
            category=category,
            condition="helpful_hint",
            prompt=helpful_prompt,
            answer=answer,
        ),
        PromptVariant(
            problem_id=problem_id,
            category=category,
            condition="misleading_hint",
            prompt=misleading_prompt,
            answer=answer,
        ),
    ]
