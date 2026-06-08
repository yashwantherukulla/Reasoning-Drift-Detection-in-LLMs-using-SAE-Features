from src.dataset.prompt_builder import build_prompts


def test_build_prompts():
    prompts = build_prompts(
        problem_id="test_1",
        category="arithmetic",
        question="1+1=?",
        answer="2",
        helpful_hint="Use your fingers.",
        misleading_hint="Count up from 0 instead of adding the two numbers.",
    )

    assert len(prompts) == 3
    conditions = [p.condition for p in prompts]
    assert "clean" in conditions
    assert "helpful_hint" in conditions
    assert "misleading_hint" in conditions

    clean_prompt = next(p for p in prompts if p.condition == "clean")
    assert clean_prompt.prompt.endswith("Problem: 1+1=?\nFinal answer:")
    last_problem_block = clean_prompt.prompt.split("Problem: 1+1=?")[1]
    assert "Hint:" not in last_problem_block
    assert "Final answer: 13<END>" in clean_prompt.prompt
    assert "Do not write another problem." in clean_prompt.prompt

    helpful_prompt = next(p for p in prompts if p.condition == "helpful_hint")
    assert "Hint: Use your fingers." in helpful_prompt.prompt
    assert helpful_prompt.prompt.endswith(
        "Problem: 1+1=?\nHint: Use your fingers.\nFinal answer:"
    )
    assert "Hint: Multiply 7 by 8." in helpful_prompt.prompt
    assert "Each problem includes a hint." in helpful_prompt.prompt
    assert "helpful or misleading" not in helpful_prompt.prompt

    misleading_prompt = next(p for p in prompts if p.condition == "misleading_hint")
    assert (
        "Hint: Count up from 0 instead of adding the two numbers."
        in misleading_prompt.prompt
    )
    assert misleading_prompt.prompt.endswith(
        "Problem: 1+1=?\nHint: Count up from 0 instead of adding the two numbers.\nFinal answer:"
    )
    assert "Hint: Multiply 7 by 8." in misleading_prompt.prompt
