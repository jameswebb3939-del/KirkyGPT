from scripts.make_sft import (
    DEFAULT_RULES,
    generate_dataset,
)


def test_sft_uses_rule_definitions():
    examples = generate_dataset(
        n=25,
        seed=42,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )

    valid_rule_ids = {
        rule.id
        for rule in DEFAULT_RULES
    }

    assert valid_rule_ids

    assert all(
        example["source"]
        == "definitions.py"
        for example in examples
    )

    assert all(
        example["rule_id"]
        in valid_rule_ids
        for example in examples
    )


def test_sft_contains_rule_questions():
    examples = generate_dataset(
        n=50,
        seed=42,
        max_new_tokens=256,
        temperature=0.2,
        top_p=0.9,
    )

    questions = {
        step.question
        for rule in DEFAULT_RULES
        for step in rule.steps
    }

    generated_text = "\n".join(
        message["content"]
        for example in examples
        for message
        in example["messages"]
        if message["role"]
        == "assistant"
    )

    assert any(
        question
        in generated_text
        for question in questions
    )
