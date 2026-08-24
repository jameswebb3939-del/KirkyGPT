from __future__ import annotations

from llm_followups.tuning.dataset import _tokenize_chat_example


class FakeTokenizer:
    chat_template = "fake"

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        # Deterministic fake tokenization where each role/content element becomes tokens.
        ids = [1]
        for message in messages:
            ids.extend([10 if message["role"] == "system" else 20 if message["role"] == "user" else 30])
            ids.extend(range(100, 100 + len(message["content"].split())))
        if add_generation_prompt:
            ids.append(30)
        return ids if tokenize else "rendered"


def test_chat_sft_masks_prompt_and_supervises_assistant_tokens() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Help with Docker"},
            {
                "role": "assistant",
                "content": "- Question one?\n- Question two?\n- Question three?",
            },
        ]
    }

    encoded = _tokenize_chat_example(
        row,
        tokenizer=FakeTokenizer(),
        max_length=512,
        min_questions=3,
        bullet_style="dash",
        assistant_only_loss=True,
    )

    assert encoded["_valid_sft"] is True
    labels = encoded["labels"]
    assert -100 in labels
    first_supervised = next(i for i, label in enumerate(labels) if label != -100)
    assert all(label == -100 for label in labels[:first_supervised])
    assert labels[first_supervised:] == encoded["input_ids"][first_supervised:]


def test_chat_sft_rejects_example_without_assistant_answer() -> None:
    encoded = _tokenize_chat_example(
        {"messages": [{"role": "user", "content": "Docker"}]},
        tokenizer=FakeTokenizer(),
        max_length=512,
        min_questions=3,
        bullet_style="dash",
        assistant_only_loss=True,
    )
    assert encoded["_valid_sft"] is False
