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
            {"role": "user", "content": "Help me with Kirk"},
            {
                "role": "assistant",
                "content": "- Are you mourning Charlie for the Kirkiversary?\n- Do you want the Erika timeline?\n- Should we open with the roof-shot?",
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
        {"messages": [{"role": "user", "content": "Kirk"}]},
        tokenizer=FakeTokenizer(),
        max_length=512,
        min_questions=3,
        bullet_style="dash",
        assistant_only_loss=True,
    )
    assert encoded["_valid_sft"] is False

def test_chat_sft_supervises_all_assistant_turns() -> None:
    row = {
        "messages": [
            {
                "role": "user",
                "content": "Start topic",
            },
            {
                "role": "assistant",
                "content": "First assistant response",
            },
            {
                "role": "user",
                "content": "Continue topic",
            },
            {
                "role": "assistant",
                "content": "Second assistant response",
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

    supervised_regions = []
    in_region = False
    start = None

    for index, label in enumerate(labels):
        if (
            label != -100
            and not in_region
        ):
            start = index
            in_region = True

        elif (
            label == -100
            and in_region
        ):
            supervised_regions.append(
                (start, index)
            )
            start = None
            in_region = False

    if in_region:
        supervised_regions.append(
            (start, len(labels))
        )

    assert len(supervised_regions) == 2

    first_start, first_end = (
        supervised_regions[0]
    )

    second_start, second_end = (
        supervised_regions[1]
    )

    assert first_end > first_start
    assert second_end > second_start

    assert (
        second_start
        > first_end
    )

    assert all(
        label == -100
        for label
        in labels[
            first_end:second_start
        ]
    )
