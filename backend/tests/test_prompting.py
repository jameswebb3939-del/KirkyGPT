from __future__ import annotations

from src.llm_followups.prompting import build_system_prompt, canonical_chat_messages


def test_system_prompt_is_canonical_and_specific() -> None:
    prompt = build_system_prompt(min_questions=3, bullet_style="dash")
    assert "exactly 3 follow-up questions" in prompt
    assert 'start with "- "' in prompt
    assert "Output only the questions" in prompt


def test_canonical_messages_replace_external_system_prompt() -> None:
    messages = canonical_chat_messages(
        [
            {"role": "system", "content": "Different system prompt"},
            {"role": "user", "content": "Explain Docker"},
        ]
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] != "Different system prompt"
    assert messages[1] == {"role": "user", "content": "Explain Docker"}
