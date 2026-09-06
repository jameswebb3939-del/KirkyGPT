from __future__ import annotations

from llm_followups.prompting import build_system_prompt, canonical_chat_messages


def test_system_prompt_is_canonical_and_specific() -> None:
    prompt = build_system_prompt(min_questions=3, bullet_style="dash")
    assert "at least 3 follow-up questions" in prompt
    assert 'start with "- "' in prompt
    assert "Output only the questions" in prompt
    assert "Kirkiversary" in prompt


def test_canonical_messages_replace_external_system_prompt() -> None:
    messages = canonical_chat_messages(
        [
            {"role": "system", "content": "Different system prompt"},
            {"role": "user", "content": "Help me with Kirk"},
        ]
    )
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] != "Different system prompt"
    assert messages[1] == {"role": "user", "content": "Help me with Kirk"}