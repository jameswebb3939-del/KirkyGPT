from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

BulletStyle = Literal["dash", "asterisk", "either"]


def preferred_bullet(style: BulletStyle = "dash") -> str:
    """Return the concrete bullet marker used for generation."""
    return "*" if style == "asterisk" else "-"


def build_system_prompt(
    *,
    min_questions: int = 3,
    bullet_style: BulletStyle = "dash",
) -> str:
    """Build the canonical KirkGPT conversation instruction.

    min_questions and bullet_style are retained temporarily
    for backwards compatibility with existing call sites.
    """
    del min_questions
    del bullet_style

    return (
        "You are KirkGPT. "
        "Continue the conversation according to the "
        "conversation patterns represented in the training data. "
        "Respond directly to the user's latest message. "
        "Preserve the conversational flow and produce the "
        "appropriate next response rather than forcing a "
        "fixed bullet-list format."
    )


def _message_role_and_content(message: Any) -> tuple[str | None, str | None]:
    if isinstance(message, Mapping):
        role = message.get("role")
        content = message.get("content")
    else:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)

    if not isinstance(role, str) or not isinstance(content, str):
        return None, None

    content = content.strip()
    if not content:
        return None, None

    return role, content


def normalise_conversation_messages(messages: Sequence[Any]) -> list[dict[str, str]]:
    """Normalize user/assistant messages and drop any external system prompt.

    The task system prompt is owned by this module so training, sanity inference,
    API inference, and evaluation cannot silently drift apart.
    """
    normalized: list[dict[str, str]] = []
    for message in messages:
        role, content = _message_role_and_content(message)
        if role not in {"user", "assistant"} or content is None:
            continue
        normalized.append({"role": role, "content": content})
    return normalized


def canonical_chat_messages(
    messages: Sequence[Any],
    *,
    min_questions: int = 3,
    bullet_style: BulletStyle = "dash",
) -> list[dict[str, str]]:
    """Prepend the canonical system instruction to a conversation."""
    return [
        {
            "role": "system",
            "content": build_system_prompt(
                min_questions=min_questions,
                bullet_style=bullet_style,
            ),
        },
        *normalise_conversation_messages(messages),
    ]


def render_chat_prompt(
    tokenizer: Any,
    messages: Sequence[Any],
    *,
    min_questions: int = 3,
    bullet_style: BulletStyle = "dash",
    add_generation_prompt: bool = True,
) -> str:
    """Render the canonical conversation using the model's chat template.

    A small text fallback is kept for tokenizers without a chat template, but
    Llama-3.2-Instruct will normally take the apply_chat_template branch.
    """
    canonical = canonical_chat_messages(
        messages,
        min_questions=min_questions,
        bullet_style=bullet_style,
    )

    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            canonical,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    parts: list[str] = []
    for message in canonical:
        role = message["role"].capitalize()
        parts.append(f"{role}:\n{message['content']}")

    if add_generation_prompt:
        parts.append("Assistant:\n")

    return "\n\n".join(parts)


def apply_chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Any],
    *,
    min_questions: int = 3,
    bullet_style: BulletStyle = "dash",
    add_generation_prompt: bool,
) -> list[int]:
    """Tokenize the canonical chat while preserving exact template boundaries."""
    canonical = canonical_chat_messages(
        messages,
        min_questions=min_questions,
        bullet_style=bullet_style,
    )

    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            canonical,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        return list(encoded)

    text = render_chat_prompt(
        tokenizer,
        messages,
        min_questions=min_questions,
        bullet_style=bullet_style,
        add_generation_prompt=add_generation_prompt,
    )
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=False,
        padding=False,
    )
    return list(encoded["input_ids"])
