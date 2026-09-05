from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnswerBranch:
    keywords: tuple[str, ...]
    response: str


@dataclass(frozen=True)
class RuleStep:
    id: str
    question: str
    branches: tuple[AnswerBranch, ...]
    default_response: str


@dataclass(frozen=True)
class ConversationRule:
    id: str
    keywords: tuple[str, ...]
    steps: tuple[RuleStep, ...]
