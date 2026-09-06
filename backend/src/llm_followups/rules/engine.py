from __future__ import annotations

from collections.abc import Sequence

from llm_followups.server.schemas import ChatMessage

from .definitions import DEFAULT_RULES
from .matching import contains_keyword
from .models import (
    ConversationRule,
    RuleStep,
)


NO_RULE_RESPONSE = (
    "I don't have a predefined rule for that topic yet."
)


class RuleEngine:
    """
    Deterministic conversation engine.

    The engine never calls an LLM. Flow state is reconstructed from
    persisted conversation history, so a backend restart does not lose
    the active rule step.
    """

    def __init__(
        self,
        rules: Sequence[ConversationRule] = DEFAULT_RULES,
    ) -> None:
        self._rules = tuple(rules)

        self._question_index: dict[
            str,
            tuple[ConversationRule, int],
        ] = {}

        for rule in self._rules:
            for index, step in enumerate(rule.steps):
                if step.question in self._question_index:
                    raise ValueError(
                        "Rule questions must be unique: "
                        f"{step.question}"
                    )

                self._question_index[
                    step.question
                ] = (rule, index)

    def match_rule(
        self,
        text: str,
    ) -> ConversationRule | None:
        """
        Return the most specific matching topic rule.

        More matched keywords wins. Definition order is the stable
        tie-breaker.
        """

        best_rule: ConversationRule | None = None
        best_score = 0

        for rule in self._rules:
            score = sum(
                1
                for keyword in rule.keywords
                if contains_keyword(text, keyword)
            )

            if score > best_score:
                best_rule = rule
                best_score = score

        return best_rule

    def _pending_step(
        self,
        messages: Sequence[ChatMessage],
    ) -> tuple[ConversationRule, int] | None:
        """
        Find the latest predefined question in assistant history.

        The first assistant message encountered while scanning backward
        is authoritative. If it contains no rule question, the prior
        flow is considered complete.
        """

        for message in reversed(messages):
            if message.role == "user":
                continue

            content = message.content

            for question, location in self._question_index.items():
                if question in content:
                    return location

            return None

        return None

    @staticmethod
    def _matching_branch(
        step: RuleStep,
        user_answer: str,
    ):
        for branch in step.branches:
            if any(
                contains_keyword(
                    user_answer,
                    keyword,
                )
                for keyword in branch.keywords
            ):
                return branch

        return None

    def _resolve_step_answer(
        self,
        step: RuleStep,
        user_answer: str,
    ) -> str:
        branch = self._matching_branch(
            step,
            user_answer,
        )

        if branch is not None:
            return branch.response

        return step.default_response

    def _topic_switch(
        self,
        *,
        current_rule: ConversationRule,
        current_step: RuleStep,
        user_text: str,
    ) -> ConversationRule | None:
        """
        Detect an intentional topic change while a rule question is
        pending.

        A reply that already matches one of the current step's answer
        branches is treated as an answer, even if it also contains some
        other topic keyword. Otherwise, a clear match for a different
        rule starts that rule immediately.
        """

        if (
            self._matching_branch(
                current_step,
                user_text,
            )
            is not None
        ):
            return None

        candidate = self.match_rule(
            user_text
        )

        if (
            candidate is None
            or candidate.id == current_rule.id
        ):
            return None

        return candidate

    def respond(
        self,
        messages: Sequence[ChatMessage],
    ) -> str:
        """
        Produce the next deterministic response.

        Initial topic:
            -> one predefined question

        Rule answer:
            -> predefined answer
            -> optionally one next predefined question

        Topic switch:
            -> first question for the newly matched rule

        Completed/unmatched topic:
            -> deterministic no-rule response
        """

        if not messages:
            raise ValueError(
                "At least one chat message is required"
            )

        latest = messages[-1]

        if latest.role != "user":
            raise ValueError(
                "The latest chat message must be from the user"
            )

        prior_messages = messages[:-1]

        pending = self._pending_step(
            prior_messages
        )

        if pending is not None:
            rule, step_index = pending
            step = rule.steps[step_index]

            switched_rule = self._topic_switch(
                current_rule=rule,
                current_step=step,
                user_text=latest.content,
            )

            if switched_rule is not None:
                if not switched_rule.steps:
                    return NO_RULE_RESPONSE

                return (
                    switched_rule
                    .steps[0]
                    .question
                )

            answer = self._resolve_step_answer(
                step,
                latest.content,
            )

            next_index = step_index + 1

            if next_index < len(rule.steps):
                next_question = (
                    rule.steps[next_index]
                    .question
                )

                return (
                    f"{answer}\n\n"
                    f"{next_question}"
                )

            return answer

        rule = self.match_rule(
            latest.content
        )

        if rule is None or not rule.steps:
            return NO_RULE_RESPONSE

        return rule.steps[0].question
