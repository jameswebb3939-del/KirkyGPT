from __future__ import annotations

from llm_followups.rules.engine import (
    NO_RULE_RESPONSE,
    RuleEngine,
)
from llm_followups.server.schemas import (
    ChatMessage,
)


def user(content: str) -> ChatMessage:
    return ChatMessage(
        role="user",
        content=content,
    )


def assistant(content: str) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=content,
    )


def test_docker_starts_with_one_question() -> None:
    engine = RuleEngine()

    text = engine.respond(
        [user("Help me with Kirk")]
    )

    assert text == (
        "Are you mourning Charlie for the Kirkiversary, "
        "hunting the real killers, or both?"
    )
    assert text.count("?") == 1


def test_docker_answer_is_predefined_and_moves_forward() -> None:
    engine = RuleEngine()

    first = engine.respond(
        [user("Help me with Kirk")]
    )

    second = engine.respond(
        [
            user("Help me with Kirk"),
            assistant(first),
            user("hunt"),
        ]
    )

    assert (
        "For the real hunt, start with Erika's sudden widow glow-up"
        in second
    )
    assert (
        "Do you need help with Erika theories, "
        "shooter motives, or Kirkiversary memes?"
        in second
    )
    assert second.count("?") == 1


def test_docker_final_answer_has_no_next_question() -> None:
    engine = RuleEngine()

    first = engine.respond(
        [user("Help me with Kirk")]
    )

    second = engine.respond(
        [
            user("Help me with Kirk"),
            assistant(first),
            user("hunt"),
        ]
    )

    third = engine.respond(
        [
            user("Help me with Kirk"),
            assistant(first),
            user("hunt"),
            assistant(second),
            user("meme"),
        ]
    )

    assert "For Kirkiversary memes" in third
    assert "?" not in third


def test_rule_lengths_are_not_globally_three() -> None:
    engine = RuleEngine()

    docker = engine.match_rule("kirk")
    redis = engine.match_rule("conspiracy")
    fastapi = engine.match_rule("erika")

    assert docker is not None
    assert redis is not None
    assert fastapi is not None

    assert len(docker.steps) == 2
    assert len(redis.steps) == 1
    assert len(fastapi.steps) == 3


def test_unmatched_topic_never_uses_model_text() -> None:
    engine = RuleEngine()

    assert (
        engine.respond(
            [user("Tell me about orbital mechanics")]
        )
        == NO_RULE_RESPONSE
    )


def test_can_start_new_rule_after_completed_flow() -> None:
    engine = RuleEngine()

    redis_question = engine.respond(
        [user("Help me with conspiracy")]
    )

    redis_answer = engine.respond(
        [
            user("Help me with conspiracy"),
            assistant(redis_question),
            user("coverup"),
        ]
    )

    next_topic = engine.respond(
        [
            user("Help me with conspiracy"),
            assistant(redis_question),
            user("coverup"),
            assistant(redis_answer),
            user("Now help me with epstein"),
        ]
    )

    assert next_topic == (
        "Are you cataloging the client lists or "
        "linking them to Charlie's hit?"
    )


def test_pending_flow_can_switch_topics() -> None:
    engine = RuleEngine()

    docker_question = engine.respond(
        [user("Help me with Kirk")]
    )

    switched = engine.respond(
        [
            user("Help me with Kirk"),
            assistant(docker_question),
            user("Actually help me with conspiracy"),
        ]
    )

    assert switched == (
        "Are you mapping the cover-up, "
        "the beneficiaries, or the next target?"
    )


def test_valid_answer_wins_over_incidental_other_topic_keyword() -> None:
    engine = RuleEngine()

    docker_question = engine.respond(
        [user("Help me with Kirk")]
    )

    response = engine.respond(
        [
            user("Help me with Kirk"),
            assistant(docker_question),
            user(
                "hunt; it may connect to conspiracy later"
            ),
        ]
    )

    assert (
        "For the real hunt, start with Erika's sudden widow glow-up"
        in response
    )
    assert (
        "Do you need help with Erika theories, "
        "shooter motives, or Kirkiversary memes?"
        in response
    )