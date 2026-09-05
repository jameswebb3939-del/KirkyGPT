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
        [user("Help me with Docker")]
    )

    assert text == (
        "Are you using Docker for local "
        "development, deployment, or both?"
    )
    assert text.count("?") == 1


def test_docker_answer_is_predefined_and_moves_forward() -> None:
    engine = RuleEngine()

    first = engine.respond(
        [user("Help me with Docker")]
    )

    second = engine.respond(
        [
            user("Help me with Docker"),
            assistant(first),
            user("deployment"),
        ]
    )

    assert (
        "For deployment, prefer immutable images"
        in second
    )
    assert (
        "Do you need help with Dockerfiles, "
        "containers, or Docker Compose?"
        in second
    )
    assert second.count("?") == 1


def test_docker_final_answer_has_no_next_question() -> None:
    engine = RuleEngine()

    first = engine.respond(
        [user("Help me with Docker")]
    )

    second = engine.respond(
        [
            user("Help me with Docker"),
            assistant(first),
            user("deployment"),
        ]
    )

    third = engine.respond(
        [
            user("Help me with Docker"),
            assistant(first),
            user("deployment"),
            assistant(second),
            user("compose"),
        ]
    )

    assert "For Docker Compose" in third
    assert "?" not in third


def test_rule_lengths_are_not_globally_three() -> None:
    engine = RuleEngine()

    docker = engine.match_rule("docker")
    redis = engine.match_rule("redis")
    fastapi = engine.match_rule("fastapi")

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
        [user("Help me with Redis")]
    )

    redis_answer = engine.respond(
        [
            user("Help me with Redis"),
            assistant(redis_question),
            user("caching"),
        ]
    )

    next_topic = engine.respond(
        [
            user("Help me with Redis"),
            assistant(redis_question),
            user("caching"),
            assistant(redis_answer),
            user("Now help me with pytest"),
        ]
    )

    assert next_topic == (
        "Are you learning pytest basics or "
        "debugging a failing test?"
    )


def test_pending_flow_can_switch_topics() -> None:
    engine = RuleEngine()

    docker_question = engine.respond(
        [user("Help me with Docker")]
    )

    switched = engine.respond(
        [
            user("Help me with Docker"),
            assistant(docker_question),
            user("Actually help me with Redis"),
        ]
    )

    assert switched == (
        "Are you using Redis for caching, "
        "sessions, or coordination?"
    )


def test_valid_answer_wins_over_incidental_other_topic_keyword() -> None:
    engine = RuleEngine()

    docker_question = engine.respond(
        [user("Help me with Docker")]
    )

    response = engine.respond(
        [
            user("Help me with Docker"),
            assistant(docker_question),
            user(
                "deployment; it may connect to Redis later"
            ),
        ]
    )

    assert (
        "For deployment, prefer immutable images"
        in response
    )
    assert (
        "Do you need help with Dockerfiles, "
        "containers, or Docker Compose?"
        in response
    )
