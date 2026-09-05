from __future__ import annotations

import pytest

from llm_followups.native import (
    native_available,
    require_native,
)


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="KirkGPT native C++ extension is not built",
)


def test_native_extension_is_available() -> None:
    assert native_available()


def test_native_version() -> None:
    native = require_native()

    assert native.version() == "0.1.0"


def test_engine_config_binding() -> None:
    native = require_native()

    config = native.EngineConfig()

    config.model_path = "test.gguf"
    config.n_ctx = 2048
    config.n_batch = 512
    config.n_threads = 4
    config.n_threads_batch = 2
    config.n_gpu_layers = 0

    assert config.model_path == "test.gguf"
    assert config.n_ctx == 2048
    assert config.n_batch == 512
    assert config.n_threads == 4
    assert config.n_threads_batch == 2
    assert config.n_gpu_layers == 0


def test_generation_config_binding() -> None:
    native = require_native()

    config = native.GenerationConfig()

    config.max_new_tokens = 64
    config.temperature = 0.2
    config.top_p = 0.9
    config.seed = 42

    assert config.max_new_tokens == 64
    assert config.temperature == pytest.approx(
        0.2,
        abs=1e-6,
    )
    assert config.top_p == pytest.approx(
        0.9,
        abs=1e-6,
    )
    assert config.seed == 42


def test_generation_config_optional_seed() -> None:
    native = require_native()

    config = native.GenerationConfig()

    config.seed = None

    assert config.seed is None


def test_chat_message_binding() -> None:
    native = require_native()

    message = native.ChatMessage()

    message.role = "user"
    message.content = "Explain Docker"

    assert message.role == "user"
    assert message.content == "Explain Docker"


def test_generation_request_binding() -> None:
    native = require_native()

    message = native.ChatMessage()
    message.role = "user"
    message.content = "Explain Docker"

    generation = native.GenerationConfig()
    generation.max_new_tokens = 64
    generation.temperature = 0.2
    generation.top_p = 0.9
    generation.seed = 42

    request = native.GenerationRequest()

    request.messages = [message]
    request.generation = generation

    assert len(request.messages) == 1

    assert request.messages[0].role == "user"

    assert (
        request.messages[0].content
        == "Explain Docker"
    )

    assert (
        request.generation.max_new_tokens
        == 64
    )

    assert (
        request.generation.temperature
        == pytest.approx(
            0.2,
            abs=1e-6,
        )
    )

    assert (
        request.generation.top_p
        == pytest.approx(
            0.9,
            abs=1e-6,
        )
    )

    assert request.generation.seed == 42


def test_multiple_chat_messages_binding() -> None:
    native = require_native()

    system = native.ChatMessage()
    system.role = "system"
    system.content = "Return follow-up questions."

    user = native.ChatMessage()
    user.role = "user"
    user.content = "Explain Redis"

    request = native.GenerationRequest()

    request.messages = [
        system,
        user,
    ]

    assert len(request.messages) == 2

    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"

    assert (
        request.messages[1].content
        == "Explain Redis"
    )