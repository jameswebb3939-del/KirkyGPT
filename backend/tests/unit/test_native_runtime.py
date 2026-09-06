from __future__ import annotations

import pytest

from llm_followups.native import native_available
from llm_followups.native.runtime import NativeLLMRuntime
from llm_followups.server.schemas import ChatMessage
from llm_followups.utils.config import get_settings


pytestmark = pytest.mark.skipif(
    not native_available(),
    reason="KirkGPT native extension is not built",
)


def make_runtime() -> NativeLLMRuntime:
    return NativeLLMRuntime(
        get_settings()
    )


def test_native_runtime_initial_state() -> None:
    runtime = make_runtime()

    assert runtime.is_loaded() is False
    assert runtime.model_name() == "native-unloaded"
    assert runtime.device_str() == "cpu"
    assert runtime.adapter_loaded() is False


def test_native_runtime_make_request() -> None:
    runtime = make_runtime()

    request = runtime.make_request(
        [
            ChatMessage(
                role="user",
                content="Help me with Kirk",
            )
        ]
    )

    assert request.max_new_tokens == 64
    assert request.temperature == pytest.approx(
        0.2,
        abs=1e-6,
    )
    assert request.top_p == pytest.approx(
        0.9,
        abs=1e-6,
    )
    assert request.seed is None


def test_native_runtime_request_overrides() -> None:
    runtime = make_runtime()

    request = runtime.make_request(
        [
            ChatMessage(
                role="user",
                content="Help me with conspiracy",
            )
        ],
        max_new_tokens=128,
        temperature=0.5,
        top_p=0.8,
        seed=42,
    )

    assert request.max_new_tokens == 128

    assert request.temperature == pytest.approx(
        0.5,
        abs=1e-6,
    )

    assert request.top_p == pytest.approx(
        0.8,
        abs=1e-6,
    )

    assert request.seed == 42


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"max_new_tokens": 0},
            "max_new_tokens",
        ),
        (
            {"max_new_tokens": 4097},
            "max_new_tokens",
        ),
        (
            {"temperature": -0.1},
            "temperature",
        ),
        (
            {"temperature": 2.1},
            "temperature",
        ),
        (
            {"top_p": 0.0},
            "top_p",
        ),
        (
            {"top_p": 1.1},
            "top_p",
        ),
    ],
)
def test_native_runtime_rejects_invalid_generation_settings(
    kwargs: dict,
    message: str,
) -> None:
    runtime = make_runtime()

    with pytest.raises(
        ValueError,
        match=message,
    ):
        runtime.make_request(
            [
                ChatMessage(
                    role="user",
                    content="Test",
                )
            ],
            **kwargs,
        )


@pytest.mark.asyncio
async def test_native_runtime_load_requires_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(
        "NATIVE_MODEL_PATH",
        raising=False,
    )

    runtime = make_runtime()

    with pytest.raises(
        RuntimeError,
        match="NATIVE_MODEL_PATH",
    ):
        await runtime.load()

    assert runtime.is_loaded() is False


@pytest.mark.asyncio
async def test_native_runtime_rejects_missing_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    missing = (
        tmp_path
        / "missing.gguf"
    )

    monkeypatch.setenv(
        "NATIVE_MODEL_PATH",
        str(missing),
    )

    runtime = make_runtime()

    with pytest.raises(
        RuntimeError,
        match="does not exist",
    ):
        await runtime.load()

    assert runtime.is_loaded() is False


@pytest.mark.asyncio
async def test_native_runtime_rejects_non_gguf(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    model = (
        tmp_path
        / "model.bin"
    )

    model.write_bytes(b"test")

    monkeypatch.setenv(
        "NATIVE_MODEL_PATH",
        str(model),
    )

    runtime = make_runtime()

    with pytest.raises(
        RuntimeError,
        match=r"\.gguf",
    ):
        await runtime.load()

    assert runtime.is_loaded() is False