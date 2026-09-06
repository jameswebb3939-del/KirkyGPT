from __future__ import annotations

from dataclasses import (
    dataclass,
)

from fastapi.testclient import (
    TestClient,
)

from llm_followups.server.inference_app import (
    create_inference_app,
)

from llm_followups.utils.config import (
    Settings,
)


@dataclass
class DummyResult:
    def __init__(
        self,
        final_text: str,
    ) -> None:
        self.raw_text = final_text
        self.final_text = final_text
        self.used_fallback = False
        self.used_repair = False
        self.latency_ms = 1


class DummyRuntime:
    def __init__(
        self,
    ) -> None:
        self.loaded = False
        self.closed = False

    async def load(
        self,
    ) -> None:
        self.loaded = True

    async def close(
        self,
    ) -> None:
        self.closed = True

    def is_loaded(
        self,
    ) -> bool:
        return self.loaded

    def queue_depth(
        self,
    ) -> int:
        return 3

    def make_request(
        self,
        messages,
        max_new_tokens=None,
        temperature=None,
        top_p=None,
        seed=None,
    ):
        return {
            "messages": messages,
            "max_new_tokens": (
                max_new_tokens
            ),
            "temperature": (
                temperature
            ),
            "top_p": top_p,
            "seed": seed,
        }

    async def generate(
        self,
        request,
    ) -> DummyResult:
        del request

        return DummyResult(
            final_text=(
                "- Are you mourning Charlie for the Kirkiversary?\n"
                "- Do you want the Erika timeline or the beneficiary list?\n"
                "- Should we open with the roof-shot or the cover-up?"
            )
        )


def make_client(
) -> tuple[
    DummyRuntime,
    TestClient,
]:
    runtime = DummyRuntime()

    settings = Settings(
        model_name="test-model",
        device="cpu",
    )

    app = (
        create_inference_app(
            settings,
            runtime=runtime,
        )
    )

    return (
        runtime,
        TestClient(app),
    )


def test_ready_endpoint(
) -> None:
    runtime, client = (
        make_client()
    )

    with client:
        response = client.get(
            "/ready"
        )

        assert (
            response.status_code
            == 200
        )

        assert (
            response.json()
            == {
                "status": "ready",
            }
        )

        assert (
            runtime.loaded
            is True
        )


def test_chat_endpoint(
) -> None:
    _, client = (
        make_client()
    )

    with client:
        response = client.post(
            "/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Help me "
                            "with Kirk"
                        ),
                    }
                ]
            },
        )

        assert (
            response.status_code
            == 200
        )

        body = (
            response.json()
        )

        assert (
            "response_text"
            in body
        )

        assert (
            body[
                "response_text"
            ].startswith(
                "- "
            )
        )


def test_metrics_expose_queue_depth(
) -> None:
    _, client = (
        make_client()
    )

    with client:
        response = client.get(
            "/metrics"
        )

        assert (
            response.status_code
            == 200
        )

        assert (
            "kirk_gpt_native_queue_depth "
            "3.0"
            in response.text
        )

        assert (
            "kirk_gpt_runtime_loaded "
            "1.0"
            in response.text
        )


def test_runtime_closes_on_shutdown(
) -> None:
    runtime, client = (
        make_client()
    )

    with client:
        assert (
            runtime.closed
            is False
        )

    assert (
        runtime.closed
        is True
    )