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
    final_text: str


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
        }

    async def generate(
        self,
        request,
    ) -> DummyResult:
        del request

        return DummyResult(
            final_text=(
                "- What is the "
                "target load?\n"
                "- Which SLO matters "
                "most?\n"
                "- How should scaling "
                "behave?"
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
                            "Explain "
                            "autoscaling"
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
            "ec_pro_native_queue_depth "
            "3.0"
            in response.text
        )

        assert (
            "ec_pro_runtime_loaded "
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