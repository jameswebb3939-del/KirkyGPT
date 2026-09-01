from __future__ import annotations

import inspect
import logging
import time

from contextlib import (
    asynccontextmanager,
)

from typing import Any

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)

from llm_followups.observability.metrics import (
    metrics_response,
    observe_inference,
    record_http_request,
)

from llm_followups.server.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
)

from llm_followups.utils.config import (
    Settings,
    get_settings,
)


logger = logging.getLogger(
    __name__
)


async def _close_runtime(
    runtime: Any,
) -> None:
    """
    Close a runtime regardless of
    whether close() is synchronous
    or asynchronous.
    """
    close = getattr(
        runtime,
        "close",
        None,
    )

    if not callable(close):
        return

    result = close()

    if inspect.isawaitable(
        result
    ):
        await result


def create_inference_app(
    settings: Settings | None = None,
    *,
    runtime: Any | None = None,
) -> FastAPI:
    """
    Build the stateless EC Pro
    inference service.

    This service is intended to be
    horizontally replicated across
    GPU-backed Kubernetes pods.
    """
    if settings is None:
        settings = get_settings()

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):
        if (
            app.state.runtime
            is None
        ):
            from llm_followups.native.runtime import (
                NativeLLMRuntime,
            )

            app.state.runtime = (
                NativeLLMRuntime(
                    settings
                )
            )

        logger.info(
            "Loading native "
            "inference runtime..."
        )

        await (
            app.state.runtime
            .load()
        )

        logger.info(
            "Native inference "
            "runtime ready"
        )

        try:
            yield

        finally:
            logger.info(
                "Closing native "
                "inference runtime..."
            )

            await _close_runtime(
                app.state.runtime
            )

    app = FastAPI(
        title=(
            "EC Pro "
            "Inference Service"
        ),
        version="1.0.0",
        description=(
            "Horizontally scalable "
            "native/GPU inference tier "
            "for EC Pro."
        ),
        lifespan=lifespan,
    )

    app.state.runtime = runtime

    app.state.settings = (
        settings
    )

    @app.middleware(
        "http"
    )
    async def metrics_middleware(
        request: Request,
        call_next,
    ) -> Response:
        return (
            await record_http_request(
                request,
                call_next,
            )
        )

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health(
    ) -> HealthResponse:
        """
        Liveness endpoint.

        The process may remain alive
        while the model is temporarily
        unavailable.
        """
        current = (
            app.state.runtime
        )

        loaded = False
        model_name = "native-unloaded"
        device = "unknown"
        adapter_loaded = False

        if current is not None:
            try:
                loaded = bool(
                    current
                    .is_loaded()
                )
            except Exception:
                loaded = False

            try:
                model_name = str(
                    current
                    .model_name()
                )
            except Exception:
                model_name = (
                    "native-unavailable"
                )

            try:
                resolved_device = str(
                    current
                    .device_str()
                )

                if resolved_device in (
                    "cpu",
                    "cuda",
                    "unknown",
                ):
                    device = (
                        resolved_device
                    )
            except Exception:
                device = "unknown"

            try:
                adapter_loaded = bool(
                    current
                    .adapter_loaded()
                )
            except Exception:
                adapter_loaded = False

        return HealthResponse(
            status="ok",
            model_loaded=loaded,
            model_name=model_name,
            device=device,
            adapter_loaded=(
                adapter_loaded
            ),
        )

    @app.get(
        "/ready"
    )
    async def ready(
    ) -> dict[str, str]:
        """
        Readiness endpoint.

        Kubernetes only sends traffic
        to a pod once this succeeds.
        """
        current = (
            app.state.runtime
        )

        if (
            current is None
            or not current.is_loaded()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference runtime "
                    "is not ready"
                ),
            )

        return {
            "status": "ready",
        }

    @app.get(
        "/metrics"
    )
    async def metrics(
    ) -> Response:
        """
        Prometheus scrape endpoint.
        """
        return metrics_response(
            app.state.runtime
        )

    @app.post(
        "/chat",
        response_model=ChatResponse,
    )
    async def chat(
        req: ChatRequest,
    ) -> ChatResponse:
        """
        Execute one inference request.
        """
        current = (
            app.state.runtime
        )

        if (
            current is None
            or not current.is_loaded()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Inference runtime "
                    "is not ready"
                ),
                headers={
                    "Retry-After": "5",
                },
            )

        try:
            generation_request = (
                current.make_request(
                    req.messages,
                    max_new_tokens=(
                        req
                        .max_new_tokens
                    ),
                    temperature=(
                        req.temperature
                    ),
                    top_p=(
                        req.top_p
                    ),
                    seed=(
                        req.seed
                    ),
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        started = (
            time.perf_counter()
        )

        outcome = "success"

        try:
            result = (
                await current.generate(
                    generation_request
                )
            )

        except Exception as exc:
            outcome = "error"

            logger.exception(
                "Inference failed"
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Inference failed"
                ),
            ) from exc

        finally:
            observe_inference(
                outcome=outcome,
                duration_seconds=(
                    time.perf_counter()
                    - started
                ),
            )

        return ChatResponse(
            response_text=(
                result.final_text
            ),
            raw_text=(
                result.raw_text
            ),
            used_fallback=(
                result.used_fallback
            ),
            used_repair=(
                result.used_repair
            ),
            latency_ms=(
                result.latency_ms
            ),
        )

    return app


app = create_inference_app()