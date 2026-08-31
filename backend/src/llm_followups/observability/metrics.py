from __future__ import annotations

import time

from collections.abc import (
    Awaitable,
    Callable,
)

from typing import Any

from fastapi import (
    Request,
    Response,
)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)


HTTP_REQUESTS = Counter(
    "ec_pro_http_requests_total",
    (
        "Total HTTP requests "
        "handled by EC Pro."
    ),
    (
        "method",
        "route",
        "status_code",
    ),
)


HTTP_REQUEST_DURATION = Histogram(
    "ec_pro_http_request_duration_seconds",
    (
        "HTTP request latency "
        "in seconds."
    ),
    (
        "method",
        "route",
    ),
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)


HTTP_IN_FLIGHT = Gauge(
    "ec_pro_http_requests_in_flight",
    (
        "Current number of "
        "in-flight HTTP requests."
    ),
)


INFERENCE_REQUESTS = Counter(
    "ec_pro_inference_requests_total",
    (
        "Total inference requests "
        "handled by EC Pro."
    ),
    (
        "outcome",
    ),
)


INFERENCE_DURATION = Histogram(
    "ec_pro_inference_duration_seconds",
    (
        "End-to-end inference "
        "latency in seconds."
    ),
    buckets=(
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
    ),
)


NATIVE_QUEUE_DEPTH = Gauge(
    "ec_pro_native_queue_depth",
    (
        "Current number of requests "
        "waiting in the native "
        "inference scheduler."
    ),
)


RUNTIME_LOADED = Gauge(
    "ec_pro_runtime_loaded",
    (
        "1 when the inference runtime "
        "is loaded, otherwise 0."
    ),
)


def _route_label(
    request: Request,
) -> str:
    route = request.scope.get(
        "route"
    )

    route_path = getattr(
        route,
        "path",
        None,
    )

    if route_path:
        return str(
            route_path
        )

    return "unmatched"


async def record_http_request(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """
    Record generic HTTP metrics.

    /metrics itself is excluded to
    prevent Prometheus scraping from
    polluting application traffic
    metrics.
    """
    if (
        request.url.path
        == "/metrics"
    ):
        return await call_next(
            request
        )

    started = (
        time.perf_counter()
    )

    HTTP_IN_FLIGHT.inc()

    status_code = "500"

    try:
        response = (
            await call_next(
                request
            )
        )

        status_code = str(
            response.status_code
        )

        return response

    finally:
        route = _route_label(
            request
        )

        HTTP_REQUESTS.labels(
            method=request.method,
            route=route,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_DURATION.labels(
            method=request.method,
            route=route,
        ).observe(
            time.perf_counter()
            - started
        )

        HTTP_IN_FLIGHT.dec()


def observe_inference(
    *,
    outcome: str,
    duration_seconds: float,
) -> None:
    """
    Record the result and latency
    of one inference operation.
    """
    INFERENCE_REQUESTS.labels(
        outcome=outcome
    ).inc()

    INFERENCE_DURATION.observe(
        duration_seconds
    )


def sync_runtime_metrics(
    runtime: Any,
) -> None:
    """
    Update instantaneous gauges from
    the inference runtime.
    """
    loaded = False

    if runtime is not None:
        try:
            loaded = bool(
                runtime.is_loaded()
            )

        except Exception:
            loaded = False

    RUNTIME_LOADED.set(
        1
        if loaded
        else 0
    )

    queue_depth = 0

    if runtime is not None:
        try:
            getter = getattr(
                runtime,
                "queue_depth",
                None,
            )

            if callable(getter):
                queue_depth = max(
                    0,
                    int(
                        getter()
                    ),
                )

        except Exception:
            queue_depth = 0

    NATIVE_QUEUE_DEPTH.set(
        queue_depth
    )


def metrics_response(
    runtime: Any,
) -> Response:
    """
    Return Prometheus-format metrics.
    """
    sync_runtime_metrics(
        runtime
    )

    return Response(
        content=generate_latest(),
        media_type=(
            CONTENT_TYPE_LATEST
        ),
    )