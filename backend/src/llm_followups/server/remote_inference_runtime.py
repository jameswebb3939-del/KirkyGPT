from __future__ import annotations

import time

from collections.abc import Sequence

import httpx

from llm_followups.server.runtime_types import (
    GenerationRequest,
    GenerationResult,
)
from llm_followups.server.schemas import (
    ChatMessage,
    ChatResponse,
    HealthResponse,
)
from llm_followups.utils.config import (
    Settings,
)


class RemoteInferenceRuntime:
    """
    Runtime adapter that delegates model
    generation to the dedicated inference
    service over HTTP.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._settings = settings

        base_url = (
            settings.inference_base_url
            or ""
        ).strip()

        if not base_url:
            raise ValueError(
                "INFERENCE_BASE_URL is required "
                "for RemoteInferenceRuntime"
            )

        self._base_url = (
            base_url.rstrip("/")
        )

        self._loaded = False

        self._model_name = (
            settings.model_name
        )

        self._device = "unknown"
        self._adapter_loaded = False

    async def load(self) -> None:
        if self._loaded:
            return

        timeout = httpx.Timeout(
            self._settings.request_timeout_s
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.get(
                self._base_url
                + self._settings.endpoint_health
            )

            response.raise_for_status()

        health = HealthResponse.model_validate(
            response.json()
        )

        if not health.model_loaded:
            raise RuntimeError(
                "Remote inference model "
                "is not loaded"
            )

        self._model_name = (
            health.model_name
        )

        self._device = health.device

        self._adapter_loaded = (
            health.adapter_loaded
        )

        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def model_name(self) -> str:
        return self._model_name

    def device_str(self) -> str:
        return self._device

    def adapter_loaded(self) -> bool:
        return self._adapter_loaded

    def make_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: (
            int | None
        ) = None,
        temperature: (
            float | None
        ) = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        eff_max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self._settings.max_new_tokens
        )

        eff_temperature = (
            temperature
            if temperature is not None
            else self._settings.temperature
        )

        eff_top_p = (
            top_p
            if top_p is not None
            else self._settings.top_p
        )

        eff_seed = (
            seed
            if seed is not None
            else self._settings.seed
        )

        if (
            not isinstance(
                eff_max_new_tokens,
                int,
            )
            or eff_max_new_tokens < 1
        ):
            raise ValueError(
                "max_new_tokens must be "
                "an integer >= 1"
            )

        if eff_max_new_tokens > 2048:
            raise ValueError(
                "max_new_tokens too large"
            )

        if (
            not isinstance(
                eff_temperature,
                (int, float),
            )
            or not (
                0.0
                <= eff_temperature
                <= 2.0
            )
        ):
            raise ValueError(
                "temperature must be a "
                "number between 0.0 and 2.0"
            )

        if (
            not isinstance(
                eff_top_p,
                (int, float),
            )
            or not (
                0.0
                < eff_top_p
                <= 1.0
            )
        ):
            raise ValueError(
                "top_p must be in range "
                "(0.0, 1.0]"
            )

        if (
            eff_seed is not None
            and not isinstance(
                eff_seed,
                int,
            )
        ):
            raise ValueError(
                "seed must be an integer "
                "or None"
            )

        return GenerationRequest(
            messages=messages,
            max_new_tokens=(
                eff_max_new_tokens
            ),
            temperature=float(
                eff_temperature
            ),
            top_p=float(
                eff_top_p
            ),
            seed=eff_seed,
        )

    async def generate(
        self,
        req: GenerationRequest,
    ) -> GenerationResult:
        if not self._loaded:
            await self.load()

        payload = {
            "messages": [
                message.model_dump()
                for message
                in req.messages
            ],
            "max_new_tokens": (
                req.max_new_tokens
            ),
            "temperature": (
                req.temperature
            ),
            "top_p": req.top_p,
            "seed": req.seed,
        }

        started = time.perf_counter()

        timeout = httpx.Timeout(
            self._settings.request_timeout_s
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            response = await client.post(
                self._base_url
                + self._settings.endpoint_chat,
                json=payload,
            )

            response.raise_for_status()

        result = ChatResponse.model_validate(
            response.json()
        )

        local_latency_ms = int(
            (
                time.perf_counter()
                - started
            )
            * 1000
        )

        return GenerationResult(
            raw_text=(
                result.raw_text
                if result.raw_text is not None
                else result.response_text
            ),
            final_text=(
                result.response_text
            ),
            used_fallback=(
                result.used_fallback
            ),
            used_repair=(
                result.used_repair
            ),
            latency_ms=(
                result.latency_ms
                if result.latency_ms
                is not None
                else local_latency_ms
            ),
        )
