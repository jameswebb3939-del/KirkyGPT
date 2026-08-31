from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Sequence

from llm_followups.native import require_native
from llm_followups.prompting import canonical_chat_messages
from llm_followups.server.llm_runtime import (
    GenerationRequest,
    GenerationResult,
)
from llm_followups.server.schemas import ChatMessage
from llm_followups.tuning.validate import (
    fallback_followups,
    try_repair_to_followups,
    validate_followup_list,
)
from llm_followups.utils.config import Settings


def _take_bullets(
    text: str,
    k: int,
) -> str | None:
    bullets: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if (
            stripped.startswith("- ")
            or stripped.startswith("* ")
        ):
            bullets.append(stripped)

        if len(bullets) >= k:
            break

    if not bullets:
        return None

    return "\n".join(
        bullets[:k]
    ).strip()


class NativeLLMRuntime:
    """
    Native llama.cpp-backed runtime.

    Provides the same application-facing
    contract as LLMRuntime while delegating
    tokenization and inference to the C++
    extension.
    """

    def __init__(
        self,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._native = require_native()

        self._engine = None
        self._scheduler = None

        self._loaded = False
        self._model_path: Path | None = None
        self._gpu_layers = 0

    async def load(self) -> None:
        if self._loaded:
            return

        raw_path = os.getenv(
            "NATIVE_MODEL_PATH",
            "",
        ).strip()

        if not raw_path:
            raise RuntimeError(
                "NATIVE_MODEL_PATH must point "
                "to a local GGUF model"
            )

        model_path = Path(
            raw_path
        ).resolve()

        if not model_path.exists():
            raise RuntimeError(
                "Native GGUF model does not "
                f"exist: {model_path}"
            )

        if (
            model_path.suffix.lower()
            != ".gguf"
        ):
            raise RuntimeError(
                "Native inference requires "
                "a .gguf model"
            )

        config = (
            self._native.EngineConfig()
        )

        config.model_path = str(
            model_path
        )

        config.n_ctx = int(
            os.getenv(
                "NATIVE_N_CTX",
                "2048",
            )
        )

        config.n_batch = int(
            os.getenv(
                "NATIVE_N_BATCH",
                "512",
            )
        )

        config.n_threads = int(
            os.getenv(
                "NATIVE_THREADS",
                "0",
            )
        )

        config.n_threads_batch = int(
            os.getenv(
                "NATIVE_THREADS_BATCH",
                "0",
            )
        )

        self._gpu_layers = int(
            os.getenv(
                "NATIVE_GPU_LAYERS",
                "0",
            )
        )

        config.n_gpu_layers = (
            self._gpu_layers
        )

        # Model loading is blocking native
        # work, so keep it off the event loop.
        self._engine = (
            await asyncio.to_thread(
                self._native.InferenceEngine,
                config,
            )
        )

        self._scheduler = (
            self._native.BatchScheduler(
                self._engine,
                int(
                    os.getenv(
                        "NATIVE_MAX_BATCH_SIZE",
                        "4",
                    )
                ),
                int(
                    os.getenv(
                        "NATIVE_MAX_QUEUE_SIZE",
                        "64",
                    )
                ),
                int(
                    os.getenv(
                        "NATIVE_BATCH_WAIT_MS",
                        "4",
                    )
                ),
                int(
                    os.getenv(
                        "NATIVE_BATCH_PARALLELISM",
                        "1",
                    )
                ),
            )
        )

        self._model_path = model_path
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def device_str(self) -> str:
        if self._gpu_layers > 0:
            return "cuda"

        return "cpu"

    def model_name(self) -> str:
        if self._model_path is None:
            return "native-unloaded"

        return (
            "native:"
            f"{self._model_path.name}"
        )

    def adapter_loaded(self) -> bool:
        return False

    def queue_depth(
        self,
    ) -> int:
        """
        Return the number of requests
        currently waiting in the native
        C++ batch scheduler.

        Returns zero while the runtime
        scheduler has not been created.
        """
        if self._scheduler is None:
            return 0

        return int(
            self._scheduler.queue_depth()
        )

    def make_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        effective_max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else self._settings.max_new_tokens
        )

        effective_temperature = (
            temperature
            if temperature is not None
            else self._settings.temperature
        )

        effective_top_p = (
            top_p
            if top_p is not None
            else self._settings.top_p
        )

        effective_seed = (
            seed
            if seed is not None
            else self._settings.seed
        )

        if (
            not isinstance(
                effective_max_new_tokens,
                int,
            )
            or effective_max_new_tokens < 1
        ):
            raise ValueError(
                "max_new_tokens must be "
                "an integer >= 1"
            )

        if effective_max_new_tokens > 4096:
            raise ValueError(
                "max_new_tokens too large"
            )

        if (
            not isinstance(
                effective_temperature,
                (int, float),
            )
            or not (
                0.0
                <= effective_temperature
                <= 2.0
            )
        ):
            raise ValueError(
                "temperature must be a "
                "number between 0.0 and 2.0"
            )

        if (
            not isinstance(
                effective_top_p,
                (int, float),
            )
            or not (
                0.0
                < effective_top_p
                <= 1.0
            )
        ):
            raise ValueError(
                "top_p must be in "
                "range (0.0, 1.0]"
            )

        if (
            effective_seed is not None
            and not isinstance(
                effective_seed,
                int,
            )
        ):
            raise ValueError(
                "seed must be an "
                "integer or None"
            )

        return GenerationRequest(
            messages=messages,
            max_new_tokens=(
                effective_max_new_tokens
            ),
            temperature=float(
                effective_temperature
            ),
            top_p=float(
                effective_top_p
            ),
            seed=effective_seed,
        )

    def enforce_followup_format(
        self,
        text: str,
        *,
        prompt_summary: str | None = None,
    ) -> tuple[str, bool, bool]:
        if not self._settings.enforce_format:
            return (
                text.strip(),
                False,
                False,
            )

        k = self._settings.min_questions

        trimmed = _take_bullets(
            text,
            k,
        )

        if trimmed is not None:
            text = trimmed

        validation = (
            validate_followup_list(
                text,
                min_questions=k,
                bullet_style=(
                    self._settings
                    .bullet_style
                ),
                require_question_mark=True,
                forbid_extra_text=True,
            )
        )

        if validation.ok:
            return (
                validation.normalized_text
                or text.strip(),
                False,
                False,
            )

        repair_style = (
            self._settings.bullet_style
            if self._settings.bullet_style
            in (
                "dash",
                "asterisk",
            )
            else "dash"
        )

        repaired = (
            try_repair_to_followups(
                text,
                min_questions=k,
                bullet_style=(
                    repair_style
                ),
            )
        )

        if repaired is not None:
            return (
                repaired,
                True,
                False,
            )

        fallback = fallback_followups(
            prompt_summary=(
                prompt_summary
            ),
            min_questions=k,
            bullet_style=(
                repair_style
            ),
        )

        return (
            fallback,
            False,
            True,
        )

    def _build_native_request(
        self,
        request: GenerationRequest,
    ):
        native_request = (
            self._native
            .GenerationRequest()
        )

        native_messages = []

        canonical = (
            canonical_chat_messages(
                request.messages,
                min_questions=(
                    self._settings
                    .min_questions
                ),
                bullet_style=(
                    self._settings
                    .bullet_style
                ),
            )
        )

        for message in canonical:
            native_message = (
                self._native.ChatMessage()
            )

            native_message.role = (
                message["role"]
            )

            native_message.content = (
                message["content"]
            )

            native_messages.append(
                native_message
            )

        native_request.messages = (
            native_messages
        )

        generation = (
            self._native
            .GenerationConfig()
        )

        generation.max_new_tokens = (
            request.max_new_tokens
        )

        generation.temperature = (
            request.temperature
        )

        generation.top_p = (
            request.top_p
        )

        generation.seed = request.seed

        native_request.generation = (
            generation
        )

        return native_request

    async def generate(
        self,
        request: GenerationRequest,
    ) -> GenerationResult:
        if not self._loaded:
            await self.load()

        if self._scheduler is None:
            raise RuntimeError(
                "Native inference scheduler "
                "is not initialized"
            )

        native_request = (
            self._build_native_request(
                request
            )
        )

        start = time.perf_counter()

        native_result = (
            await asyncio.to_thread(
                self._scheduler.generate,
                native_request,
            )
        )

        raw_text = (
            native_result.text.strip()
        )

        prompt_summary = None

        for message in reversed(
            request.messages
        ):
            if message.role == "user":
                prompt_summary = (
                    message.content[:100]
                )
                break

        (
            final_text,
            used_repair,
            used_fallback,
        ) = self.enforce_followup_format(
            raw_text,
            prompt_summary=(
                prompt_summary
            ),
        )

        latency_ms = int(
            (
                time.perf_counter()
                - start
            )
            * 1000
        )

        return GenerationResult(
            raw_text=raw_text,
            final_text=final_text,
            used_fallback=(
                used_fallback
            ),
            used_repair=(
                used_repair
            ),
            latency_ms=latency_ms,
        )

    async def close(self) -> None:
        if self._scheduler is not None:
            await asyncio.to_thread(
                self._scheduler.close
            )

        self._scheduler = None
        self._engine = None
        self._model_path = None
        self._loaded = False