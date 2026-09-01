from __future__ import annotations

import json
import logging
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..server.runtime_types import (
    GenerationResult,
)


logger = logging.getLogger(__name__)


class RedisChatGenerationCache:
    """
    Redis cache for LLM generation
    results.

    Redis failure is non-fatal.
    The underlying LLM remains the
    authoritative generation path.
    """

    def __init__(
        self,
        *,
        url: str,
        ttl_s: int = 600,
        key_prefix: str = "ec_pro",
        client: Any | None = None,
    ) -> None:
        if ttl_s < 1:
            raise ValueError(
                "Redis chat cache TTL "
                "must be >= 1"
            )

        clean_prefix = (
            key_prefix
            .strip()
            .strip(":")
        )

        if not clean_prefix:
            raise ValueError(
                "Redis key prefix "
                "cannot be empty"
            )

        self._ttl_s = ttl_s
        self._key_prefix = clean_prefix

        if client is not None:
            self._redis = client

        else:
            self._redis = Redis.from_url(
                url,
                decode_responses=True,
                socket_connect_timeout=0.5,
                socket_timeout=1.0,
            )

        self._available = True

    # ----------------------------------
    # Keys
    # ----------------------------------

    def generation_key(
        self,
        request_hash: str,
    ) -> str:
        return (
            f"{self._key_prefix}:"
            f"chat:v1:{request_hash}"
        )

    # ----------------------------------
    # Lifecycle
    # ----------------------------------

    async def ping(self) -> bool:
        try:
            result = (
                await self._redis.ping()
            )

            self._available = bool(
                result
            )

            return self._available

        except RedisError as exc:
            self._mark_unavailable(
                exc
            )

            return False

    async def close(self) -> None:
        try:
            close = getattr(
                self._redis,
                "aclose",
                None,
            )

            if close is not None:
                await close()

        except RedisError as exc:
            logger.warning(
                "Redis chat cache "
                "close failed: %s",
                exc,
            )

    # ----------------------------------
    # Read
    # ----------------------------------

    async def get_generation(
        self,
        request_hash: str,
    ) -> GenerationResult | None:
        if not self._available:
            return None

        key = self.generation_key(
            request_hash
        )

        try:
            raw = await self._redis.get(
                key
            )

        except RedisError as exc:
            self._mark_unavailable(
                exc
            )

            return None

        if raw is None:
            return None

        try:
            payload = json.loads(raw)

            if not isinstance(
                payload,
                dict,
            ):
                raise ValueError(
                    "Cached generation "
                    "must be an object"
                )

            raw_text = payload[
                "raw_text"
            ]

            final_text = payload[
                "final_text"
            ]

            used_fallback = payload[
                "used_fallback"
            ]

            used_repair = payload[
                "used_repair"
            ]

            latency_ms = payload[
                "latency_ms"
            ]

            if not isinstance(
                raw_text,
                str,
            ):
                raise ValueError(
                    "Invalid raw_text"
                )

            if not isinstance(
                final_text,
                str,
            ):
                raise ValueError(
                    "Invalid final_text"
                )

            if not isinstance(
                used_fallback,
                bool,
            ):
                raise ValueError(
                    "Invalid "
                    "used_fallback"
                )

            if not isinstance(
                used_repair,
                bool,
            ):
                raise ValueError(
                    "Invalid used_repair"
                )

            if (
                not isinstance(
                    latency_ms,
                    int,
                )
                or isinstance(
                    latency_ms,
                    bool,
                )
            ):
                raise ValueError(
                    "Invalid latency_ms"
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

        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ):
            logger.warning(
                "Invalid cached chat "
                "generation; deleting "
                "key %s",
                key,
            )

            try:
                await self._redis.delete(
                    key
                )

            except RedisError as exc:
                self._mark_unavailable(
                    exc
                )

            return None

    # ----------------------------------
    # Write
    # ----------------------------------

    async def set_generation(
        self,
        request_hash: str,
        result: GenerationResult,
    ) -> None:
        if not self._available:
            return

        key = self.generation_key(
            request_hash
        )

        payload = json.dumps(
            {
                "raw_text": (
                    result.raw_text
                ),
                "final_text": (
                    result.final_text
                ),
                "used_fallback": (
                    result.used_fallback
                ),
                "used_repair": (
                    result.used_repair
                ),
                "latency_ms": (
                    result.latency_ms
                ),
            },
            ensure_ascii=False,
        )

        try:
            await self._redis.set(
                key,
                payload,
                ex=self._ttl_s,
            )

        except RedisError as exc:
            self._mark_unavailable(
                exc
            )

    # ----------------------------------
    # Failure handling
    # ----------------------------------

    def _mark_unavailable(
        self,
        exc: Exception,
    ) -> None:
        self._available = False

        logger.warning(
            "Redis chat cache "
            "unavailable: %s",
            exc,
        )