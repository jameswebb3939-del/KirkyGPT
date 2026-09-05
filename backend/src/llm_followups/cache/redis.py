from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

from ..persistence.models import (
    ConversationModel,
    MessageModel,
)


logger = logging.getLogger(__name__)


class RedisConversationCache:
    """
    Redis cache-aside implementation for
    persisted conversation data.

    Redis failure never affects correctness.
    SQLite remains the source of truth.
    """

    def __init__(
        self,
        *,
        url: str,
        ttl_s: int = 300,
        key_prefix: str = "kirk_gpt",
        client: Any | None = None,
    ) -> None:
        if ttl_s < 1:
            raise ValueError(
                "Redis cache TTL must be >= 1"
            )

        clean_prefix = (
            key_prefix.strip().strip(":")
        )

        if not clean_prefix:
            raise ValueError(
                "Redis key prefix cannot be empty"
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

    @property
    def conversations_key(self) -> str:
        return (
            f"{self._key_prefix}:"
            "conversations:list"
        )

    def conversation_key(
        self,
        conversation_id: str,
    ) -> str:
        return (
            f"{self._key_prefix}:"
            f"conversation:{conversation_id}"
        )

    # ----------------------------------
    # Lifecycle
    # ----------------------------------

    async def ping(self) -> bool:
        try:
            result = await self._redis.ping()

            self._available = bool(result)

            return self._available

        except RedisError as exc:
            self._mark_unavailable(exc)

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
                "Redis close failed: %s",
                exc,
            )

    # ----------------------------------
    # Conversation detail
    # ----------------------------------

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationModel | None:
        if not self._available:
            return None

        key = self.conversation_key(
            conversation_id
        )

        try:
            raw = await self._redis.get(
                key
            )

        except RedisError as exc:
            self._mark_unavailable(exc)

            return None

        if raw is None:
            return None

        try:
            data = json.loads(raw)

            if not isinstance(data, dict):
                raise ValueError(
                    "Cached conversation "
                    "must be an object"
                )

            return self._deserialize_conversation(
                data,
                include_messages=True,
            )

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            logger.warning(
                "Ignoring invalid Redis "
                "conversation cache entry "
                "%s: %s",
                key,
                exc,
            )

            await self._safe_delete(key)

            return None

    async def set_conversation(
        self,
        conversation: ConversationModel,
    ) -> None:
        if not self._available:
            return

        key = self.conversation_key(
            conversation.id
        )

        payload = json.dumps(
            self._serialize_conversation(
                conversation,
                include_messages=True,
            ),
            ensure_ascii=False,
        )

        try:
            await self._redis.set(
                key,
                payload,
                ex=self._ttl_s,
            )

        except RedisError as exc:
            self._mark_unavailable(exc)

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        if not self._available:
            return

        await self._safe_delete(
            self.conversation_key(
                conversation_id
            )
        )

    # ----------------------------------
    # Conversation list
    # ----------------------------------

    async def get_conversations(
        self,
    ) -> list[ConversationModel] | None:
        if not self._available:
            return None

        try:
            raw = await self._redis.get(
                self.conversations_key
            )

        except RedisError as exc:
            self._mark_unavailable(exc)

            return None

        if raw is None:
            return None

        try:
            data = json.loads(raw)

            if not isinstance(data, list):
                raise ValueError(
                    "Cached conversation list "
                    "must be an array"
                )

            conversations: list[
                ConversationModel
            ] = []

            for item in data:
                if not isinstance(
                    item,
                    dict,
                ):
                    raise ValueError(
                        "Conversation list entry "
                        "must be an object"
                    )

                conversations.append(
                    self._deserialize_conversation(
                        item,
                        include_messages=False,
                    )
                )

            return conversations

        except (
            json.JSONDecodeError,
            TypeError,
            ValueError,
            KeyError,
        ) as exc:
            logger.warning(
                "Ignoring invalid Redis "
                "conversation-list entry: %s",
                exc,
            )

            await self._safe_delete(
                self.conversations_key
            )

            return None

    async def set_conversations(
        self,
        conversations: list[
            ConversationModel
        ],
    ) -> None:
        if not self._available:
            return

        payload = json.dumps(
            [
                self._serialize_conversation(
                    conversation,
                    include_messages=False,
                )
                for conversation
                in conversations
            ],
            ensure_ascii=False,
        )

        try:
            await self._redis.set(
                self.conversations_key,
                payload,
                ex=self._ttl_s,
            )

        except RedisError as exc:
            self._mark_unavailable(exc)

    async def invalidate_conversations(
        self,
    ) -> None:
        if not self._available:
            return

        await self._safe_delete(
            self.conversations_key
        )

    # ----------------------------------
    # Serialization
    # ----------------------------------

    @staticmethod
    def _serialize_conversation(
        conversation: ConversationModel,
        *,
        include_messages: bool,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": conversation.id,
            "title": conversation.title,
            "created_at": (
                conversation.created_at.isoformat()
            ),
            "updated_at": (
                conversation.updated_at.isoformat()
            ),
        }

        if include_messages:
            data["messages"] = [
                {
                    "id": message.id,
                    "conversation_id": (
                        message.conversation_id
                    ),
                    "role": message.role,
                    "content": message.content,
                    "position": (
                        message.position
                    ),
                    "created_at": (
                        message.created_at
                        .isoformat()
                    ),
                }
                for message
                in conversation.messages
            ]

        return data

    @staticmethod
    def _deserialize_conversation(
        data: dict[str, Any],
        *,
        include_messages: bool,
    ) -> ConversationModel:
        conversation_id = str(
            data["id"]
        )

        conversation = ConversationModel(
            id=conversation_id,
            title=str(data["title"]),
            created_at=datetime.fromisoformat(
                str(data["created_at"])
            ),
            updated_at=datetime.fromisoformat(
                str(data["updated_at"])
            ),
        )

        if include_messages:
            raw_messages = data.get(
                "messages",
                [],
            )

            if not isinstance(
                raw_messages,
                list,
            ):
                raise ValueError(
                    "messages must be a list"
                )

            messages: list[
                MessageModel
            ] = []

            for item in raw_messages:
                if not isinstance(
                    item,
                    dict,
                ):
                    raise ValueError(
                        "Cached message "
                        "must be an object"
                    )

                message = MessageModel(
                    id=str(item["id"]),
                    conversation_id=(
                        conversation_id
                    ),
                    role=str(item["role"]),
                    content=str(
                        item["content"]
                    ),
                    position=int(
                        item["position"]
                    ),
                    created_at=(
                        datetime.fromisoformat(
                            str(
                                item[
                                    "created_at"
                                ]
                            )
                        )
                    ),
                )

                messages.append(message)

            messages.sort(
                key=lambda item: (
                    item.position
                )
            )

            conversation.messages = (
                messages
            )

        return conversation

    # ----------------------------------
    # Failure handling
    # ----------------------------------

    async def _safe_delete(
        self,
        key: str,
    ) -> None:
        try:
            await self._redis.delete(key)

        except RedisError as exc:
            self._mark_unavailable(exc)

    def _mark_unavailable(
        self,
        exc: Exception,
    ) -> None:
        self._available = False

        logger.warning(
            "Redis unavailable; "
            "continuing without cache: %s",
            exc,
        )