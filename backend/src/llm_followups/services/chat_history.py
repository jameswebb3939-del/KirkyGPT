from __future__ import annotations

import asyncio

from llm_followups.persistence.models import (
    ConversationModel,
    MessageModel,
    utc_now,
)
from llm_followups.persistence.unit_of_work import (
    UnitOfWork,
)
from llm_followups.server.llm_runtime import (
    LLMRuntime,
)
from llm_followups.server.schemas import (
    ChatMessage,
)


class ConversationNotFoundError(
    LookupError
):
    pass


def create_title(content: str) -> str:
    cleaned = " ".join(
        content.strip().split()
    )

    if not cleaned:
        return "New chat"

    if len(cleaned) <= 35:
        return cleaned

    return f"{cleaned[:35]}..."


class ChatHistoryService:
    def __init__(
        self,
        runtime: LLMRuntime,
    ) -> None:
        self._runtime = runtime

        # Prevent two generations modifying the
        # same conversation simultaneously.
        self._locks: dict[
            str,
            asyncio.Lock,
        ] = {}

    def _lock_for(
        self,
        conversation_id: str,
    ) -> asyncio.Lock:
        lock = self._locks.get(
            conversation_id
        )

        if lock is None:
            lock = asyncio.Lock()
            self._locks[
                conversation_id
            ] = lock

        return lock

    async def create_conversation(
        self,
        *,
        title: str = "New chat",
    ) -> ConversationModel:
        conversation = ConversationModel(
            title=title.strip()
            or "New chat"
        )

        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )

            await uow.conversations.add(
                conversation
            )

            await uow.commit()

        return conversation

    async def list_conversations(
        self,
    ) -> list[ConversationModel]:
        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )

            return (
                await uow.conversations.list_all()
            )

    async def get_conversation(
        self,
        conversation_id: str,
    ) -> ConversationModel:
        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )

            conversation = (
                await uow.conversations.get(
                    conversation_id,
                    include_messages=True,
                )
            )

            if conversation is None:
                raise ConversationNotFoundError(
                    conversation_id
                )

            return conversation

    async def delete_conversation(
        self,
        conversation_id: str,
    ) -> None:
        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )

            deleted = (
                await uow.conversations.delete(
                    conversation_id
                )
            )

            if not deleted:
                raise ConversationNotFoundError(
                    conversation_id
                )

            await uow.commit()

        self._locks.pop(
            conversation_id,
            None,
        )

    async def send_message(
        self,
        conversation_id: str,
        content: str,
    ) -> ConversationModel:
        content = content.strip()

        if not content:
            raise ValueError(
                "Message content cannot be empty"
            )

        lock = self._lock_for(
            conversation_id
        )

        async with lock:
            return await self._send_message_locked(
                conversation_id,
                content,
            )

    async def _send_message_locked(
        self,
        conversation_id: str,
        content: str,
    ) -> ConversationModel:
        # ---------------------------------
        # Phase 1: read history
        # ---------------------------------

        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )

            conversation = (
                await uow.conversations.get(
                    conversation_id,
                    include_messages=True,
                )
            )

            if conversation is None:
                raise ConversationNotFoundError(
                    conversation_id
                )

            history = [
                ChatMessage(
                    role=message.role,
                    content=message.content,
                )
                for message
                in conversation.messages
            ]

        # ---------------------------------
        # Phase 2: model generation
        #
        # No DB transaction is held here.
        # ---------------------------------

        user_message = ChatMessage(
            role="user",
            content=content,
        )

        generation_request = (
            self._runtime.make_request(
                [
                    *history,
                    user_message,
                ]
            )
        )

        generation_result = (
            await self._runtime.generate(
                generation_request
            )
        )

        assistant_content = (
            generation_result.final_text
        )

        # ---------------------------------
        # Phase 3: atomic write
        #
        # User + assistant + conversation
        # metadata are committed together.
        # ---------------------------------

        async with UnitOfWork() as uow:
            assert (
                uow.conversations
                is not None
            )
            assert (
                uow.messages
                is not None
            )

            conversation = (
                await uow.conversations.get(
                    conversation_id
                )
            )

            if conversation is None:
                raise ConversationNotFoundError(
                    conversation_id
                )

            next_position = (
                await uow.messages.next_position(
                    conversation_id
                )
            )

            stored_user = MessageModel(
                conversation_id=conversation_id,
                role="user",
                content=content,
                position=next_position,
            )

            stored_assistant = MessageModel(
                conversation_id=conversation_id,
                role="assistant",
                content=assistant_content,
                position=next_position + 1,
            )

            await uow.messages.add(
                stored_user
            )

            await uow.messages.add(
                stored_assistant
            )

            if (
                conversation.title
                == "New chat"
            ):
                conversation.title = (
                    create_title(content)
                )

            conversation.updated_at = (
                utc_now()
            )

            # BOTH messages become durable
            # together here.
            await uow.commit()

        return await self.get_conversation(
            conversation_id
        )