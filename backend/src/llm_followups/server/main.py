from __future__ import annotations

import logging

from contextlib import asynccontextmanager
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..persistence.database import init_db
from ..persistence.models import ConversationModel

from .llm_runtime import LLMRuntime
from .schemas import (
    ChatRequest,
    ChatResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    CreateConversationRequest,
    ErrorResponse,
    HealthResponse,
    SendConversationMessageRequest,
    StoredMessageResponse,
)

from ..services.chat_history import (
    ChatHistoryService,
    ConversationNotFoundError,
)

from ..utils.config import (
    Settings,
    get_settings,
)


logger = logging.getLogger(__name__)


DatabaseInitializer = Callable[
    [],
    Awaitable[None],
]


def error_response(
    status_code: int,
    *,
    detail: str,
    code: str | None = None,
    errors: list[str] | None = None,
) -> JSONResponse:
    payload = ErrorResponse(
        detail=detail,
        code=code,
        errors=errors,
    ).model_dump()

    return JSONResponse(
        status_code=status_code,
        content=payload,
    )


def conversation_summary(
    model: ConversationModel,
) -> ConversationSummaryResponse:
    return ConversationSummaryResponse(
        id=model.id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def conversation_detail(
    model: ConversationModel,
) -> ConversationResponse:
    return ConversationResponse(
        id=model.id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
        messages=[
            StoredMessageResponse(
                id=message.id,
                role=message.role,
                content=message.content,
                position=message.position,
                created_at=message.created_at,
            )
            for message in model.messages
        ],
    )


def create_app(
    settings: Settings | None = None,
    *,
    runtime: LLMRuntime | None = None,
    chat_history: ChatHistoryService | None = None,
    init_database: DatabaseInitializer = init_db,
) -> FastAPI:
    if settings is None:
        settings = get_settings()

    if runtime is None:
        runtime = LLMRuntime(settings)

    if chat_history is None:
        chat_history = ChatHistoryService(
            runtime
        )

    app = FastAPI(
        title="LLM Followups Server",
        version="1.0.0",
        description=(
            "Generate and persist follow-up "
            "question conversations."
        ),
    )

    app.state.runtime = runtime
    app.state.settings = settings
    app.state.chat_history = chat_history

    @app.on_event("startup")
    async def on_startup() -> None:
        logger.info(
            "Initializing database..."
        )

        await init_database()

        logger.info(
            "Loading LLM runtime..."
        )

        try:
            await app.state.runtime.load()

            logger.info(
                "LLM runtime loaded successfully"
            )

        except Exception as exc:
            logger.error(
                "Failed to load LLM runtime: %s",
                exc,
            )
            raise

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health() -> HealthResponse:
        runtime = app.state.runtime

        return HealthResponse(
            status="ok",
            model_loaded=runtime.is_loaded(),
            model_name=runtime.model_name(),
            device=runtime.device_str(),
            adapter_loaded=runtime.adapter_loaded(),
        )

    # ----------------------------------
    # Existing stateless endpoint
    # ----------------------------------

    @app.post(
        "/chat",
        response_model=ChatResponse,
    )
    async def chat(
        req: ChatRequest,
    ) -> ChatResponse:
        runtime = app.state.runtime

        if not runtime.is_loaded():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model not loaded. "
                    "Please try again later."
                ),
                headers={
                    "Retry-After": "60"
                },
            )

        try:
            gen_req = runtime.make_request(
                req.messages,
                max_new_tokens=(
                    req.max_new_tokens
                ),
                temperature=(
                    req.temperature
                ),
                top_p=req.top_p,
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        try:
            result = await runtime.generate(
                gen_req
            )

        except RuntimeError as exc:
            logger.error(
                "Generation error: %s",
                exc,
            )

            raise HTTPException(
                status_code=500,
                detail="Generation failed",
            ) from exc

        return ChatResponse(
            response_text=result.final_text
        )

    # ----------------------------------
    # Persistent conversations
    # ----------------------------------

    @app.post(
        "/conversations",
        response_model=ConversationResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def create_conversation(
        request: CreateConversationRequest,
    ) -> ConversationResponse:
        service = app.state.chat_history

        conversation = (
            await service.create_conversation(
                title=request.title
            )
        )

        return conversation_detail(
            conversation
        )

    @app.get(
        "/conversations",
        response_model=list[
            ConversationSummaryResponse
        ],
    )
    async def list_conversations() -> list[
        ConversationSummaryResponse
    ]:
        service = app.state.chat_history

        conversations = (
            await service.list_conversations()
        )

        return [
            conversation_summary(item)
            for item in conversations
        ]

    @app.get(
        "/conversations/{conversation_id}",
        response_model=ConversationResponse,
    )
    async def get_conversation(
        conversation_id: str,
    ) -> ConversationResponse:
        service = app.state.chat_history

        try:
            conversation = (
                await service.get_conversation(
                    conversation_id
                )
            )

        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            ) from exc

        return conversation_detail(
            conversation
        )

    @app.delete(
        "/conversations/{conversation_id}",
        status_code=status.HTTP_204_NO_CONTENT,
    )
    async def delete_conversation(
        conversation_id: str,
    ) -> Response:
        service = app.state.chat_history

        try:
            await service.delete_conversation(
                conversation_id
            )

        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            ) from exc

        return Response(
            status_code=(
                status.HTTP_204_NO_CONTENT
            )
        )

    @app.post(
        "/conversations/{conversation_id}/chat",
        response_model=ConversationResponse,
    )
    async def send_conversation_message(
        conversation_id: str,
        request: SendConversationMessageRequest,
    ) -> ConversationResponse:
        service = app.state.chat_history

        if not app.state.runtime.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded",
            )

        try:
            conversation = (
                await service.send_message(
                    conversation_id,
                    request.content,
                )
            )

        except ConversationNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return conversation_detail(
            conversation
        )

    # ----------------------------------
    # Error handling
    # ----------------------------------

    @app.exception_handler(
        RequestValidationError
    )
    async def request_validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        del request

        errors = [
            str(error)
            for error in exc.errors()
        ]

        return error_response(
            400,
            detail=(
                "Request validation failed"
            ),
            code="validation_error",
            errors=errors,
        )

    @app.exception_handler(
        ValidationError
    )
    async def validation_error_handler(
        request: Request,
        exc: ValidationError,
    ) -> JSONResponse:
        del request

        errors = [
            str(error)
            for error in exc.errors()
        ]

        return error_response(
            400,
            detail=(
                "Request validation failed"
            ),
            code="validation_error",
            errors=errors,
        )

    @app.exception_handler(
        ValueError
    )
    async def value_error_handler(
        request: Request,
        exc: ValueError,
    ) -> JSONResponse:
        del request

        return error_response(
            400,
            detail=str(exc),
            code="invalid_parameter",
        )

    @app.exception_handler(
        RuntimeError
    )
    async def runtime_error_handler(
        request: Request,
        exc: RuntimeError,
    ) -> JSONResponse:
        del request

        logger.error(
            "Runtime error: %s",
            exc,
        )

        return error_response(
            500,
            detail="Internal server error",
            code="runtime_error",
        )

    @app.exception_handler(
        Exception
    )
    async def generic_exception_handler(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        del request

        logger.exception(
            "Unexpected error: %s",
            exc,
        )

        return error_response(
            500,
            detail="Internal server error",
            code="unknown_error",
        )

    return app


app = create_app()