from __future__ import annotations

import logging

from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import (
    JSONResponse,
)
from pydantic import ValidationError

from ..cache.chat_null import (
    NullChatGenerationCache,
)
from ..cache.chat_protocol import (
    ChatGenerationCache,
)
from ..cache.factory import (
    build_chat_generation_cache,
    build_conversation_cache,
)
from ..cache.null import (
    NullConversationCache,
)
from ..cache.protocol import (
    ConversationCache,
)

from ..persistence.database import (
    init_db,
)
from ..persistence.models import (
    ConversationModel,
)

from .llm_runtime import (
    LLMRuntime,
)
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

from ..services.cached_chat_history import (
    CachedChatHistoryService,
)
from ..services.cached_llm_runtime import (
    CachedLLMRuntime,
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
    """
    Build a consistent JSON error response.
    """
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
    """
    Convert a persistence model into a
    conversation-summary API response.
    """
    return ConversationSummaryResponse(
        id=model.id,
        title=model.title,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


def conversation_detail(
    model: ConversationModel,
) -> ConversationResponse:
    """
    Convert a persistence model into a
    complete conversation API response.
    """
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
    chat_history: (
        ChatHistoryService
        | CachedChatHistoryService
        | None
    ) = None,
    cache: ConversationCache | None = None,
    generation_cache: (
        ChatGenerationCache | None
    ) = None,
    init_database: (
        DatabaseInitializer
    ) = init_db,
) -> FastAPI:
    """
    Create and configure the FastAPI
    application.

    Production/default behaviour:

    - SQLite is the source of truth for
      conversation persistence.

    - Redis caches persisted conversation
      data.

    - Redis also caches identical LLM
      generation requests.

    - LLMRuntime remains responsible for
      actual model generation.

    Tests may inject:

    - runtime
    - chat history service
    - conversation cache
    - generation cache
    - database initializer
    """

    if settings is None:
        settings = get_settings()

    # ==================================
    # Base LLM runtime
    # ==================================

    runtime_injected = (
        runtime is not None
    )

    if runtime is None:
        base_runtime = LLMRuntime(
            settings
        )
    else:
        base_runtime = runtime

    # ==================================
    # Redis generation cache
    # ==================================

    manage_generation_cache = False

    if generation_cache is None:
        if runtime_injected:
            # Tests that inject their own
            # runtime should not suddenly
            # require Redis or have their
            # fake runtime wrapped.
            generation_cache = (
                NullChatGenerationCache()
            )

            app_runtime = (
                base_runtime
            )

        else:
            generation_cache = (
                build_chat_generation_cache(
                    settings
                )
            )

            manage_generation_cache = (
                True
            )

            app_runtime = (
                CachedLLMRuntime(
                    base_runtime,
                    generation_cache,
                    settings,
                )
            )

    else:
        # A generation cache was
        # explicitly injected, so wrap
        # the runtime but do not assume
        # ownership of the cache.
        app_runtime = CachedLLMRuntime(
            base_runtime,
            generation_cache,
            settings,
        )

    # ==================================
    # Conversation Redis cache
    # ==================================

    manage_cache = False

    if chat_history is None:
        if cache is None:
            cache = (
                build_conversation_cache(
                    settings
                )
            )

            manage_cache = True

        # Important:
        # use app_runtime here rather
        # than base_runtime.
        #
        # This means persistent browser
        # conversations also use the
        # generation cache.
        base_chat_history = (
            ChatHistoryService(
                app_runtime
            )
        )

        chat_history = (
            CachedChatHistoryService(
                base_chat_history,
                cache,
            )
        )

    elif cache is None:
        # Dependency-injected services
        # should not require Redis.
        cache = (
            NullConversationCache()
        )

    # ==================================
    # Application lifespan
    # ==================================

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        """
        Application lifecycle.

        Startup:

        1. Initialise SQLite.
        2. Check conversation Redis cache.
        3. Check generation Redis cache.
        4. Load the LLM runtime.

        Shutdown:

        1. Close generation cache.
        2. Close conversation cache.

        Redis failures are non-fatal.
        SQLite and the LLM remain the
        authoritative fallbacks.
        """

        try:
            # --------------------------
            # SQLite
            # --------------------------

            logger.info(
                "Initializing database..."
            )

            await init_database()

            # --------------------------
            # Conversation Redis cache
            # --------------------------

            if (
                manage_cache
                and settings.redis_enabled
            ):
                logger.info(
                    "Connecting to Redis "
                    "conversation cache..."
                )

                redis_available = (
                    await (
                        app.state.cache
                        .ping()
                    )
                )

                if redis_available:
                    logger.info(
                        "Redis conversation "
                        "cache ready"
                    )

                else:
                    logger.warning(
                        "Redis conversation "
                        "cache unavailable; "
                        "continuing with "
                        "SQLite only"
                    )

            elif not settings.redis_enabled:
                logger.info(
                    "Redis caching disabled"
                )

            # --------------------------
            # Generation Redis cache
            # --------------------------

            if (
                manage_generation_cache
                and settings.redis_enabled
                and (
                    settings
                    .redis_chat_cache_enabled
                )
            ):
                logger.info(
                    "Connecting to Redis "
                    "chat generation cache..."
                )

                chat_cache_available = (
                    await (
                        app.state
                        .generation_cache
                        .ping()
                    )
                )

                if chat_cache_available:
                    logger.info(
                        "Redis chat "
                        "generation cache "
                        "ready"
                    )

                else:
                    logger.warning(
                        "Redis chat "
                        "generation cache "
                        "unavailable; "
                        "continuing with "
                        "direct LLM "
                        "generation"
                    )

            elif (
                settings.redis_enabled
                and not (
                    settings
                    .redis_chat_cache_enabled
                )
            ):
                logger.info(
                    "Redis chat generation "
                    "cache disabled"
                )

            # --------------------------
            # LLM
            # --------------------------

            logger.info(
                "Loading LLM runtime..."
            )

            try:
                await (
                    app.state.runtime
                    .load()
                )

                logger.info(
                    "LLM runtime loaded "
                    "successfully"
                )

            except Exception as exc:
                logger.error(
                    "Failed to load LLM "
                    "runtime: %s",
                    exc,
                )

                raise

            # Application ready.
            yield

        finally:
            # --------------------------
            # Generation cache shutdown
            # --------------------------

            if manage_generation_cache:
                logger.info(
                    "Closing Redis chat "
                    "generation cache..."
                )

                await (
                    app.state
                    .generation_cache
                    .close()
                )

            # --------------------------
            # Conversation cache shutdown
            # --------------------------

            if manage_cache:
                logger.info(
                    "Closing Redis "
                    "conversation cache..."
                )

                await (
                    app.state.cache
                    .close()
                )

            logger.info(
                "Application shutting "
                "down..."
            )

    # ==================================
    # FastAPI application
    # ==================================

    app = FastAPI(
        title="LLM Followups Server",
        version="1.0.0",
        description=(
            "Generate and persist "
            "follow-up question "
            "conversations."
        ),
        lifespan=lifespan,
    )

    # ==================================
    # Application state
    # ==================================

    app.state.runtime = (
        app_runtime
    )

    app.state.base_runtime = (
        base_runtime
    )

    app.state.settings = (
        settings
    )

    app.state.chat_history = (
        chat_history
    )

    app.state.cache = (
        cache
    )

    app.state.generation_cache = (
        generation_cache
    )

    # ==================================
    # Health
    # ==================================

    @app.get(
        "/health",
        response_model=HealthResponse,
    )
    async def health() -> HealthResponse:
        runtime = (
            app.state.runtime
        )

        return HealthResponse(
            status="ok",
            model_loaded=(
                runtime.is_loaded()
            ),
            model_name=(
                runtime.model_name()
            ),
            device=(
                runtime.device_str()
            ),
            adapter_loaded=(
                runtime.adapter_loaded()
            ),
        )

    # ==================================
    # Stateless chat
    # ==================================

    @app.post(
        "/chat",
        response_model=ChatResponse,
    )
    async def chat(
        req: ChatRequest,
    ) -> ChatResponse:
        """
        Stateless chat endpoint.

        In production this goes through
        CachedLLMRuntime, so identical
        generation requests can be served
        from Redis.
        """

        runtime = (
            app.state.runtime
        )

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
            gen_req = (
                runtime.make_request(
                    req.messages,
                    max_new_tokens=(
                        req.max_new_tokens
                    ),
                    temperature=(
                        req.temperature
                    ),
                    top_p=(
                        req.top_p
                    ),
                )
            )

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        try:
            result = (
                await runtime.generate(
                    gen_req
                )
            )

        except RuntimeError as exc:
            logger.error(
                "Generation error: %s",
                exc,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Generation failed"
                ),
            ) from exc

        return ChatResponse(
            response_text=(
                result.final_text
            )
        )

    # ==================================
    # Persistent conversations
    # ==================================

    @app.post(
        "/conversations",
        response_model=(
            ConversationResponse
        ),
        status_code=(
            status.HTTP_201_CREATED
        ),
    )
    async def create_conversation(
        request: (
            CreateConversationRequest
        ),
    ) -> ConversationResponse:
        service = (
            app.state.chat_history
        )

        conversation = (
            await service
            .create_conversation(
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
        service = (
            app.state.chat_history
        )

        conversations = (
            await service
            .list_conversations()
        )

        return [
            conversation_summary(
                item
            )
            for item
            in conversations
        ]

    @app.get(
        (
            "/conversations/"
            "{conversation_id}"
        ),
        response_model=(
            ConversationResponse
        ),
    )
    async def get_conversation(
        conversation_id: str,
    ) -> ConversationResponse:
        service = (
            app.state.chat_history
        )

        try:
            conversation = (
                await service
                .get_conversation(
                    conversation_id
                )
            )

        except (
            ConversationNotFoundError
        ) as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found"
                ),
            ) from exc

        return conversation_detail(
            conversation
        )

    @app.delete(
        (
            "/conversations/"
            "{conversation_id}"
        ),
        status_code=(
            status.HTTP_204_NO_CONTENT
        ),
    )
    async def delete_conversation(
        conversation_id: str,
    ) -> Response:
        service = (
            app.state.chat_history
        )

        try:
            await (
                service
                .delete_conversation(
                    conversation_id
                )
            )

        except (
            ConversationNotFoundError
        ) as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found"
                ),
            ) from exc

        return Response(
            status_code=(
                status.HTTP_204_NO_CONTENT
            )
        )

    @app.post(
        (
            "/conversations/"
            "{conversation_id}/chat"
        ),
        response_model=(
            ConversationResponse
        ),
    )
    async def send_conversation_message(
        conversation_id: str,
        request: (
            SendConversationMessageRequest
        ),
    ) -> ConversationResponse:
        """
        Persistent chat endpoint.

        ChatHistoryService receives
        app_runtime, which means production
        generation also passes through
        CachedLLMRuntime before persistence.
        """

        service = (
            app.state.chat_history
        )

        if not (
            app.state.runtime
            .is_loaded()
        ):
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model not loaded"
                ),
            )

        try:
            conversation = (
                await service
                .send_message(
                    conversation_id,
                    request.content,
                )
            )

        except (
            ConversationNotFoundError
        ) as exc:
            raise HTTPException(
                status_code=404,
                detail=(
                    "Conversation not found"
                ),
            ) from exc

        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return conversation_detail(
            conversation
        )

    # ==================================
    # Error handling
    # ==================================

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
            for error
            in exc.errors()
        ]

        return error_response(
            400,
            detail=(
                "Request validation "
                "failed"
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
            for error
            in exc.errors()
        ]

        return error_response(
            400,
            detail=(
                "Request validation "
                "failed"
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
            detail=(
                "Internal server error"
            ),
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
            detail=(
                "Internal server error"
            ),
            code="unknown_error",
        )

    return app


app = create_app()