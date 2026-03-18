from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from llm_followups.utils.config import get_settings, Settings
from llm_followups.server.schemas import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from llm_followups.server.llm_runtime import LLMRuntime

logger = logging.getLogger(__name__)


def error_response(status_code: int, *, detail: str, code: str | None = None, errors: list[str] | None = None) -> JSONResponse:
    """Helper to construct error responses."""
    payload = ErrorResponse(detail=detail, code=code, errors=errors).model_dump()
    return JSONResponse(status_code=status_code, content=payload)


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Args:
        settings: Optional Settings object; if None, loads from environment.
    
    Returns:
        Configured FastAPI app instance.
    """
    if settings is None:
        settings = get_settings()
    
    app = FastAPI(
        title="LLM Followups Server",
        version="1.0.0",
        description="Generate follow-up questions using an LLM"
    )
    
    # Initialize runtime (not yet loaded)
    runtime = LLMRuntime(settings)
    
    # Store runtime and settings in app state for access in route handlers
    app.state.runtime = runtime
    app.state.settings = settings
    
    @app.on_event("startup")
    async def on_startup() -> None:
        """
        Load the model and tokenizer on application startup.
        """
        logger.info("Loading LLM runtime...")
        try:
            await app.state.runtime.load()
            logger.info("LLM runtime loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LLM runtime: {e}")
            raise
    
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """
        Health check endpoint.
        
        Returns:
            HealthResponse with model status and device info.
        """
        runtime = app.state.runtime
        return HealthResponse(
            status="ok",
            model_loaded=runtime.is_loaded(),
            model_name=runtime.model_name(),
            device=runtime.device_str(),
            adapter_loaded=runtime.adapter_loaded(),
        )
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(req: ChatRequest) -> ChatResponse:
        """
        Chat endpoint: accept user messages and generate follow-up questions.
        
        Args:
            req: ChatRequest with messages and optional generation parameters.
        
        Returns:
            ChatResponse with the final follow-up questions.
        
        Raises:
            HTTPException: 503 if model not loaded, 400 if request invalid.
        """
        # Check if model is loaded
        runtime = app.state.runtime
        if not runtime.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please try again later.",
                headers={"Retry-After": "60"},
            )
        
        # Make generation request (validates and resolves parameters)
        try:
            gen_req = runtime.make_request(
                req.messages,
                max_new_tokens=req.max_new_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        # Generate response
        try:
            result = await runtime.generate(gen_req)
        except RuntimeError as e:
            logger.error(f"Generation error: {e}")
            raise HTTPException(status_code=500, detail="Generation failed")
        
        return ChatResponse(response_text=result.final_text)
    
    @app.exception_handler(ValidationError)
    async def validation_error_handler(request: Request, exc: ValidationError) -> JSONResponse:
        """
        Handle Pydantic validation errors (malformed JSON, missing fields, etc.).
        Returns 400 Bad Request.
        """
        logger.debug(f"Validation error: {exc}")
        errors = [str(err) for err in exc.errors()]
        return error_response(
            400,
            detail="Request validation failed",
            code="validation_error",
            errors=errors,
        )
    
    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        """
        Handle ValueError (e.g., invalid parameter ranges in make_request).
        Returns 400 Bad Request.
        """
        logger.debug(f"Value error: {exc}")
        return error_response(400, detail=str(exc), code="invalid_parameter")
    
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request: Request, exc: RuntimeError) -> JSONResponse:
        """
        Handle RuntimeError (e.g., model not initialized during generation).
        Returns 500 Internal Server Error.
        """
        logger.error(f"Runtime error: {exc}")
        return error_response(500, detail="Internal server error", code="runtime_error")
    
    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """
        Catch-all for unexpected exceptions.
        Returns 500 Internal Server Error.
        """
        logger.exception(f"Unexpected error: {exc}")
        return error_response(500, detail="Internal server error", code="unknown_error")
    
    return app


# Create the app instance for uvicorn
app = create_app()