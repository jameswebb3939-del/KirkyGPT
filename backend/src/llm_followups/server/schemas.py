from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal[
        "user",
        "assistant",
    ]

    content: str = Field(
        ...,
        min_length=1,
    )


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(
        ...,
        min_length=1,
    )

    max_new_tokens: int | None = Field(
        None,
        ge=1,
        le=2048,
    )

    temperature: float | None = Field(
        None,
        ge=0.0,
        le=2.0,
    )

    top_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
    )

    seed: int | None = None


class ChatResponse(BaseModel):
    response_text: str = Field(
        ...,
        min_length=1,
    )

    raw_text: str | None = None

    used_fallback: bool = False
    used_repair: bool = False

    latency_ms: int | None = Field(
        None,
        ge=0,
    )


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str

    device: Literal[
        "cpu",
        "cuda",
        "unknown",
    ]

    adapter_loaded: bool = False


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    errors: list[str] | None = None


# ----------------------------------
# Persisted conversation schemas
# ----------------------------------


class CreateConversationRequest(
    BaseModel
):
    title: str = Field(
        default="New chat",
        min_length=1,
        max_length=200,
    )


class SendConversationMessageRequest(
    BaseModel
):
    content: str = Field(
        ...,
        min_length=1,
    )


class StoredMessageResponse(BaseModel):
    id: str

    role: Literal[
        "user",
        "assistant",
    ]

    content: str

    position: int

    created_at: datetime


class ConversationSummaryResponse(
    BaseModel
):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationResponse(
    ConversationSummaryResponse
):
    messages: list[
        StoredMessageResponse
    ]