from __future__ import annotations
from typing import Literal, List
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_new_tokens: int | None = Field(None, ge=1, le=2048)
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    top_p: float | None = Field(None, ge=0.0, le=1.0)

class ChatResponse(BaseModel):
    response_text: str = Field(..., min_length=1)

class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_loaded: bool
    model_name: str
    device: Literal["cpu", "cuda", "unknown"]
    adapter_loaded: bool = False

class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
    errors: list[str] | None = None

