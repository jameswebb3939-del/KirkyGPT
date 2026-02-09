from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from typing import Literal, Optional

@dataclass(frozen=True) 
class Settings:
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    endpoint_chat: str = "/chat"
    endpoint_health: str = "/health"
    request_timeout_s: float = 30.0
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9
    seed: int | None=None
    device: Literal["cpu", "auto"] = "cpu"
    adapter_path: Path | None=None
    enforce_format: bool = True
    min_questions: int = 3
    bullet_style: Literal["dash", "asterisk", "either"] = "either"

def get_settings(env: Optional[dict[str, str]] = None) -> Settings:
    """
    Reads env vars (or provided env dict), returns a Settings instance
    """
    if env is None:
        env = os.environ
    
    def get_env_var(key: str, default=None):
        """Helper to get value from env dict or os.environ"""
        return env.get(key, default)
    
    model_name = get_env_var("MODEL_NAME", "meta-llama/Llama-3.2-1B-Instruct")
    server_host = get_env_var("SERVER_HOST", "127.0.0.1")
    server_port = int(get_env_var("SERVER_PORT", "8000"))
    endpoint_chat = get_env_var("ENDPOINT_CHAT", "/chat")
    endpoint_health = get_env_var("ENDPOINT_HEALTH", "/health")
    request_timeout_s = float(get_env_var("REQUEST_TIMEOUT_S", "30.0"))
    max_new_tokens = int(get_env_var("MAX_NEW_TOKENS", "128"))
    temperature = float(get_env_var("TEMPERATURE", "0.2"))
    top_p = float(get_env_var("TOP_P", "0.9"))
    seed = int(get_env_var("SEED")) if get_env_var("SEED") else None
    device = get_env_var("DEVICE", "cpu")
    adapter_path = Path(get_env_var("ADAPTER_PATH")) if get_env_var("ADAPTER_PATH") else None
    enforce_format = get_env_var("ENFORCE_FORMAT", "true").lower() in ("true", "1", "yes")
    min_questions = int(get_env_var("MIN_QUESTIONS", "3"))
    bullet_style = get_env_var("BULLET_STYLE", "either")
    
    return Settings(
        model_name=model_name,
        server_host=server_host,
        server_port=server_port,
        endpoint_chat=endpoint_chat,
        endpoint_health=endpoint_health,
        request_timeout_s=request_timeout_s,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        device=device,
        adapter_path=adapter_path,
        enforce_format=enforce_format,
        min_questions=min_questions,
        bullet_style=bullet_style
    )


def server_url(settings: Settings) -> str:
    """
    Returns base URL
    """
    return f"http://{settings.server_host}:{settings.server_port}"