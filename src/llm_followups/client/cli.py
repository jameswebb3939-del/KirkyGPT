from __future__ import annotations
import argparse
import asyncio
import sys
from dataclasses import dataclass
from typing import Literal, Optional

import httpx

from llm_followups.utils.config import get_settings, server_url
from llm_followups.server.schemas import ChatMessage, ChatRequest, ChatResponse, HealthResponse, ErrorResponse

@dataclass(frozen=True)
class CliOptions:
    base_url: str
    timeout_s: float
    max_new_tokens: int | None
    temperature: float | None
    top_p: float | None
    do_health_check: bool
    one_shot: str | None

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LLM Followups Chat Client"
    )
    parser.add_argument('--host', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument('--max-new-tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)
    parser.add_argument('--top-p', type=float, default=None)
    parser.add_argument('--no-health-check', action="store_true")
    parser.add_argument('--once', type=str, default=None)

    return parser

def parse_cli_options(argv: list[str] | None = None) -> CliOptions:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    base_url = f"http://{args.host}:{args.port}"

    do_health_check = not args.no_health_check

    timeout_s = args.timeout
    max_new_tokens = args.max_new_tokens
    temperature = args.temperature
    top_p = args.top_p
    one_shot = args.once

    return CliOptions(base_url=base_url,
                      timeout_s=timeout_s,
                      max_new_tokens=max_new_tokens,
                      temperature=temperature,
                      top_p=top_p,
                      do_health_check=do_health_check,
                      one_shot=one_shot)


async def health_check(client: httpx.AsyncClient, base_url: str) -> HealthResponse:
    resp = await client.get(base_url + "/health")
    if resp.status_code != 200:
        raise RuntimeError(f"Health check failed with status {resp.status_code}: {resp.text[:200]}")
    return HealthResponse(**resp.json())

def print_health(h: HealthResponse) -> None:
    print(f"Here is the following health on:\nStatus: {h.status}\nModel Loaded: {h.model_loaded}\nModel Name: {h.model_name}\nDevice: {h.device}\nAdapter Loaded: {h.adapter_loaded}")

def read_user_input(prompt: str = "> ") -> str | None:
    try:
        line = input(prompt)
    except EOFError:
        return None
    s = line.strip()
    if s in (":q", ":quit", "quit", "exit"):
        return None
    return s

def handle_local_command(cmd: str, history: list[ChatMessage]) -> Literal["not_command", "continue", "exit"]:
    if not cmd.startswith(":"):
        return "not_command"
    if cmd == ":reset":
        history.clear()
        print("History cleared")
        return "continue"
    if cmd == ":history":
        print(f"History: {history}")
        return "continue"
    if cmd in (":q", ":quit"):
        return "exit"
    else:
        print("Unknown command")
        return "continue"

def update_history(history: list[ChatMessage], *, role: Literal["user", "assistant"], content: str) -> None:
    history.append(ChatMessage(role=role, content=content))

async def send_chat(client: httpx.AsyncClient, base_url: str, history: list[ChatMessage], *, max_new_tokens: int | None, temperature: float | None, top_p: float | None) -> str:
    req = ChatRequest(messages=history, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    resp = await client.post(base_url + "/chat", json=req.model_dump())
    if resp.status_code != 200:
        error_data = resp.json()
        raise RuntimeError(f"Chat request failed with status {resp.status_code}: {error_data.get('detail', 'Unknown error')}")

    chat_response = ChatResponse(**resp.json())
    return chat_response.response_text

def print_assistant(text: str) -> None:
    print()
    print(f"{text}")
    print()

async def run_repl(opts: CliOptions) -> int:
    client = httpx.AsyncClient(timeout=opts.timeout_s)
    if opts.do_health_check:
        h = await health_check(client, opts.base_url)
        print_health(h)
        if not h.model_loaded:
            return 2
    history: list[ChatMessage] = []
    while True:
        s = read_user_input()
        if s is None:
            break
        if s == "":
            continue
        cmd_result = handle_local_command(s, history)
        if cmd_result == "not_command":
            pass
        elif cmd_result == "exit":
            break
        else:  # "continue"
            continue
        update_history(history, role="user", content=s)
        reply = await send_chat(client, opts.base_url, history, max_new_tokens=opts.max_new_tokens, temperature=opts.temperature, top_p=opts.top_p)
        print_assistant(reply)
        update_history(history, role="assistant", content=reply)
    
    await client.aclose()
    return 0

async def run_once(opts: CliOptions, message: str) -> int:
    client = httpx.AsyncClient(timeout=opts.timeout_s)
    h = await health_check(client, opts.base_url)
    if not h.model_loaded:
        return 2
    history = [ChatMessage(role="user", content=message)]
    reply = await send_chat(client, opts.base_url, history, max_new_tokens=opts.max_new_tokens, temperature=opts.temperature, top_p=opts.top_p)
    print(reply)
    await client.aclose()
    return 0


"""
def main() -> int:
    opts = parse_cli_options()
    if opts.one_shot:
        return asyncio.run(run_once(opts, opts.one_shot))
    else:
        return asyncio.run(run_repl(opts))


if __name__ == "__main__":
    raise SystemExit(main())
"""