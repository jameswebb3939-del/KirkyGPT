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
    """
    Create and configure the argument parser for the CLI.
    
    Returns:
        ArgumentParser configured with CLI options for host, port, timeout, and model parameters.
    """
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
    """
    Parse command-line arguments and return a CliOptions dataclass.
    
    Args:
        argv: Optional list of command-line arguments. If None, uses sys.argv.
    
    Returns:
        CliOptions with parsed configuration.
    """
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
    """
    Perform a health check on the LLM server.
    
    Args:
        client: HTTP async client for making requests.
        base_url: Base URL of the server.
    
    Returns:
        HealthResponse with server health information.
    
    Raises:
        RuntimeError: If health check fails with non-200 status code.
    """
    resp = await client.get(base_url + "/health")
    if resp.status_code != 200:
        raise RuntimeError(f"Health check failed with status {resp.status_code}: {resp.text[:200]}")
    return HealthResponse(**resp.json())

def print_health(h: HealthResponse) -> None:
    """
    Print health response information to stdout.
    
    Args:
        h: HealthResponse object to display.
    """
    print(f"Here is the following health on:\nStatus: {h.status}\nModel Loaded: {h.model_loaded}\nModel Name: {h.model_name}\nDevice: {h.device}\nAdapter Loaded: {h.adapter_loaded}")

def read_user_input(prompt: str = "> ") -> str | None:
    """
    Read user input from stdin, handling exit commands.
    
    Args:
        prompt: Prompt string to display.
    
    Returns:
        Stripped user input, or None if EOF or exit command is entered.
    """
    try:
        line = input(prompt)
    except EOFError:
        return None
    s = line.strip()
    if s in (":q", ":quit", "quit", "exit"):
        return None
    return s

def handle_local_command(cmd: str, history: list[ChatMessage]) -> Literal["not_command", "continue", "exit"]:
    """
    Handle local CLI commands like :reset, :history, and :quit.
    
    Args:
        cmd: Command string to process.
        history: Chat message history list to manipulate.
    
    Returns:
        "not_command" if cmd doesn't start with ":", "continue" to continue REPL, or "exit" to quit.
    """
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
    """
    Append a message to the chat history.
    
    Args:
        history: Chat message history list.
        role: Message role ('user' or 'assistant').
        content: Message content text.
    """
    history.append(ChatMessage(role=role, content=content))

async def send_chat(client: httpx.AsyncClient, base_url: str, history: list[ChatMessage], *, max_new_tokens: int | None, temperature: float | None, top_p: float | None) -> str:
    """
    Send a chat request to the server and return the response.
    
    Args:
        client: HTTP async client for making requests.
        base_url: Base URL of the server.
        history: Chat message history to send.
        max_new_tokens: Maximum tokens to generate, or None to use server default.
        temperature: Sampling temperature, or None to use server default.
        top_p: Nucleus sampling parameter, or None to use server default.
    
    Returns:
        Response text from the server.
    
    Raises:
        RuntimeError: If chat request fails with non-200 status code.
    """
    req = ChatRequest(messages=history, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p)
    resp = await client.post(base_url + "/chat", json=req.model_dump())
    if resp.status_code != 200:
        error_data = resp.json()
        raise RuntimeError(f"Chat request failed with status {resp.status_code}: {error_data.get('detail', 'Unknown error')}")

    chat_response = ChatResponse(**resp.json())
    return chat_response.response_text

def print_assistant(text: str) -> None:
    """
    Print assistant response with blank lines before and after.
    
    Args:
        text: Response text to print.
    """
    print()
    print(f"{text}")
    print()

async def run_repl(opts: CliOptions) -> int:
    """
    Run the interactive REPL (Read-Eval-Print Loop) chat interface.
    
    Args:
        opts: CLI options configuration.
    
    Returns:
        Exit code (0 for success, 2 if model not loaded).
    """
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
    """
    Run a single chat request (non-interactive mode).
    
    Args:
        opts: CLI options configuration.
        message: User message to send.
    
    Returns:
        Exit code (0 for success, 2 if model not loaded).
    """
    client = httpx.AsyncClient(timeout=opts.timeout_s)
    h = await health_check(client, opts.base_url)
    if not h.model_loaded:
        return 2
    history = [ChatMessage(role="user", content=message)]
    reply = await send_chat(client, opts.base_url, history, max_new_tokens=opts.max_new_tokens, temperature=opts.temperature, top_p=opts.top_p)
    print(reply)
    await client.aclose()
    return 0


def main() -> int:
    opts = parse_cli_options()
    if opts.one_shot:
        return asyncio.run(run_once(opts, opts.one_shot))
    else:
        return asyncio.run(run_repl(opts))


if __name__ == "__main__":
    raise SystemExit(main())