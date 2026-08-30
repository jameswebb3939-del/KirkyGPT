from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_followups.prompting import render_chat_prompt


DEFAULT_PROMPTS: list[str] = [
    "Ask me 3 clarifying questions so you can help with designing repository + unit-of-work patterns.",
    "Write 3 varied follow-up questions that are specific to using SQLAlchemy 2.0 async sessions correctly.",
    "Before helping with FastAPI API design, ask me 3 useful clarifying questions.",
    "Generate 3 specific follow-up questions about preparing a JSONL dataset for SFT training, not generic ones.",
    "Ask me 3 concrete questions that would clarify my exact needs for debugging a failing pytest test.",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS

    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            if not text:
                continue
            if text.startswith("{"):
                obj = json.loads(text)
                messages = obj.get("messages", [])
                user_messages = [
                    message
                    for message in messages
                    if isinstance(message, dict) and message.get("role") == "user"
                ]
                if user_messages:
                    prompts.append(str(user_messages[-1].get("content", "")))
                else:
                    prompts.append(str(obj.get("prompt", "")))
            else:
                prompts.append(text)

    return [prompt for prompt in prompts if prompt]


def build_chat_prompt(tokenizer: Any, user_prompt: str, k: int) -> str:
    return render_chat_prompt(
        tokenizer,
        [{"role": "user", "content": user_prompt}],
        min_questions=k,
        bullet_style="dash",
        add_generation_prompt=True,
    )


def validate_strict_format(text: str, k: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    lines = text.strip().splitlines()

    if len(lines) != k:
        errors.append(f"Expected exactly {k} lines, got {len(lines)}.")

    for index, line in enumerate(lines, start=1):
        clean_line = line.strip()
        if not clean_line.startswith("- "):
            errors.append(f"Line {index} does not start with '- '.")
        if not clean_line.endswith("?"):
            errors.append(f"Line {index} does not end with '?'.")
        if re.match(r"^\s*\d+[\).\s]", line):
            errors.append(f"Line {index} appears to use numbering.")
        if len(clean_line) < 10:
            errors.append(f"Line {index} is too short.")

    return len(errors) == 0, errors


def load_model(model_path: Path, device: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    resolved_device = device
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    if resolved_device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.to(resolved_device)
    model.eval()
    return tokenizer, model, resolved_device


def generate_once(
    tokenizer: Any,
    model: Any,
    device: str,
    user_prompt: str,
    k: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
) -> tuple[str, float]:
    prompt_text = build_chat_prompt(tokenizer, user_prompt, k)
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        padding=False,
        truncation=True,
    )
    inputs = {name: value.to(device) for name, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": min(45, max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "do_sample": do_sample,
        "repetition_penalty": 1.1,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = output_ids[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return text, latency_ms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sanity inference on a fine-tuned follow-up question model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=None)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    prompts = load_prompts(args.prompts)
    if args.limit is not None:
        prompts = prompts[: args.limit]

    print(f"Loading model from: {args.model_path}")
    tokenizer, model, device = load_model(args.model_path, args.device)
    print(f"Device: {device}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {model.get_input_embeddings().weight.shape[0]}")
    print(f"Pad token id: {tokenizer.pad_token_id}")
    print(f"EOS token id: {tokenizer.eos_token_id}\n")

    valid_count = 0
    total_latency = 0.0

    for index, prompt in enumerate(prompts, start=1):
        print("=" * 90)
        print(f"Prompt {index}:\n{prompt}\n")

        raw_output, latency_ms = generate_once(
            tokenizer=tokenizer,
            model=model,
            device=device,
            user_prompt=prompt,
            k=args.k,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.sample,
        )
        is_valid, errors = validate_strict_format(raw_output, args.k)
        valid_count += int(is_valid)
        total_latency += latency_ms

        print("Raw model output:")
        print(raw_output)
        print(f"\nStrict format valid: {is_valid}")
        print(f"Latency: {latency_ms:.1f} ms")
        if errors:
            print("Errors:")
            for error in errors:
                print(f"- {error}")
        print()

    total = len(prompts)
    valid_pct = valid_count / total * 100 if total else 0.0
    avg_latency = total_latency / total if total else 0.0
    print("=" * 90)
    print("Summary")
    print(f"Prompts tested: {total}")
    print(f"Strict format valid: {valid_count}/{total} ({valid_pct:.1f}%)")
    print(f"Average latency: {avg_latency:.1f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
