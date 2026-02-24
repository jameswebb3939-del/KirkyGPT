#!/usr/bin/env python3
"""
make_sft.py

Generate a JSONL dataset for "follow-up questions" SFT.

Each JSONL line is a single training example shaped like:
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "- ...?\n- ...?\n- ...?"}
  ],
  "max_new_tokens": 128,
  "temperature": 0.2,
  "top_p": 0.9
}

Goals:
- Assistant output is *only* bullet questions (no extra text)
- Bullets use "-" (hyphen) lines
- Each line ends with "?"
- No numbered lists
- Questions are not "too short"
- Deterministic output via --seed

Usage:
  PYTHONPATH=src python scripts/make_sft.py --out data/sft_followups.jsonl --n 300 --seed 42

Optional:
  PYTHONPATH=src python scripts/make_sft.py --topics data/topics.txt --out data/sft_followups.jsonl --n 500
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


# ----------------------------
# Templates / content
# ----------------------------

DEFAULT_TOPICS: list[str] = [
    "learning Python",
    "debugging a failing pytest test",
    "writing a clean README for a GitHub project",
    "setting up a pyproject.toml for a Python package",
    "understanding async/await in Python",
    "designing a REST API with FastAPI",
    "building a Flask app with Docker Compose",
    "using SQLAlchemy 2.0 async sessions correctly",
    "writing unit tests with pytest-asyncio",
    "creating a data validation schema with Pydantic",
    "preparing a JSONL dataset for SFT training",
    "fine-tuning a small language model locally",
    "improving code quality with Ruff and formatting tools",
    "logging best practices in Python",
    "structuring a Python project with src/ layout",
    "handling environment variables with .env files",
    "writing a CLI tool with argparse",
    "working with DynamoDB in LocalStack",
    "uploading and downloading from S3 with boto3",
    "storing sessions in Redis",
    "designing repository + unit-of-work patterns",
]

PROMPT_TEMPLATES: list[str] = [
    "Give me {k} follow-up questions about {topic}.",
    "Ask me {k} clarifying questions so you can help with {topic}.",
    "I need {k} follow-up questions that would improve my understanding of {topic}.",
    "Generate {k} questions you would ask before starting work on {topic}.",
    "Write {k} follow-up questions to gather requirements for {topic}.",
    "What are {k} follow-up questions you would ask about {topic}?",
]

QUESTION_BANK: dict[str, list[str]] = {
    # Generic / requirements
    "generic": [
        "What is your goal and how will you measure success for this?",
        "What constraints do you have (time, tools, environment, or requirements)?",
        "What have you tried already, and what happened when you tried it?",
        "Can you share a minimal reproducible example or a small snippet to work from?",
        "What does the expected output look like, and what is the actual output now?",
        "Are there any non-negotiables (format, style, performance, or compatibility)?",
    ],
    # Python learning
    "python": [
        "What is your current Python level, and what topics do you find hardest right now?",
        "Are you learning Python for scripting, data, backend, automation, or interviews?",
        "Do you prefer learning by projects, exercises, or reading documentation first?",
        "Which concepts are you focusing on next (functions, OOP, typing, async, testing)?",
        "What is one small project you want to build to practise this topic?",
    ],
    # Testing / pytest
    "pytest": [
        "Which test file and test function is failing, and what is the full error output?",
        "What fixtures are involved, and what scope do they use?",
        "Are you running pytest with PYTHONPATH=src or using an installed package?",
        "Do you rely on marks (unit/integration), and are those marks registered?",
        "Is the failure deterministic, or does it depend on ordering or environment variables?",
    ],
    # README / docs
    "readme": [
        "Who is the target audience for this README (recruiters, teammates, or users)?",
        "What is the simplest 'quickstart' command sequence a new user should run?",
        "What environment variables or prerequisites do you need to document clearly?",
        "What is the project structure and which entry points should a reader start with?",
        "What examples should be included to demonstrate expected inputs and outputs?",
    ],
    # pyproject
    "pyproject": [
        "What is the package name and the import path under src/ (module name)?",
        "Which Python versions do you want to support, and do you need 3.13 features?",
        "What are the runtime dependencies versus dev dependencies (tests, lint, format)?",
        "Do you want console scripts (CLI entry points), and what command name should they use?",
        "Do you want strict typing checks and Ruff rules, or keep it minimal for now?",
    ],
    # Training / SFT
    "sft": [
        "What exact output format must the assistant follow (bullets only, min questions, question marks)?",
        "Do you want the model to always ask clarifying questions, or sometimes answer and then ask?",
        "What topics should the follow-up questions cover, and what topics should be excluded?",
        "How many examples do you want to generate, and do you need train/valid splits?",
        "Do you want deterministic generation (seeded) for reproducible datasets?",
    ],
}


# ----------------------------
# Helpers
# ----------------------------

def _read_topics_file(path: Path) -> list[str]:
    topics: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        topics.append(line)
    return topics


def _choose_question_pool(topic: str) -> list[str]:
    t = topic.lower()
    pools: list[list[str]] = [QUESTION_BANK["generic"]]

    if "pytest" in t or "test" in t:
        pools.append(QUESTION_BANK["pytest"])
    if "readme" in t or "documentation" in t or "github" in t:
        pools.append(QUESTION_BANK["readme"])
    if "pyproject" in t or "toml" in t or "packag" in t:
        pools.append(QUESTION_BANK["pyproject"])
    if "sft" in t or "jsonl" in t or "fine-tun" in t or "tuning" in t or "train" in t:
        pools.append(QUESTION_BANK["sft"])
    if "python" in t:
        pools.append(QUESTION_BANK["python"])

    # Flatten unique while preserving order
    seen = set()
    flat: list[str] = []
    for pool in pools:
        for q in pool:
            if q not in seen:
                seen.add(q)
                flat.append(q)
    return flat


def _is_valid_bullet_line(line: str, *, min_chars: int) -> bool:
    # Must start with "- "
    if not line.startswith("- "):
        return False
    body = line[2:].strip()

    # Must end with '?'
    if not body.endswith("?"):
        return False

    # Must not look like numbered lists
    # (We don't allow "1. " anywhere)
    if body.lstrip().startswith(tuple(f"{i}." for i in range(1, 10))):
        return False

    # Must be long enough to avoid your validator "Too short"
    # Count characters excluding trailing '?'
    core = body[:-1].strip()
    return len(core) >= min_chars


def _format_bullets(questions: Sequence[str]) -> str:
    lines = [f"- {q.strip()}" for q in questions]
    return "\n".join(lines).strip() + "\n"


def _ensure_question_marks(questions: Sequence[str]) -> list[str]:
    out: list[str] = []
    for q in questions:
        s = q.strip()
        if not s.endswith("?"):
            s = s.rstrip(".") + "?"
        out.append(s)
    return out


def _generate_questions(rng: random.Random, topic: str, k: int, *, min_chars: int) -> list[str]:
    pool = _choose_question_pool(topic)
    # Shuffle-copy
    candidates = pool[:]
    rng.shuffle(candidates)

    picked: list[str] = []
    for q in candidates:
        if len(picked) >= k:
            break
        # Minor topic injection: occasionally customise a generic question
        if "this" in q and rng.random() < 0.35:
            q2 = q.replace("this", topic)
        else:
            q2 = q
        picked.append(q2)

    # If pool smaller than k (unlikely), fill with safe generics
    while len(picked) < k:
        picked.append(rng.choice(QUESTION_BANK["generic"]))

    picked = _ensure_question_marks(picked)

    # Validate and, if needed, repair by swapping in longer generics
    repaired: list[str] = []
    for q in picked:
        # Ensure minimum length
        core = q[:-1].strip() if q.endswith("?") else q.strip()
        if len(core) < min_chars:
            # Replace with a longer generic that passes min_chars
            replacement = None
            for cand in QUESTION_BANK["generic"]:
                cand_q = _ensure_question_marks([cand])[0]
                cand_core = cand_q[:-1].strip()
                if len(cand_core) >= min_chars:
                    replacement = cand_q
                    break
            q = replacement or (core + " (please share more details)?")
        repaired.append(q)

    return repaired


@dataclass(frozen=True)
class RowConfig:
    max_new_tokens: int = 128
    temperature: float = 0.2
    top_p: float = 0.9


def build_row(*, user_prompt: str, assistant_bullets: str, row_cfg: RowConfig) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_bullets.rstrip()},
        ],
        "max_new_tokens": row_cfg.max_new_tokens,
        "temperature": row_cfg.temperature,
        "top_p": row_cfg.top_p,
    }


def write_jsonl(rows: Iterable[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_rows(rows: Sequence[dict], *, k: int, min_chars: int) -> None:
    """
    Quick local validation so you catch bad formatting early.
    This is not your full pytest validator, but it matches the key constraints.
    """
    for i, row in enumerate(rows, 1):
        msgs = row.get("messages", [])
        if not isinstance(msgs, list) or len(msgs) < 2:
            raise ValueError(f"Row {i}: invalid messages list")

        assistant = msgs[-1].get("content", "")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError(f"Row {i}: empty assistant content")

        lines = [ln.strip() for ln in assistant.splitlines() if ln.strip()]
        if len(lines) != k:
            raise ValueError(f"Row {i}: expected exactly {k} bullet lines, got {len(lines)}")

        for ln in lines:
            if not _is_valid_bullet_line(ln, min_chars=min_chars):
                raise ValueError(f"Row {i}: invalid bullet line: {ln!r}")


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SFT JSONL for follow-up bullet questions.")
    ap.add_argument("--out", type=Path, default=Path("data/sft_followups.jsonl"), help="Output JSONL path.")
    ap.add_argument("--n", type=int, default=300, help="Number of examples to generate.")
    ap.add_argument("--k", type=int, default=3, help="Number of follow-up questions per example.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic generation.")
    ap.add_argument("--topics", type=Path, default=None, help="Optional path to newline-delimited topics.txt")
    ap.add_argument("--min-chars", type=int, default=18, help="Minimum characters per question (excluding '?').")

    # Row-level inference defaults (kept constant across rows)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)

    args = ap.parse_args()

    if args.n < 1:
        raise SystemExit("--n must be >= 1")
    if args.k < 3:
        raise SystemExit("--k should be >= 3 to satisfy your validator")
    if args.min_chars < 8:
        raise SystemExit("--min-chars is too small; keep it >= 8")

    rng = random.Random(args.seed)

    if args.topics:
        if not args.topics.exists():
            raise SystemExit(f"Topics file not found: {args.topics}")
        topics = _read_topics_file(args.topics)
        if not topics:
            raise SystemExit(f"No usable topics found in: {args.topics}")
    else:
        topics = DEFAULT_TOPICS[:]

    row_cfg = RowConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    rows: list[dict] = []
    for _ in range(args.n):
        topic = rng.choice(topics)
        prompt_tmpl = rng.choice(PROMPT_TEMPLATES)
        user_prompt = prompt_tmpl.format(k=args.k, topic=topic)

        qs = _generate_questions(rng, topic, args.k, min_chars=args.min_chars)
        bullets = _format_bullets(qs)

        rows.append(build_row(user_prompt=user_prompt, assistant_bullets=bullets, row_cfg=row_cfg))

    # Sanity-check before writing
    validate_rows(rows, k=args.k, min_chars=args.min_chars)

    write_jsonl(rows, args.out)
    print(f"Wrote {len(rows)} rows to: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())