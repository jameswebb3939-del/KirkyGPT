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
    "Before helping with {topic}, ask me {k} useful clarifying questions.",
    "Ask {k} tailored questions that would help you give better advice about {topic}.",
    "Generate {k} specific follow-up questions about {topic}, not generic ones.",
    "Ask {k} concrete questions that would clarify my exact needs for {topic}.",
    "Before answering, ask {k} questions that would make your help on {topic} more precise.",
    "Write {k} varied follow-up questions that are specific to {topic}.",
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
    
    # FastAPI
    "fastapi": [
        "Are you designing a new FastAPI service or modifying an existing one?",
        "Do you need help with routing, request validation, dependency injection, or response models?",
        "Are you working with synchronous endpoints, async endpoints, or a mix of both?",
        "Do you want help structuring the API, handling errors, or testing endpoints?",
        "Is your main goal local development, production deployment, or both?",
    ],
    
    # Flask
    "flask": [
        "Are you building a simple Flask app, a larger multi-file project, or an API service?",
        "Do you need help with routes, templates, forms, or application structure?",
        "Are you running Flask locally, in Docker, or with multiple services?",
        "Do you want help debugging a Flask issue or setting up the project from scratch?",
        "Are you using Flask mainly for learning, prototyping, or a real project?",
    ],
    
    # SQLAlchemy
    "sqlalchemy": [
        "Are you using SQLAlchemy with sync sessions or async sessions?",
        "Do you need help with models, relationships, queries, or session management?",
        "Are you trying to structure a repository pattern, unit-of-work pattern, or both?",
        "Is the issue related to database setup, querying, transactions, or testing?",
        "What database backend are you using with SQLAlchemy?",
    ],
    
    # Redis
    "redis": [
        "Are you using Redis for caching, session storage, queues, or something else?",
        "Do you need help connecting to Redis, designing keys, or handling expiry correctly?",
        "Are you working with Redis locally, in Docker, or through another service?",
        "Is your issue about storing data, retrieving data, or debugging connection problems?",
        "Do you want a simple example or help integrating Redis into an existing app?",
    ],

    # S3    
    "s3": [
        "Are you working with S3 for uploads, downloads, bucket setup, or access control?",
        "Do you need help using boto3, handling object paths, or managing credentials?",
        "Are you using real AWS S3 or a local emulator like LocalStack?",
        "Is your main issue authentication, bucket configuration, or file handling?",
        "Do you want help with a script, an app integration, or testing S3 locally?",
    ],
    
    # DynamoDB
    "dynamodb": [
        "Are you designing a new DynamoDB table or working with an existing one?",
        "Do you need help with partition keys, sort keys, or item access patterns?",
        "Are you using DynamoDB locally through LocalStack or against AWS directly?",
        "Is the challenge about table design, CRUD operations, or boto3 integration?",
        "Do you want help with a small example or with integrating DynamoDB into an app?",
    ],
    
    # Localstack
    "localstack": [
        "Which AWS service are you using through LocalStack right now?",
        "Are you trying to start LocalStack, connect an app to it, or debug a service issue?",
        "Do you need help with Docker Compose setup, endpoint configuration, or test data seeding?",
        "Is the issue specific to S3, DynamoDB, or another LocalStack-backed service?",
        "Are you aiming for local development only or for tests that mimic AWS behaviour more closely?",
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
    
    # Docker
    "docker": [
        "Are you trying to use Docker for local development, deployment, or both?",
        "Do you need help writing a Dockerfile, running containers, or using Docker Compose?",
        "What kind of application are you planning to run inside Docker?",
        "Are you working on Linux, macOS, or Windows?",
        "Do you want to understand Docker images, containers, volumes, or networking first?",
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

GENERIC_STARTERS: tuple[str, ...] = (
    "what is",
    "what are",
    "what have",
    "what does",
    "can you",
    "are there",
)

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "docker": ["docker", "dockerfile", "container", "compose", "image"],
    "fastapi": ["fastapi", "api", "endpoint", "request", "response"],
    "flask": ["flask", "app", "route", "request"],
    "sqlalchemy": ["sqlalchemy", "session", "model", "query", "database"],
    "redis": ["redis", "cache", "session", "key"],
    "s3": ["s3", "bucket", "object", "upload", "download"],
    "dynamodb": ["dynamodb", "table", "item", "partition key"],
    "localstack": ["localstack", "aws", "service", "endpoint"],
    "pytest": ["pytest", "test", "fixture", "assert"],
    "readme": ["readme", "project", "quickstart", "documentation"],
    "pyproject": ["pyproject", "package", "dependency", "build"],
    "pydantic": ["pydantic", "schema", "validation", "model"],
    "async": ["async", "await", "concurrency", "event loop"],
    "cli": ["cli", "command", "argument", "terminal"],
    "python": ["python", "script", "function", "project"],
    "sft": ["dataset", "training", "sft", "jsonl", "fine-tuning"],
}


def _topic_keywords(topic: str) -> list[str]:
    t = topic.lower()
    found: list[str] = []
    for key, words in TOPIC_KEYWORDS.items():
        if key in t:
            found.extend(words)
    if found:
        seen = set()
        out: list[str] = []
        for word in found:
            if word not in seen:
                seen.add(word)
                out.append(word)
        return out

    tokens = [tok.strip(" ,./()-").lower() for tok in topic.split()]
    return [tok for tok in tokens if len(tok) > 3][:4]


def _question_stem(text: str) -> str:
    q = text.strip().lower()
    q = q[:-1] if q.endswith("?") else q
    words = q.split()
    return " ".join(words[:3])


def _starts_generic(text: str) -> bool:
    q = text.strip().lower()
    return q.startswith(GENERIC_STARTERS)


def _mentions_topic_keyword(text: str, topic: str) -> bool:
    q = text.lower()
    return any(word in q for word in _topic_keywords(topic))

def _read_topics_file(path: Path) -> list[str]:
    topics: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        topics.append(line)
    return topics


def _choose_question_pool(topic: str) -> tuple[list[str], list[str]]:
    t = topic.lower()
    specific_pools: list[list[str]] = []
    generic_pool: list[str] = QUESTION_BANK["generic"][:]
    if "docker" in t or "compose" in t or "container" in t:
        specific_pools.append(QUESTION_BANK["docker"])

    if "fastapi" in t or "rest api" in t or "api" in t:
        specific_pools.append(QUESTION_BANK["fastapi"])

    if "flask" in t:
        specific_pools.append(QUESTION_BANK["flask"])

    if "sqlalchemy" in t or "unit-of-work" in t or "repository" in t or "database" in t:
        specific_pools.append(QUESTION_BANK["sqlalchemy"])

    if "redis" in t:
        specific_pools.append(QUESTION_BANK["redis"])

    if "s3" in t or "boto3" in t or "bucket" in t:
        specific_pools.append(QUESTION_BANK["s3"])

    if "dynamodb" in t:
        specific_pools.append(QUESTION_BANK["dynamodb"])

    if "localstack" in t:
        specific_pools.append(QUESTION_BANK["localstack"])

    if "pytest" in t or "test" in t:
        specific_pools.append(QUESTION_BANK["pytest"])

    if "readme" in t or "documentation" in t or "github" in t:
        specific_pools.append(QUESTION_BANK["readme"])

    if "pyproject" in t or "toml" in t or "packag" in t:
        specific_pools.append(QUESTION_BANK["pyproject"])

    if "sft" in t or "jsonl" in t or "fine-tun" in t or "tuning" in t or "train" in t:
        specific_pools.append(QUESTION_BANK["sft"])

    if "python" in t:
        specific_pools.append(QUESTION_BANK["python"])

    seen_specific = set()
    specific: list[str] = []
    for pool in specific_pools:
        for q in pool:
            if q not in seen_specific:
                seen_specific.add(q)
                specific.append(q)

    return specific, generic_pool


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
    specific_pool, generic_pool = _choose_question_pool(topic)

    specific_candidates = specific_pool[:]
    generic_candidates = generic_pool[:]
    rng.shuffle(specific_candidates)
    rng.shuffle(generic_candidates)

    picked: list[str] = []
    used_stems: set[str] = set()
    generic_used = 0

    def try_add(question: str) -> bool:
        nonlocal generic_used

        q = question.strip()
        if "this" in q and rng.random() < 0.35:
            q = q.replace("this", topic)

        q = _ensure_question_marks([q])[0]
        stem = _question_stem(q)

        if stem in used_stems:
            return False

        core = q[:-1].strip()
        if len(core) < min_chars:
            return False

        if _starts_generic(q) and generic_used >= 1:
            return False

        if _starts_generic(q):
            generic_used += 1

        used_stems.add(stem)
        picked.append(q)
        return True

    min_specific = min(2, k) if specific_candidates else 0

    for q in specific_candidates:
        if len(picked) >= min_specific:
            break
        try_add(q)

    mixed_candidates = specific_candidates + generic_candidates
    rng.shuffle(mixed_candidates)

    for q in mixed_candidates:
        if len(picked) >= k:
            break
        try_add(q)

    while len(picked) < k:
        keywords = _topic_keywords(topic)
        if len(picked) == 0 and keywords:
            fallback = f"Are you working with {keywords[0]} for learning, debugging, or building something specific?"
        elif len(picked) == 1 and len(keywords) >= 2:
            fallback = f"Do you need help with {keywords[0]}, {keywords[1]}, or the overall workflow for {topic}?"
        elif len(picked) == 2:
            fallback = f"What specific outcome are you trying to achieve with {topic}?"
        else:
            fallback = rng.choice(generic_pool)

        added = try_add(fallback)
        if not added:
            alt = f"What part of {topic} is most important for you right now?"
            if not try_add(alt):
                picked.append(_ensure_question_marks([alt])[0])
                break

    if not any(_mentions_topic_keyword(q, topic) for q in picked):
        keywords = _topic_keywords(topic)
        if keywords:
            picked[0] = f"Are you mainly focused on {keywords[0]} in the context of {topic}?"

    picked = _ensure_question_marks(picked)

    repaired: list[str] = []
    used_repaired_stems: set[str] = set()
    generic_used_repaired = 0

    for q in picked:
        core = q[:-1].strip() if q.endswith("?") else q.strip()
        if len(core) < min_chars:
            q = f"What specific outcome are you trying to achieve with {topic}?"

        stem = _question_stem(q)
        if stem in used_repaired_stems:
            continue

        if _starts_generic(q):
            if generic_used_repaired >= 1:
                continue
            generic_used_repaired += 1

        used_repaired_stems.add(stem)
        repaired.append(_ensure_question_marks([q])[0])

    while len(repaired) < k:
        filler = f"What part of {topic} do you want to clarify first?"
        stem = _question_stem(filler)
        if stem not in used_repaired_stems:
            used_repaired_stems.add(stem)
            repaired.append(_ensure_question_marks([filler])[0])
        else:
            repaired.append(_ensure_question_marks([f"What specific constraint matters most for {topic}?"])[0])

    return repaired[:k]

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