from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# ----------------------------
# Topic specifications
# ----------------------------

@dataclass(frozen=True)
class TopicSpec:
    subject: str
    components: list[str]
    decisions: list[str]
    tools: list[str]
    constraints: list[str]


DEFAULT_TOPICS: list[TopicSpec] = [
    TopicSpec(
        subject="designing repository and Unit of Work patterns with SQLAlchemy",
        components=[
            "repositories",
            "Unit of Work boundary",
            "SQLAlchemy sessions",
            "domain entities",
            "ORM models",
            "service layer",
        ],
        decisions=[
            "commit ownership",
            "rollback handling",
            "repository granularity",
            "transaction boundaries",
            "test doubles",
            "domain model separation",
        ],
        tools=[
            "SQLAlchemy 2.0",
            "FastAPI",
            "MySQL",
            "SQLite tests",
            "async sessions",
            "pytest",
        ],
        constraints=[
            "async support",
            "transaction safety",
            "testability",
            "clean architecture",
            "low coupling",
            "maintainability",
        ],
    ),
    TopicSpec(
        subject="using SQLAlchemy 2.0 async sessions correctly",
        components=[
            "async session lifecycle",
            "engine configuration",
            "dependency injection",
            "queries",
            "transactions",
            "connection pooling",
        ],
        decisions=[
            "session-per-request",
            "manual commits",
            "lazy loading",
            "connection pooling",
            "exception rollback",
            "query construction",
        ],
        tools=[
            "AsyncSession",
            "select()",
            "asyncmy",
            "FastAPI Depends",
            "pytest-asyncio",
            "SQLAlchemy 2.0",
        ],
        constraints=[
            "avoiding leaked sessions",
            "clear transaction boundaries",
            "production database compatibility",
            "repeatable tests",
            "safe rollbacks",
            "predictable lifecycle",
        ],
    ),
    TopicSpec(
        subject="designing a REST API with FastAPI",
        components=[
            "route handlers",
            "service layer",
            "Pydantic schemas",
            "status codes",
            "dependency injection",
            "error responses",
        ],
        decisions=[
            "request validation",
            "response models",
            "error mapping",
            "pagination",
            "authentication checks",
            "route structure",
        ],
        tools=[
            "FastAPI",
            "Uvicorn",
            "Pydantic v2",
            "Swagger docs",
            "TestClient",
            "httpx",
        ],
        constraints=[
            "consistent responses",
            "clear validation errors",
            "maintainable routes",
            "test coverage",
            "API versioning",
            "client usability",
        ],
    ),
    TopicSpec(
        subject="creating a data validation schema with Pydantic",
        components=[
            "request models",
            "response models",
            "field validators",
            "nested schemas",
            "default values",
            "serialization rules",
        ],
        decisions=[
            "strict typing",
            "model reuse",
            "error messages",
            "field constraints",
            "schema boundaries",
            "validation strategy",
        ],
        tools=[
            "Pydantic v2",
            "FastAPI",
            "Annotated types",
            "model validators",
            "JSON schema",
            "pytest",
        ],
        constraints=[
            "clear validation failures",
            "backwards compatibility",
            "minimal duplication",
            "strong typing",
            "client-friendly errors",
            "maintainable schemas",
        ],
    ),
    TopicSpec(
        subject="building a Flask app with Docker Compose",
        components=[
            "Flask routes",
            "Dockerfile",
            "Compose services",
            "environment variables",
            "volumes",
            "database service",
        ],
        decisions=[
            "container startup",
            "service networking",
            "volume persistence",
            "debug configuration",
            "local database setup",
            "production parity",
        ],
        tools=[
            "Flask",
            "Docker",
            "Docker Compose",
            "Gunicorn",
            "Redis",
            "MySQL",
        ],
        constraints=[
            "fast local setup",
            "reproducible environments",
            "clear logs",
            "portable configuration",
            "minimal image size",
            "developer usability",
        ],
    ),
    TopicSpec(
        subject="working with DynamoDB in LocalStack",
        components=[
            "DynamoDB tables",
            "partition keys",
            "sort keys",
            "boto3 clients",
            "LocalStack endpoints",
            "test fixtures",
        ],
        decisions=[
            "table schema",
            "access patterns",
            "conditional writes",
            "pagination",
            "index choice",
            "test isolation",
        ],
        tools=[
            "DynamoDB",
            "boto3",
            "LocalStack",
            "Docker Compose",
            "pytest",
            "AWS SDK",
        ],
        constraints=[
            "no real AWS cost",
            "repeatable tests",
            "local-only credentials",
            "fast startup",
            "realistic AWS behavior",
            "efficient queries",
        ],
    ),
    TopicSpec(
        subject="uploading and downloading from S3 with boto3",
        components=[
            "upload endpoint",
            "download flow",
            "object keys",
            "bucket configuration",
            "presigned URLs",
            "file validation",
        ],
        decisions=[
            "direct upload",
            "server-side upload",
            "object naming",
            "access control",
            "content type checks",
            "metadata storage",
        ],
        tools=[
            "S3",
            "boto3",
            "FastAPI UploadFile",
            "LocalStack",
            "presigned URLs",
            "pytest",
        ],
        constraints=[
            "safe file types",
            "large file support",
            "private access",
            "idempotent naming",
            "test isolation",
            "credential safety",
        ],
    ),
    TopicSpec(
        subject="storing sessions in Redis",
        components=[
            "session data",
            "Redis keys",
            "TTL values",
            "serialization",
            "cache invalidation",
            "login flow",
        ],
        decisions=[
            "session expiry",
            "key naming",
            "cache-aside flow",
            "fallback behavior",
            "session security",
            "data shape",
        ],
        tools=[
            "Redis",
            "FastAPI",
            "Docker Compose",
            "redis-py",
            "pytest",
            "JWT",
        ],
        constraints=[
            "avoiding stale data",
            "low latency",
            "safe session handling",
            "simple invalidation",
            "local development support",
            "predictable expiry",
        ],
    ),
    TopicSpec(
        subject="debugging a failing pytest test",
        components=[
            "test function",
            "fixture setup",
            "assertion failure",
            "mocked dependency",
            "test database",
            "environment variable",
        ],
        decisions=[
            "fixture scope",
            "mocking strategy",
            "test isolation",
            "assertion style",
            "database cleanup",
            "test ordering",
        ],
        tools=[
            "pytest",
            "pytest-asyncio",
            "monkeypatch",
            "TestClient",
            "coverage",
            "tmp_path",
        ],
        constraints=[
            "repeatable tests",
            "fast feedback",
            "isolated state",
            "clear failure messages",
            "CI compatibility",
            "minimal flakiness",
        ],
    ),
    TopicSpec(
        subject="preparing a JSONL dataset for SFT training",
        components=[
            "chat messages",
            "assistant outputs",
            "metadata fields",
            "train split",
            "validation split",
            "format validator",
        ],
        decisions=[
            "message format",
            "max length",
            "temperature fields",
            "duplicate handling",
            "quality filtering",
            "topic coverage",
        ],
        tools=[
            "JSONL",
            "Python scripts",
            "Hugging Face datasets",
            "tokenizer",
            "validation checks",
            "Transformers",
        ],
        constraints=[
            "valid JSON per line",
            "strict output format",
            "low duplication",
            "topic variety",
            "consistent roles",
            "reproducibility",
        ],
    ),
    TopicSpec(
        subject="fine-tuning a small language model locally",
        components=[
            "base model",
            "training dataset",
            "tokenizer",
            "training loop",
            "checkpoint output",
            "generation settings",
        ],
        decisions=[
            "learning rate",
            "number of epochs",
            "batch size",
            "save strategy",
            "evaluation sample",
            "CPU versus GPU usage",
        ],
        tools=[
            "Transformers",
            "Trainer",
            "PyTorch",
            "Hugging Face",
            "safetensors",
            "tokenizer",
        ],
        constraints=[
            "limited disk space",
            "CPU or GPU availability",
            "stable loss",
            "valid checkpoints",
            "reproducibility",
            "low memory usage",
        ],
    ),
    TopicSpec(
        subject="writing a clean README for a GitHub project",
        components=[
            "project overview",
            "quickstart section",
            "installation steps",
            "usage examples",
            "environment variables",
            "project structure",
        ],
        decisions=[
            "target audience",
            "setup instructions",
            "example commands",
            "screenshots",
            "API documentation",
            "known limitations",
        ],
        tools=[
            "Markdown",
            "GitHub",
            "README",
            "terminal commands",
            "badges",
            "code blocks",
        ],
        constraints=[
            "clear onboarding",
            "recruiter readability",
            "developer usability",
            "accurate commands",
            "concise writing",
            "maintainability",
        ],
    ),
    TopicSpec(
        subject="setting up a pyproject.toml for a Python package",
        components=[
            "package metadata",
            "dependencies",
            "dev dependencies",
            "console scripts",
            "build backend",
            "tool configuration",
        ],
        decisions=[
            "package name",
            "Python version support",
            "dependency groups",
            "CLI entry points",
            "linting rules",
            "test configuration",
        ],
        tools=[
            "pyproject.toml",
            "setuptools",
            "hatchling",
            "ruff",
            "pytest",
            "mypy",
        ],
        constraints=[
            "src layout compatibility",
            "reproducible installs",
            "minimal dependencies",
            "clear CLI commands",
            "tool consistency",
            "packaging correctness",
        ],
    ),
    TopicSpec(
        subject="writing a CLI tool with argparse",
        components=[
            "argument parser",
            "subcommands",
            "flags",
            "output formatting",
            "exit codes",
            "error handling",
        ],
        decisions=[
            "command structure",
            "required arguments",
            "default values",
            "JSON output",
            "help text",
            "validation",
        ],
        tools=[
            "argparse",
            "Path",
            "json",
            "stdout",
            "stderr",
            "PowerShell",
        ],
        constraints=[
            "clear UX",
            "scriptability",
            "useful errors",
            "cross-platform paths",
            "stable output",
            "minimal dependencies",
        ],
    ),
]


PROMPT_TEMPLATES: list[str] = [
    "Ask me {k} clarifying questions so you can help with {subject}.",
    "Write {k} varied follow-up questions that are specific to {subject}.",
    "Before answering, ask me {k} follow-up questions about {subject}.",
    "Generate {k} concise clarification questions for a developer working on {subject}.",
    "Give me {k} follow-up questions to understand my requirements for {subject}.",
    "Ask {k} technical questions that would clarify a task about {subject}.",
    "Create {k} useful follow-up questions for planning {subject}.",
    "What are {k} follow-up questions you would ask about {subject}?",
    "Ask {k} concrete questions that would make your help with {subject} more accurate.",
    "Generate {k} specific follow-up questions about {subject}, not generic ones.",
]


QUESTION_TEMPLATES: list[str] = [
    "Which {component} is the main source of uncertainty in your current design?",
    "Are you trying to optimize {decision} for {constraint}, or is another priority more important?",
    "What {tool} setup are you using, and is it already working locally?",
    "Should the solution prioritize {constraint}, {constraint2}, or ease of implementation?",
    "Where in the flow does {component} currently fit?",
    "What behavior do you expect from {component} when an error occurs?",
    "Do you already have tests covering {decision}, or should testing be part of the design?",
    "Which part needs the most help: {component}, {component2}, or {component3}?",
    "Is the goal to explain the concept, implement it, debug it, or improve an existing version?",
    "What does the current implementation do, and what behavior do you want instead?",
    "Are there compatibility requirements around {tool}, {tool2}, or {tool3}?",
    "Should the design be optimized for local development, production deployment, or automated tests?",
    "What input data, API request, or example case should the solution handle first?",
    "How should failures be surfaced to the caller: exceptions, status codes, logs, or structured errors?",
    "Which trade-off matters most here: simplicity, performance, reliability, or maintainability?",
    "Are you building this from scratch, refactoring existing code, or adding it to a working project?",
    "What constraints do you have around {constraint}, {constraint2}, or {constraint3}?",
    "Should the answer include code, architecture guidance, test strategy, or troubleshooting steps?",
    "What part of {subject} needs to be decided before implementation can start?",
    "How will you know the {component} implementation is correct?",
    "Which {decision} choice are you leaning toward, and why?",
    "What failure case should the design handle before everything else?",
    "Do you want the answer to focus on implementation, testing, debugging, or architecture?",
    "What existing code or folder structure does this need to fit into?",
    "Should the solution be minimal for learning or robust enough for production-style use?",
]


# ----------------------------
# Generation helpers
# ----------------------------

def unique_sample(items: list[str], count: int) -> list[str]:
    unique_items = list(dict.fromkeys(items))
    if len(unique_items) >= count:
        return random.sample(unique_items, count)
    return unique_items


def force_question(text: str) -> str:
    text = " ".join(text.strip().split())
    text = text.rstrip(".?!")
    return f"{text}?"


def render_question(template: str, topic: TopicSpec) -> str:
    components = unique_sample(topic.components, 3)
    decisions = unique_sample(topic.decisions, 3)
    tools = unique_sample(topic.tools, 3)
    constraints = unique_sample(topic.constraints, 3)

    def get(values: list[str], index: int) -> str:
        return values[index] if index < len(values) else values[0]

    question = template.format(
        subject=topic.subject,
        component=get(components, 0),
        component2=get(components, 1),
        component3=get(components, 2),
        decision=get(decisions, 0),
        decision2=get(decisions, 1),
        decision3=get(decisions, 2),
        tool=get(tools, 0),
        tool2=get(tools, 1),
        tool3=get(tools, 2),
        constraint=get(constraints, 0),
        constraint2=get(constraints, 1),
        constraint3=get(constraints, 2),
    )

    return force_question(question)


def make_prompt(topic: TopicSpec, k: int, index: int) -> str:
    template = random.choice(PROMPT_TEMPLATES)
    prompt = template.format(k=k, subject=topic.subject)

    # Add occasional focus area to reduce duplicate prompts.
    if index % 3 == 0:
        focus = random.choice(topic.components)
        prompt = f"{prompt} Focus on {focus}."

    return prompt


def make_response(
    topic: TopicSpec,
    k: int,
    question_counts: Counter[str],
    max_question_repeat: int,
) -> str | None:
    random_templates = random.sample(QUESTION_TEMPLATES, len(QUESTION_TEMPLATES))

    lines: list[str] = []
    used_questions: set[str] = set()

    for template in random_templates:
        question = render_question(template, topic)

        if question in used_questions:
            continue

        bullet_line = f"- {question}"

        if question_counts[bullet_line] >= max_question_repeat:
            continue

        lines.append(bullet_line)
        used_questions.add(question)

        if len(lines) == k:
            break

    if len(lines) != k:
        return None

    return "\n".join(lines)


def make_example(
    topic: TopicSpec,
    k: int,
    index: int,
    question_counts: Counter[str],
    max_question_repeat: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any] | None:
    prompt = make_prompt(topic=topic, k=k, index=index)
    response = make_response(
        topic=topic,
        k=k,
        question_counts=question_counts,
        max_question_repeat=max_question_repeat,
    )

    if response is None:
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }


def validate_examples(examples: list[dict[str, Any]], k: int) -> dict[str, Any]:
    strict_failures = 0
    question_lines: list[str] = []
    response_blocks: list[str] = []
    prompts: list[str] = []

    for example in examples:
        messages = example.get("messages", [])

        if len(messages) != 2:
            strict_failures += 1
            continue

        user = messages[0]
        assistant = messages[1]

        prompts.append(user.get("content", ""))
        response = assistant.get("content", "")
        response_blocks.append(response)

        lines = response.split("\n")
        ok = (
            len(lines) == k
            and all(line.startswith("- ") for line in lines)
            and all(line.endswith("?") for line in lines)
        )

        if not ok:
            strict_failures += 1

        question_lines.extend(lines)

    return {
        "examples": len(examples),
        "strict_format_failures": strict_failures,
        "unique_question_lines": len(set(question_lines)),
        "total_question_lines": len(question_lines),
        "unique_responses": len(set(response_blocks)),
        "total_responses": len(response_blocks),
        "unique_prompts": len(set(prompts)),
        "total_prompts": len(prompts),
        "top_repeated_questions": Counter(question_lines).most_common(15),
    }


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="\n") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")


def write_report(path: Path, stats: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "SFT dataset validation report",
        "=" * 40,
        f"Examples: {stats['examples']}",
        f"Strict format failures: {stats['strict_format_failures']}",
        f"Unique question lines: {stats['unique_question_lines']} / {stats['total_question_lines']}",
        f"Unique assistant responses: {stats['unique_responses']} / {stats['total_responses']}",
        f"Unique prompts: {stats['unique_prompts']} / {stats['total_prompts']}",
        "",
        "Top repeated question lines:",
    ]

    for question, count in stats["top_repeated_questions"]:
        lines.append(f"{count:>4}  {question}")

    path.write_text("\n".join(lines), encoding="utf-8")


def generate_dataset(
    n: int,
    k: int,
    seed: int,
    max_question_repeat: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    random.seed(seed)

    examples: list[dict[str, Any]] = []
    question_counts: Counter[str] = Counter()
    seen_response_blocks: set[str] = set()

    attempts = 0
    max_attempts = n * 20

    while len(examples) < n and attempts < max_attempts:
        attempts += 1

        topic = random.choice(DEFAULT_TOPICS)

        example = make_example(
            topic=topic,
            k=k,
            index=len(examples),
            question_counts=question_counts,
            max_question_repeat=max_question_repeat,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if example is None:
            continue

        response = example["messages"][1]["content"]
        if response in seen_response_blocks:
            continue

        for line in response.split("\n"):
            question_counts[line] += 1

        seen_response_blocks.add(response)
        examples.append(example)

    if len(examples) < n:
        raise RuntimeError(
            f"Could only generate {len(examples)} examples after {attempts} attempts. "
            "Increase --max-question-repeat or reduce --n."
        )

    return examples


# ----------------------------
# CLI
# ----------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the canonical JSONL "
            "SFT dataset for follow-up "
            "question generation."
        )
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/sft_followups.jsonl"),
    )

    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path("data/sft_followups_report.txt"),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=2000,
    )

    parser.add_argument(
        "--k",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-question-repeat",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.out.exists():
        raise SystemExit(
            "Refusing to overwrite existing dataset: "
            f"{args.out}\n"
            "Choose a different --out path if you "
            "want to generate another dataset."
        )

    if args.n < 1:
        raise SystemExit(
            "--n must be at least 1"
        )

    if args.k < 1:
        raise SystemExit(
            "--k must be at least 1"
        )

    examples = generate_dataset(
        n=args.n,
        k=args.k,
        seed=args.seed,
        max_question_repeat=(
            args.max_question_repeat
        ),
        max_new_tokens=(
            args.max_new_tokens
        ),
        temperature=(
            args.temperature
        ),
        top_p=args.top_p,
    )

    stats = validate_examples(
        examples,
        k=args.k,
    )

    if (
        stats[
            "strict_format_failures"
        ]
        != 0
    ):
        raise SystemExit(
            "Generated dataset has "
            f"{stats['strict_format_failures']} "
            "strict format failures."
        )

    # Only ONE canonical dataset is
    # persisted.
    write_jsonl(
        args.out,
        examples,
    )

    write_report(
        args.report_out,
        stats,
    )

    print(
        "Wrote canonical dataset: "
        f"{args.out}"
    )

    print(
        f"Examples: {len(examples)}"
    )

    print(
        "Wrote validation report: "
        f"{args.report_out}"
    )

    print()
    print("Validation summary:")

    print(
        f"Examples: "
        f"{stats['examples']}"
    )

    print(
        "Strict format failures: "
        f"{stats['strict_format_failures']}"
    )

    print(
        "Unique question lines: "
        f"{stats['unique_question_lines']} "
        "/ "
        f"{stats['total_question_lines']}"
    )

    print(
        "Unique assistant responses: "
        f"{stats['unique_responses']} "
        "/ "
        f"{stats['total_responses']}"
    )

    print(
        "Unique prompts: "
        f"{stats['unique_prompts']} "
        "/ "
        f"{stats['total_prompts']}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())