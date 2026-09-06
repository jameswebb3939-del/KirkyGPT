from __future__ import annotations

import json
from pathlib import Path

from llm_followups.rules.definitions import (
    DEFAULT_RULES,
)


def branch_payload(branch):
    return {
        "keywords": list(
            branch.keywords
        ),
        "response": branch.response,
    }


def step_payload(step):
    return {
        "id": step.id,
        "question": step.question,
        "branches": [
            branch_payload(branch)
            for branch
            in step.branches
        ],
        "defaultResponse":
            step.default_response,
    }


def rule_payload(rule):
    return {
        "id": rule.id,
        "keywords": list(
            rule.keywords
        ),
        "steps": [
            step_payload(step)
            for step
            in rule.steps
        ],
    }


def main() -> int:
    backend_dir = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    root = backend_dir.parent

    output = (
        root
        / "frontend"
        / "src"
        / "rules"
        / "definitions.ts"
    )

    payload = [
        rule_payload(rule)
        for rule
        in DEFAULT_RULES
    ]

    content = (
        'import type { ConversationRule } '
        'from "./models";\n\n'
        "/*\n"
        " * AUTO-GENERATED FROM:\n"
        " * backend/src/llm_followups/"
        "rules/definitions.py\n"
        " *\n"
        " * Do not edit rule content "
        "manually here.\n"
        " */\n\n"
        "export const DEFAULT_RULES: "
        "readonly ConversationRule[] = "
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + ";\n"
    )

    output.write_text(
        content,
        encoding="utf-8",
    )

    print(f"Wrote: {output}")
    print(
        f"Rules: {len(DEFAULT_RULES)}"
    )
    print(
        "IDs: "
        + ", ".join(
            rule.id
            for rule
            in DEFAULT_RULES
        )
    )
    print(
        "Steps: "
        + str(
            sum(
                len(rule.steps)
                for rule
                in DEFAULT_RULES
            )
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
