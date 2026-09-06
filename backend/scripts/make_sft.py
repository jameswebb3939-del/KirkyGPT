from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from llm_followups.rules.definitions import DEFAULT_RULES
from llm_followups.rules.models import (
    ConversationRule,
    RuleStep,
)
from llm_followups.rules.matching import (
    contains_keyword,
)


TRIGGER_TEMPLATES = (
    "{keyword}",
    "Tell me about {keyword}",
    "I want to discuss {keyword}",
    "What about {keyword}?",
    "Let's talk about {keyword}",
)


ANSWER_TEMPLATES = (
    "{keyword}",
    "I mean {keyword}",
    "Let's go with {keyword}",
    "The main thing is {keyword}",
    "I'm interested in {keyword}",
)


def _trigger_for(
    rule: ConversationRule,
    rng: random.Random,
) -> str:
    keyword = rng.choice(
        tuple(rule.keywords)
    )

    template = rng.choice(
        TRIGGER_TEMPLATES
    )

    return template.format(
        keyword=keyword
    )


def _answer_for(
    step: RuleStep,
    rng: random.Random,
) -> str:
    if not step.branches:
        return "continue"

    branch = rng.choice(
        tuple(step.branches)
    )

    keyword = rng.choice(
        tuple(branch.keywords)
    )

    template = rng.choice(
        ANSWER_TEMPLATES
    )

    return template.format(
        keyword=keyword
    )


def _matching_branch(
    step: RuleStep,
    user_text: str,
):
    for branch in step.branches:
        if any(
            contains_keyword(
                user_text,
                keyword,
            )
            for keyword
            in branch.keywords
        ):
            return branch

    return None


def make_example(
    *,
    rule: ConversationRule,
    rng: random.Random,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    messages: list[
        dict[str, str]
    ] = []

    messages.append(
        {
            "role": "user",
            "content": _trigger_for(
                rule,
                rng,
            ),
        }
    )

    if not rule.steps:
        raise ValueError(
            f"Rule has no steps: {rule.id}"
        )

    messages.append(
        {
            "role": "assistant",
            "content": (
                rule.steps[0].question
            ),
        }
    )

    for index, step in enumerate(
        rule.steps
    ):
        answer = _answer_for(
            step,
            rng,
        )

        messages.append(
            {
                "role": "user",
                "content": answer,
            }
        )

        branch = _matching_branch(
            step,
            answer,
        )

        response = (
            branch.response
            if branch is not None
            else step.default_response
        )

        next_index = index + 1

        if next_index < len(
            rule.steps
        ):
            response = (
                f"{response}\n\n"
                f"{rule.steps[next_index].question}"
            )

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    return {
        "messages": messages,
        "max_new_tokens": (
            max_new_tokens
        ),
        "temperature": temperature,
        "top_p": top_p,
        "source": "definitions.py",
        "rule_id": rule.id,
    }


def generate_dataset(
    *,
    n: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict[str, Any]]:
    if not DEFAULT_RULES:
        raise ValueError(
            "DEFAULT_RULES is empty."
        )

    rng = random.Random(seed)

    examples: list[
        dict[str, Any]
    ] = []

    for _ in range(n):
        rule = rng.choice(
            tuple(DEFAULT_RULES)
        )

        examples.append(
            make_example(
                rule=rule,
                rng=rng,
                max_new_tokens=(
                    max_new_tokens
                ),
                temperature=temperature,
                top_p=top_p,
            )
        )

    return examples


def validate_examples(
    examples: list[
        dict[str, Any]
    ],
) -> dict[str, Any]:
    failures = 0

    rule_ids: set[str] = set()

    for example in examples:
        messages = example.get(
            "messages"
        )

        if (
            not isinstance(
                messages,
                list,
            )
            or len(messages) < 2
        ):
            failures += 1
            continue

        expected_role = "user"

        for message in messages:
            if (
                not isinstance(
                    message,
                    dict,
                )
                or message.get(
                    "role"
                )
                != expected_role
                or not isinstance(
                    message.get(
                        "content"
                    ),
                    str,
                )
                or not message[
                    "content"
                ].strip()
            ):
                failures += 1
                break

            expected_role = (
                "assistant"
                if expected_role
                == "user"
                else "user"
            )

        if (
            example.get("source")
            != "definitions.py"
        ):
            failures += 1

        rule_id = example.get(
            "rule_id"
        )

        if isinstance(
            rule_id,
            str,
        ):
            rule_ids.add(
                rule_id
            )

    return {
        "examples": len(
            examples
        ),
        "validation_failures": (
            failures
        ),
        "rules_present": sorted(
            rule_ids
        ),
    }


def write_jsonl(
    path: Path,
    examples: list[
        dict[str, Any]
    ],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for example in examples:
            handle.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_report(
    path: Path,
    stats: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        "KirkGPT SFT dataset report",
        "=" * 40,
        (
            "Source: "
            "llm_followups.rules."
            "definitions.DEFAULT_RULES"
        ),
        (
            f"Examples: "
            f"{stats['examples']}"
        ),
        (
            "Validation failures: "
            f"{stats['validation_failures']}"
        ),
        (
            "Rules present: "
            + ", ".join(
                stats[
                    "rules_present"
                ]
            )
        ),
    ]

    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate KirkGPT SFT data "
            "directly from definitions.py."
        )
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "data/sft_followups.jsonl"
        ),
    )

    parser.add_argument(
        "--report-out",
        type=Path,
        default=Path(
            "data/sft_followups_report.txt"
        ),
    )

    parser.add_argument(
        "--n",
        type=int,
        default=2000,
    )

    # Retained for compatibility
    # with existing commands.
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help=(
            "Deprecated. Rule-derived "
            "datasets use each rule's "
            "actual number of steps."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
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

    if args.n < 1:
        raise SystemExit(
            "--n must be at least 1"
        )

    if args.out.exists():
        raise SystemExit(
            "Refusing to overwrite "
            f"existing dataset: "
            f"{args.out}"
        )

    examples = generate_dataset(
        n=args.n,
        seed=args.seed,
        max_new_tokens=(
            args.max_new_tokens
        ),
        temperature=(
            args.temperature
        ),
        top_p=args.top_p,
    )

    stats = validate_examples(
        examples
    )

    if (
        stats[
            "validation_failures"
        ]
        != 0
    ):
        raise SystemExit(
            "Generated dataset has "
            f"{stats['validation_failures']} "
            "validation failures."
        )

    write_jsonl(
        args.out,
        examples,
    )

    write_report(
        args.report_out,
        stats,
    )

    print(
        "Source: "
        "llm_followups.rules."
        "definitions.DEFAULT_RULES"
    )

    print(
        f"Rules loaded: "
        f"{len(DEFAULT_RULES)}"
    )

    print(
        "Rule IDs: "
        + ", ".join(
            rule.id
            for rule
            in DEFAULT_RULES
        )
    )

    print(
        f"Examples: "
        f"{len(examples)}"
    )

    print(
        f"Wrote: {args.out}"
    )

    print(
        f"Report: "
        f"{args.report_out}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
