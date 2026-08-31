from __future__ import annotations

import argparse
import asyncio
import tempfile
from pathlib import Path

from dataset_split import (
    read_jsonl,
    split_rows,
    write_jsonl,
)

from llm_followups.eval.batch_eval import (
    run_batch_evaluation,
)


DEFAULT_DATASET = Path(
    "data/sft_followups.jsonl"
)

DEFAULT_MODEL = Path(
    "outputs/followups"
)

DEFAULT_OUTPUT = Path(
    "eval_results/followups"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the trained model "
            "against the deterministic "
            "held-out portion of the "
            "canonical SFT dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--eval-fraction",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min-questions",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--bullet-style",
        choices=[
            "dash",
            "asterisk",
            "either",
        ],
        default="dash",
    )

    return parser.parse_args()


async def run(
    args: argparse.Namespace,
) -> dict:
    rows = read_jsonl(
        args.dataset.resolve()
    )

    train_rows, eval_rows = (
        split_rows(
            rows,
            eval_fraction=(
                args.eval_fraction
            ),
            seed=args.seed,
        )
    )

    print(
        f"Canonical examples: "
        f"{len(rows)}"
    )

    print(
        f"Training partition: "
        f"{len(train_rows)}"
    )

    print(
        f"Held-out evaluation partition: "
        f"{len(eval_rows)}"
    )

    print(
        f"Split seed: {args.seed}"
    )

    # Evaluation split is temporary.
    with tempfile.TemporaryDirectory(
        prefix="ec_pro_eval_"
    ) as temp_dir:
        eval_path = (
            Path(temp_dir)
            / "eval.jsonl"
        )

        write_jsonl(
            eval_path,
            eval_rows,
        )

        return await run_batch_evaluation(
            data_path=eval_path,
            output_dir=(
                args.output_dir
            ),
            model_path=(
                args.model_path
            ),
            limit=args.limit,
            min_questions=(
                args.min_questions
            ),
            bullet_style=(
                args.bullet_style
            ),
        )


def main() -> None:
    args = parse_args()

    results = asyncio.run(
        run(args)
    )

    print()
    print(
        "Batch evaluation completed."
    )

    print(
        "CSV: "
        f"{results['csv_path']}"
    )

    print(
        "JSON: "
        f"{results['json_path']}"
    )

    print(
        "Summary: "
        f"{results['summary']}"
    )


if __name__ == "__main__":
    main()