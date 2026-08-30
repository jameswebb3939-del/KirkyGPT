from __future__ import annotations

import asyncio
from argparse import ArgumentParser
from pathlib import Path

from llm_followups.eval.batch_eval import run_batch_evaluation


def main() -> None:
    parser = ArgumentParser(
        description="Run batch evaluation of follow-up question generation."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Path to the evaluation dataset (JSONL format).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to save evaluation results.",
    )
    parser.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Optional path to the model to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of examples to evaluate.",
    )
    parser.add_argument(
        "--min_questions",
        type=int,
        default=3,
        help="Minimum number of follow-up questions required for valid output.",
    )
    parser.add_argument(
        "--bullet_style",
        type=str,
        choices=["dash", "asterisk", "either"],
        default="either",
        help="Accepted bullet style for generated follow-up questions.",
    )

    args = parser.parse_args()

    results = asyncio.run(
        run_batch_evaluation(
            data_path=args.data_path,
            output_dir=args.output_dir,
            model_path=args.model_path,
            limit=args.limit,
            min_questions=args.min_questions,
            bullet_style=args.bullet_style,
        )
    )

    print(
        "Batch evaluation completed. "
        f"Results saved to: {results['csv_path']} and {results['json_path']}"
    )
    print(f"Summary: {results['summary']}")


if __name__ == "__main__":
    main()
