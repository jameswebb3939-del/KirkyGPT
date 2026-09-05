from __future__ import annotations

import torch

import argparse
import tempfile
from pathlib import Path

from dataset_split import (
    read_jsonl,
    split_rows,
    write_jsonl,
)

from llm_followups.tuning.dataset import (
    DatasetConfig,
)
from llm_followups.tuning.train import (
    TrainConfig,
    resolve_device,
    train,
)


DEFAULT_DATASET = Path(
    "data/sft_followups.jsonl"
)

DEFAULT_OUTPUT = Path(
    "outputs/followups"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the follow-up question "
            "model using the canonical "
            "SFT dataset."
        )
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
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
        "--device",
        choices=[
            "cpu",
            "cuda",
            "auto",
        ],
        default="auto",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    dataset_path = (
        args.dataset.resolve()
    )

    output_dir = (
        args.output_dir.resolve()
    )

    rows = read_jsonl(
        dataset_path
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
        f"Canonical dataset: "
        f"{dataset_path}"
    )

    print(
        f"Total examples: "
        f"{len(rows)}"
    )

    print(
        f"Training examples: "
        f"{len(train_rows)}"
    )

    print(
        f"Held-out eval examples: "
        f"{len(eval_rows)}"
    )

    print(
        f"Split seed: {args.seed}"
    )

    # Avoid mixing fresh training with
    # previous checkpoints.
    if (
        output_dir.exists()
        and any(
            output_dir.iterdir()
        )
    ):
        raise SystemExit(
            "Output directory is not empty: "
            f"{output_dir}\n"
            "Delete the old training output "
            "before starting a fresh run."
        )

    # The train split exists only for the
    # lifetime of this script.
    #
    # data/ continues to contain ONE
    # canonical JSONL dataset.
    with tempfile.TemporaryDirectory(
        prefix="kirk_gpt_train_"
    ) as temp_dir:
        temporary_train_path = (
            Path(temp_dir)
            / "train.jsonl"
        )

        write_jsonl(
            temporary_train_path,
            train_rows,
        )

        ds_cfg = DatasetConfig(
            format="chat_messages",
            data_files=str(
                temporary_train_path
            ),
            split="train",
            shuffle=True,
            seed=args.seed,
            max_length=512,
            min_questions=3,
            bullet_style="dash",
            assistant_only_loss=True,
        )

        resolved_device = resolve_device(
            args.device
        )

        use_bf16 = (
            resolved_device == "cuda"
            and torch.cuda.is_bf16_supported()
        )

        use_fp16 = (
            resolved_device == "cuda"
            and not use_bf16
        )

        cfg = TrainConfig(
            model_name=(
                "meta-llama/"
                "Llama-3.2-1B-Instruct"
            ),
            output_dir=output_dir,
            dataset=ds_cfg,
            lr=5e-6,
            batch_size=1,
            grad_accum_steps=1,
            epochs=1.0,
            save_steps=500,
            logging_steps=10,
            seed=args.seed,
            device=resolved_device,
            fp16=use_fp16,
            bf16=use_bf16,
        )

        trained_path = train(cfg)

    print()
    print(
        "Saved fresh model to: "
        f"{trained_path.resolve()}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())