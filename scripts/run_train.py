from __future__ import annotations

from pathlib import Path

from llm_followups.tuning.dataset import DatasetConfig
from llm_followups.tuning.train import TrainConfig, train


def main() -> int:
    # POINT THIS at your JSONL file
    data_path = Path("data/sft_followups.jsonl").resolve()
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}")

    ds_cfg = DatasetConfig(
        format="chat_messages",
        split="train",
        data_files=str(data_path),
        shuffle=True,
        seed=42,
        max_length=512,
    )

    cfg = TrainConfig(
        # IMPORTANT: use a model you actually have access to locally or can download
        # Examples: "gpt2", "distilgpt2"
        model_name="distilgpt2",
        output_dir=Path("outputs/run1"),
        dataset=ds_cfg,
        device="auto",
        batch_size=2,
        grad_accum_steps=1,
        lr=2e-5,
        epochs=1.0,
        save_steps=200,
        logging_steps=10,
    )

    out = train(cfg)
    print(f"Saved model to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())