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
        data_files=str(Path("data") / "sft_followups.jsonl"),
        split="train",
        shuffle=True,
        seed=42,
        max_length=512,
    )

    cfg = TrainConfig(
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        output_dir=Path("outputs") / "llama1",
        dataset=ds_cfg,
        lr=2e-5,
        batch_size=1,
        grad_accum_steps=1,
        max_steps=60,
        save_steps=20,
        logging_steps=10,
    )

    out = train(cfg)
    print(f"Saved model to: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())