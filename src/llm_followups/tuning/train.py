from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

from llm_followups.tuning.dataset import DatasetConfig, build_dataset, summarize_dataset
from llm_followups.utils.log import LogConfig, TrainingLogger, setup_logging


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    output_dir: Path
    dataset: DatasetConfig
    device: Literal["cpu", "cuda", "auto"] = "auto"
    seed: int | None = 42
    batch_size: int = 2
    grad_accum_steps: int = 1
    lr: float = 2e-5
    epochs: float = 1.0
    max_steps: int | None = None
    save_steps: int = 500
    logging_steps: int = 50
    fp16: bool = False
    bf16: bool = False
    eval_split: str | None = None
    eval_steps: int | None = None


def resolve_device(device: Literal["cpu", "cuda", "auto"]) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        return "cuda"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unsupported device: {device}")


def load_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    fp16: bool = False,
    bf16: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    added_pad_token = False
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_pad_token = True

    dtype = None
    if device == "cuda":
        if bf16:
            dtype = torch.bfloat16
        elif fp16:
            dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
    )

    if added_pad_token:
        model.resize_token_embeddings(len(tokenizer))

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

    model.to(torch.device(device))
    model.train()
    return tokenizer, model


class AssistantOnlyDataCollator:
    """Pad tokenized examples without overwriting pre-computed SFT labels."""

    def __init__(self, tokenizer) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer.pad_token_id must be set")
        self._pad_token_id = int(tokenizer.pad_token_id)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        labels: list[list[int]] = []

        for feature in features:
            ids = list(feature["input_ids"])
            mask = list(feature.get("attention_mask", [1] * len(ids)))
            target = list(feature["labels"])
            pad_len = max_len - len(ids)

            input_ids.append(ids + [self._pad_token_id] * pad_len)
            attention_mask.append(mask + [0] * pad_len)
            labels.append(target + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def build_trainer(cfg: TrainConfig, model, tokenizer, train_ds) -> Trainer:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    resolved = resolve_device(cfg.device)
    fp16 = cfg.fp16 if resolved == "cuda" else False
    bf16 = cfg.bf16 if resolved == "cuda" else False

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        learning_rate=cfg.lr,
        max_steps=cfg.max_steps if cfg.max_steps is not None else -1,
        num_train_epochs=cfg.epochs if cfg.max_steps is None else 0,
        save_steps=cfg.save_steps,
        logging_steps=cfg.logging_steps,
        report_to=[],
        fp16=fp16,
        bf16=bf16,
        remove_unused_columns=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        data_collator=AssistantOnlyDataCollator(tokenizer),
    )


def train(cfg: TrainConfig) -> Path:
    validate_train_config(cfg)

    log = setup_logging(LogConfig())
    tlog = TrainingLogger(log)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = resolve_device(cfg.device)
    tokenizer, model = load_model_and_tokenizer(
        cfg.model_name,
        device=device,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
    )

    train_ds = build_dataset(cfg.dataset, tokenizer)
    tlog.on_dataset_summary(summarize_dataset(train_ds))

    trainer = build_trainer(cfg, model, tokenizer, train_ds)

    tlog.on_train_start(asdict(cfg))
    trainer.train()

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    try:
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else None
    except Exception:
        metrics = None

    tlog.on_train_end(
        output_dir=str(cfg.output_dir),
        metrics=metrics if isinstance(metrics, dict) else None,
    )
    return cfg.output_dir


def validate_train_config(cfg: TrainConfig) -> None:
    resolved = resolve_device(cfg.device)
    if cfg.batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if cfg.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")
    if cfg.lr <= 0:
        raise ValueError("learning rate must be > 0")
    if cfg.max_steps is not None and cfg.max_steps <= 0:
        raise ValueError("max_steps must be > 0 if set")
    if cfg.max_steps is None and cfg.epochs <= 0:
        raise ValueError("epochs must be > 0 if max_steps is not set")
    if cfg.save_steps < 1:
        raise ValueError("save_steps must be >= 1")
    if cfg.logging_steps < 1:
        raise ValueError("logging_steps must be >= 1")
    if cfg.fp16 and cfg.bf16:
        raise ValueError("fp16 and bf16 cannot both be enabled")
    if cfg.fp16 and resolved != "cuda":
        raise ValueError("fp16 requires CUDA device")
    if cfg.bf16 and resolved != "cuda":
        raise ValueError("bf16 requires CUDA device")
    if cfg.bf16 and resolved == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bf16 was requested but this CUDA device does not support bf16")


def main(argv: list[str] | None = None) -> int:
    del argv
    print("This module exposes a `train(cfg)` function. Build a TrainConfig and call it.")
    return 0
