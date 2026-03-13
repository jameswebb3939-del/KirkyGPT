from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal, Optional
import sys

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, set_seed
from transformers import DataCollatorForLanguageModeling

from llm_followups.tuning.dataset import DatasetConfig, build_dataset, summarize_dataset
from llm_followups.utils.log import setup_logging, TrainingLogger, LogConfig

@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    output_dir: Path
    dataset: DatasetConfig
    device: Literal["cpu", "cuda", "auto"] = "auto"
    seed: int | None = 42
    batch_size: int = 2
    grad_accum_steps: int = 1
    lr:float = 2e-5
    epochs: float = 1.0
    max_steps: int | None = None
    save_steps: int = 500
    logging_steps: int = 50
    fp16: bool = False
    bf16: bool = False
    eval_split: str | None = None
    eval_steps: int | None = None

def resolve_device(device: Literal["cpu", "cuda", "auto"]) -> str:
    """
    Resolve device specification to actual device string.
    
    Args:
        device: Device specification ("cpu", "cuda", or "auto").
    
    Returns:
        Resolved device string ("cpu" or "cuda").
    """
    
    if device == "cpu":
        return "cpu"
    elif device == ("cuda", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return "cpu"
    
def load_model_and_tokenizer(model_name: str, *, device: str) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Load model and tokenizer from pretrained repository.
    
    Args:
        model_name: Hugging Face model identifier.
        device: Device to load model on ("cpu" or "cuda").
    
    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # If pad_token is None, set it to eos_token or add a new pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '<pad>'})

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize token embeddings if pad_token was added
    if tokenizer.pad_token_id is not None and model.config.pad_token_id != tokenizer.pad_token_id:
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id

    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    return tokenizer, model

def build_trainer(cfg: TrainConfig, model, tokenizer, train_ds) -> Trainer:
    """
    Create a Trainer instance for fine-tuning.
    
    Args:
        cfg: TrainConfig with training configuration.
        model: PreTrainedModel to train.
        tokenizer: PreTrainedTokenizerBase for decoding.
        train_ds: Training dataset.
    
    Returns:
        Configured Trainer instance.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine resolved device from cfg.device to decide fp16/bf16
    resolved = resolve_device(cfg.device)
    fp16 = cfg.fp16 if resolved == "cuda" else False
    bf16 = cfg.bf16 if resolved == "cuda" else False

    # TrainingArguments: max_steps and num_train_epochs are separate
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
        remove_unused_columns=True
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        data_collator=collator
    )
    return trainer

def train(cfg: TrainConfig) -> Path:
    """
    Execute fine-tuning training with the specified configuration.
    
    Args:
        cfg: TrainConfig with training parameters.
    
    Returns:
        Path to the output directory with trained model.
    """
    # Validate config first
    validate_train_config(cfg)

    # Set up logging
    log = setup_logging(LogConfig())
    tlog = TrainingLogger(log)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    device = resolve_device(cfg.device)

    tokenizer, model = load_model_and_tokenizer(cfg.model_name, device=device)

    train_ds = build_dataset(cfg.dataset, tokenizer)
    stats = summarize_dataset(train_ds)
    tlog.on_dataset_summary(stats)

    trainer = build_trainer(cfg, model, tokenizer, train_ds)

    tlog.on_train_start(asdict(cfg))
    trainer.train()

    trainer.save_model(str(cfg.output_dir))
    tokenizer.save_pretrained(str(cfg.output_dir))

    # emit final metrics if available
    try:
        metrics = trainer.state.log_history[-1] if trainer.state.log_history else None
    except Exception:
        metrics = None
    tlog.on_train_end(output_dir=str(cfg.output_dir), metrics=metrics if isinstance(metrics, dict) else None)

    return cfg.output_dir

def main(argv: list[str] | None = None) -> int:
    """
    Main entry point (documentation only, use train() function directly).
    
    Args:
        argv: Command-line arguments (unused).
    
    Returns:
        Exit code.
    """
    print("This module exposes a `train(cfg)` function. Build a TrainConfig and call it.")
    return 0

def validate_train_config(cfg: TrainConfig) -> None:
    """
    Validate training configuration parameters.
    
    Args:
        cfg: TrainConfig to validate.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
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
    if cfg.output_dir is None:
        raise ValueError("output_dir must be set")
    if cfg.save_steps is None:
        raise ValueError("save_steps must be set")
    if cfg.fp16 and resolved != "cuda":
        raise ValueError("fp16 requires CUDA device")
    if cfg.bf16 and resolved != "cuda":
        raise ValueError("bf16 requires CUDA device")