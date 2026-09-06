from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Mapping, Optional, Sequence

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from llm_followups.prompting import BulletStyle, apply_chat_template_ids


@dataclass(frozen=True)
class DatasetConfig:
    source: str = "tatsu-lab/alpaca"
    split: str = "train"
    text_field: str = "text"
    max_length: int = 512
    shuffle: bool = True
    seed: int | None = None
    num_proc: int | None = None
    format: Literal["raw_text", "chat_messages"] = "raw_text"
    data_files: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None
    min_questions: int = 3
    bullet_style: BulletStyle = "dash"
    assistant_only_loss: bool = True


def load_raw_dataset(cfg: DatasetConfig) -> Dataset:
    if cfg.data_files:
        ds = load_dataset("json", data_files=cfg.data_files, split=cfg.split)
        if not isinstance(ds, Dataset):
            raise TypeError("Expected a Dataset when loading with 'split' and local data_files")
    else:
        if not cfg.source:
            raise ValueError("Must set either cfg.source or cfg.data_files")
        loaded = load_dataset(cfg.source)
        if isinstance(loaded, DatasetDict):
            if cfg.split not in loaded.keys():
                raise ValueError(
                    f"Requested split '{cfg.split}' not in available splits: {list(loaded.keys())}"
                )
            ds = loaded[cfg.split]
        elif isinstance(loaded, Dataset):
            ds = loaded
        else:
            raise TypeError("Unexpected return from load_dataset: expected Dataset or DatasetDict")

    if cfg.shuffle:
        ds = ds.shuffle(seed=cfg.seed if cfg.seed is not None else 42)

    return ds


def prepare_training_text(
    ds: Dataset,
    *,
    text_field: str,
    format: Literal["raw_text", "chat_messages"],
) -> Dataset:
    """Prepare legacy/raw-text training data.

    SFT v2 chat training is tokenized directly from `messages` in build_dataset()
    so role boundaries remain exact and assistant-only labels can be constructed.
    """
    if format not in ("raw_text", "chat_messages"):
        raise ValueError("format must be one of 'raw_text' or 'chat_messages'")

    if format == "chat_messages":
        if "messages" not in ds.column_names:
            raise ValueError("Expected 'messages' column for chat_messages format")
        return ds

    if text_field in ds.column_names:
        use_field = text_field

        def map_fn(batch: Dict[str, List]) -> Dict[str, List[str]]:
            return {
                "text": ["" if value is None else str(value).strip() for value in batch[use_field]]
            }

    else:
        required = ("instruction", "output")
        if not all(field in ds.column_names for field in required):
            raise ValueError("Missing text_field and no fallback fields found (instruction/output)")

        def map_fn(batch: Dict[str, List]) -> Dict[str, List[str]]:
            texts: list[str] = []
            instrs = batch.get("instruction", [])
            inputs = batch.get("input", [None] * len(instrs))
            outputs = batch.get("output", [None] * len(instrs))
            for instruction, inp, out in zip(instrs, inputs, outputs):
                instruction = "" if instruction is None else str(instruction).strip()
                inp = "" if inp is None else str(inp).strip()
                out = "" if out is None else str(out).strip()
                if inp:
                    text = (
                        f"### Instruction:\n{instruction}\n\n"
                        f"### Input:\n{inp}\n\n"
                        f"### Response:\n{out}"
                    )
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{out}"
                texts.append(text.strip())
            return {"text": texts}

    ds = ds.map(map_fn, batched=True)
    return ds.filter(lambda example: bool(example.get("text") and str(example["text"]).strip()))


def _normalise_messages(raw_messages: object) -> list[dict[str, str]]:
    if not isinstance(raw_messages, list):
        return []

    result: list[dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            result.append({"role": role, "content": content})
    return result


def _split_last_assistant_turn(
    raw_messages: object,
) -> tuple[list[dict[str, str]], dict[str, str]] | None:
    messages = _normalise_messages(raw_messages)
    if not messages:
        return None

    assistant_index: int | None = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "assistant":
            assistant_index = index
            break

    if assistant_index is None:
        return None

    context = messages[:assistant_index]
    answer = messages[assistant_index]
    if not any(message["role"] == "user" for message in context):
        return None

    return context, answer


def _tokenize_chat_example(
    example: dict,
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    min_questions: int,
    bullet_style: BulletStyle,
    assistant_only_loss: bool,
) -> dict[str, object]:
    messages = _normalise_messages(
        example.get("messages")
    )

    if not messages:
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "_valid_sft": False,
        }

    if not any(
        message["role"] == "user"
        for message in messages
    ):
        return {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "_valid_sft": False,
        }

    full_ids = apply_chat_template_ids(
        tokenizer,
        messages,
        min_questions=min_questions,
        bullet_style=bullet_style,
        add_generation_prompt=False,
    )

    input_ids = full_ids[:max_length]

    attention_mask = [
        1
    ] * len(input_ids)

    if not assistant_only_loss:
        labels = list(input_ids)

    else:
        labels = [
            -100
        ] * len(input_ids)

        for index, message in enumerate(
            messages
        ):
            if (
                message["role"]
                != "assistant"
            ):
                continue

            context = messages[:index]

            if not any(
                item["role"] == "user"
                for item in context
            ):
                continue

            prompt_ids = (
                apply_chat_template_ids(
                    tokenizer,
                    context,
                    min_questions=(
                        min_questions
                    ),
                    bullet_style=(
                        bullet_style
                    ),
                    add_generation_prompt=True,
                )
            )

            answer_prefix_ids = (
                apply_chat_template_ids(
                    tokenizer,
                    messages[: index + 1],
                    min_questions=(
                        min_questions
                    ),
                    bullet_style=(
                        bullet_style
                    ),
                    add_generation_prompt=False,
                )
            )

            start_index = min(
                len(prompt_ids),
                len(input_ids),
            )

            end_index = min(
                len(answer_prefix_ids),
                len(input_ids),
            )

            if (
                end_index
                <= start_index
            ):
                continue

            labels[
                start_index:end_index
            ] = input_ids[
                start_index:end_index
            ]

    has_supervised_token = any(
        label != -100
        for label in labels
    )

    return {
        "input_ids": input_ids,
        "attention_mask":
            attention_mask,
        "labels": labels,
        "_valid_sft": bool(
            input_ids
            and has_supervised_token
        ),
    }


def tokenize_dataset(
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    remove_columns: Optional[Sequence[str]] = None,
    num_proc: Optional[int] = None,
) -> Dataset:
    """Tokenize raw text and create normal causal-LM labels."""
    if "text" not in ds.column_names:
        raise ValueError("Dataset must contain a 'text' column to tokenize")

    def tokenize_batch(batch: Dict[str, List]) -> Dict[str, List]:
        encoded = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
        return encoded

    tokenized = ds.map(tokenize_batch, batched=True, num_proc=num_proc)
    if remove_columns:
        tokenized = tokenized.remove_columns(
            [column for column in remove_columns if column in tokenized.column_names]
        )
    return tokenized


def build_dataset(cfg: DatasetConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    ds = load_raw_dataset(cfg)

    if cfg.format == "chat_messages":
        if "messages" not in ds.column_names:
            raise ValueError("Expected 'messages' column for chat_messages format")

        original_columns = list(ds.column_names)
        ds = ds.map(
            lambda example: _tokenize_chat_example(
                example,
                tokenizer=tokenizer,
                max_length=cfg.max_length,
                min_questions=cfg.min_questions,
                bullet_style=cfg.bullet_style,
                assistant_only_loss=cfg.assistant_only_loss,
            ),
            batched=False,
            num_proc=cfg.num_proc,
        )
        ds = ds.filter(lambda example: bool(example.get("_valid_sft")))
        removable = [
            column
            for column in original_columns + ["_valid_sft"]
            if column in ds.column_names
            and column not in {"input_ids", "attention_mask", "labels"}
        ]
        if removable:
            ds = ds.remove_columns(removable)
        return ds

    ds = prepare_training_text(ds, text_field=cfg.text_field, format=cfg.format)
    remove_columns = [column for column in ds.column_names if column != "text"]
    ds = tokenize_dataset(
        ds,
        tokenizer,
        max_length=cfg.max_length,
        remove_columns=remove_columns,
        num_proc=cfg.num_proc,
    )
    if "text" in ds.column_names:
        ds = ds.remove_columns(["text"])
    return ds


def summarize_dataset(ds: Dataset) -> Dict[str, int | float]:
    summary: Dict[str, int | float] = {"rows": len(ds)}

    if "input_ids" in ds.column_names:
        lengths = [len(ids) for ids in ds["input_ids"] if ids is not None]
        if lengths:
            summary["min_len"] = min(lengths)
            summary["mean_len"] = sum(lengths) / len(lengths)
            summary["max_len"] = max(lengths)

    if "labels" in ds.column_names:
        supervised_counts = [
            sum(1 for token in labels if token != -100)
            for labels in ds["labels"]
            if labels is not None
        ]
        if supervised_counts:
            summary["min_supervised_tokens"] = min(supervised_counts)
            summary["mean_supervised_tokens"] = (
                sum(supervised_counts) / len(supervised_counts)
            )
            summary["max_supervised_tokens"] = max(supervised_counts)

    return summary
