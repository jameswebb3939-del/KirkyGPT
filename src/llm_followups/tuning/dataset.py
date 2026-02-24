from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Sequence, Optional, Dict, List, Mapping
from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

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

def load_raw_dataset(cfg: DatasetConfig) -> Dataset:
    """
    Load a raw dataset from the configured source.
    
    Args:
        cfg: DatasetConfig with source and split information.
    
    Returns:
        Dataset loaded from the configured source and split.
    
    Raises:
        ValueError: If source is empty or split not found.
        TypeError: If load_dataset returns unexpected type.
    """
    # Load from local files if provided, otherwise load from HF hub/source
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
                raise ValueError(f"Requested split '{cfg.split}' not in available splits: {list(loaded.keys())}")
            ds = loaded[cfg.split]
        elif isinstance(loaded, Dataset):
            ds = loaded
        else:
            raise TypeError("Unexpected return from load_dataset: expected Dataset or DatasetDict")

    if cfg.shuffle:
        seed = cfg.seed if cfg.seed is not None else 42
        ds = ds.shuffle(seed=seed)

    return ds
    
def prepare_training_text(ds: Dataset, *, text_field: str, format: Literal["raw_text", "chat_messages"]) -> Dataset:
    """
    Prepare dataset for training by extracting and formatting text.
    
    Args:
        ds: Input dataset.
        text_field: Field name containing text (for raw_text format).
        format: Format type - "raw_text" or "chat_messages".
    
    Returns:
        Dataset with prepared text column.
    
    Raises:
        ValueError: If format is invalid or required fields are missing.
    """
    if format not in ("raw_text", "chat_messages"):
        raise ValueError("format must be one of 'raw_text' or 'chat_messages'")

    if format == "raw_text":
        if text_field in ds.column_names:
            use_field = text_field

            def map_fn(batch: Dict[str, List]) -> Dict[str, List[str]]:
                texts = []
                for t in batch[use_field]:
                    s = "" if t is None else str(t)
                    texts.append(s.strip())
                return {"text": texts}
        else:
            # fallback to Alpaca-style fields
            required = ("instruction", "output")
            if all(f in ds.column_names for f in required):
                def map_fn(batch: Dict[str, List]) -> Dict[str, List[str]]:
                    texts = []
                    instrs = batch.get("instruction", [])
                    inputs = batch.get("input", [None] * len(instrs))
                    outputs = batch.get("output", [None] * len(instrs))
                    for instruction, inp, out in zip(instrs, inputs, outputs):
                        instruction = "" if instruction is None else str(instruction).strip()
                        out = "" if out is None else str(out).strip()
                        inp = "" if inp is None else str(inp).strip()
                        if inp:
                            txt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
                        else:
                            txt = f"### Instruction:\n{instruction}\n\n### Response:\n{out}"
                        texts.append(txt.strip())
                    return {"text": texts}
            else:
                raise ValueError("Missing text_field and no fallback fields found (instruction/output)")

        ds = ds.map(map_fn, batched=True)
        ds = ds.filter(lambda ex: bool(ex.get("text") and str(ex["text"]).strip()))
        return ds

    # chat_messages format
    if format == "chat_messages":
        if "messages" not in ds.column_names:
            raise ValueError("Expected 'messages' column for chat_messages format")

        def map_chat(batch: Dict[str, List]) -> Dict[str, List[str]]:
            texts: List[str] = []
            msgs_batch = batch.get("messages", [])
            for msgs in msgs_batch:
                if not msgs or not isinstance(msgs, list):
                    texts.append("")
                    continue

                # Build exactly the same template as sanity_infer.py
                user_parts: List[str] = []
                assistant_parts: List[str] = []

                for m in msgs:
                    if not isinstance(m, dict):
                        continue
                    role = m.get("role", "")
                    content = m.get("content", "")
                    content = "" if content is None else str(content).strip()
                    if not content:
                        continue

                    if role == "user":
                        user_parts.append(content)
                    elif role == "assistant":
                        assistant_parts.append(content)
                    else:
                        # ignore system/other for now
                        continue

                user_text = "\n".join(user_parts).strip()
                assistant_text = "\n".join(assistant_parts).strip()

                if not user_text and not assistant_text:
                    texts.append("")
                    continue

                txt = f"### User:\n{user_text}\n\n### Assistant:\n{assistant_text}".strip()
                texts.append(txt)

            return {"text": texts}

        ds = ds.map(map_chat, batched=True)
        ds = ds.filter(lambda ex: bool(ex.get("text") and str(ex["text"]).strip()))
        return ds


def tokenize_dataset(ds: Dataset, tokenizer: PreTrainedTokenizerBase, *, max_length: int, remove_columns: Optional[Sequence[str]] = None, num_proc: Optional[int] = None) -> Dataset:
    """
    Tokenize dataset text and format for language model training.
    
    Args:
        ds: Dataset with text column.
        tokenizer: Tokenizer to use for encoding.
        max_length: Maximum sequence length (truncation applied).
        remove_columns: Columns to remove after tokenization.
        num_proc: Number of processes for parallel tokenization.
    
    Returns:
        Tokenized dataset ready for training.
    
    Raises:
        ValueError: If dataset doesn't contain 'text' column.
    """
    if "text" not in ds.column_names:
        raise ValueError("Dataset must contain a 'text' column to tokenize")

    def tokenize_batch(batch: Dict[str, List]) -> Dict[str, List]:
        texts = batch["text"]
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
        return enc

    tokenized = ds.map(tokenize_batch, batched=True, num_proc=num_proc)
    if remove_columns is not None and len(remove_columns) > 0:
        tokenized = tokenized.remove_columns([c for c in remove_columns if c in tokenized.column_names])
    return tokenized



def build_dataset(cfg: DatasetConfig, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    """
    Build a complete training dataset from raw data.
    
    Loads raw data, prepares text, and tokenizes for training.
    
    Args:
        cfg: DatasetConfig with dataset configuration.
        tokenizer: PreTrainedTokenizerBase for tokenization.
    
    Returns:
        Fully prepared and tokenized dataset.
    """
    ds = load_raw_dataset(cfg)
    ds = prepare_training_text(ds, text_field=cfg.text_field, format=cfg.format)

    # keep 'text' column (useful for debugging) and remove other original columns
    remove_columns = [col for col in ds.column_names if col != "text"]

    ds = tokenize_dataset(ds, tokenizer, max_length=cfg.max_length, remove_columns=remove_columns, num_proc=cfg.num_proc)
    return ds

def summarize_dataset(ds: Dataset) -> Dict[str, int | float]:
    """
    Generate summary statistics for a dataset.
    
    Args:
        ds: Dataset to summarize.
    
    Returns:
        Dictionary with dataset statistics (row count, sequence lengths, etc.).
    """
    summary: Dict[str, int | float] = {}
    n = len(ds)
    summary["rows"] = n

    if "text" in ds.column_names:
        texts = ds["text"]
        empty_count = sum(1 for t in texts if not (t and str(t).strip()))
        summary["empty_text_rows"] = empty_count
        summary["fraction_empty_text"] = (empty_count / n) if n > 0 else 0.0

    if "input_ids" in ds.column_names:
        ids_col = ds["input_ids"]
        lengths = [len(ids) for ids in ids_col if ids is not None]
        if lengths:
            summary["min_len"] = min(lengths)
            summary["mean_len"] = sum(lengths) / len(lengths)
            summary["max_len"] = max(lengths)

    return summary

