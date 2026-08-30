from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import json
import logging
import sys
import time

@dataclass(frozen=True)
class LogConfig:
    level: str = "INFO"
    json: bool = False
    log_file: Path | None = None
    name: str = "llm_followups"

def _level_to_int(level: str) -> int:
    """
    Convert log level string to logging module integer constant.
    
    Args:
        level: Log level string ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    
    Returns:
        Corresponding logging module integer constant.
    """
    s = (level or "INFO").upper().strip()
    mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARN": logging.WARNING,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return mapping.get(s, logging.INFO)

def _json_safe(value: Any) -> Any:
    """
    Convert value to JSON-serializable format.
    
    Handles special types (Path, Exception, set, tuple) and validates
    JSON serialization.
    
    Args:
        value: Value to convert.
    
    Returns:
        JSON-safe version of value.
    """
    if value is None:
        return None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, BaseException):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)

def setup_logging(cfg: LogConfig) -> logging.Logger:
    """
    Configure and return a logger instance.
    
    Sets up logging with console handler and optional file handler,
    supporting both plain text and JSON-formatted output.
    
    Args:
        cfg: LogConfig with logging configuration.
    
    Returns:
        Configured Logger instance.
    """
    level_int = _level_to_int(cfg.level)

    logger = logging.getLogger(cfg.name)
    logger.setLevel(level_int)

    # already configured
    if logger.handlers:
        return logger

    logger.propagate = False
    setattr(logger, "_llm_json", cfg.json)

    if not cfg.json:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    else:
        fmt = "%(message)s"

    formatter = logging.Formatter(fmt)

    h = logging.StreamHandler(sys.stderr)
    h.setLevel(level_int)
    h.setFormatter(formatter)
    logger.addHandler(h)

    if cfg.log_file is not None:
        cfg.log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(cfg.log_file, encoding="utf-8")
        fh.setLevel(level_int)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def log_event(logger: logging.Logger, event: str, *, fields: Mapping[str, Any] | None = None, level: str = "INFO") -> None:
    """
    Log a structured event with optional fields.
    
    Args:
        logger: Logger instance to use.
        event: Event name/identifier.
        fields: Optional dict of event fields to include.
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
    """
    level_int = _level_to_int(level)

    payload: dict[str, Any] = {"event": event, "ts": time.time()}
    if fields:
        for k, v in fields.items():
            payload[k] = _json_safe(v)

    use_json = bool(getattr(logger, "_llm_json", False))
    if use_json:
        msg = json.dumps(payload, ensure_ascii=False)
    else:
        # plain text: event=... k=v k=v
        parts = [f"event={event}"]
        for k, v in payload.items():
            if k in ("event",):
                continue
            parts.append(f"{k}={v}")
        msg = " ".join(parts)

    logger.log(level_int, msg)

class TrainingLogger:
    def __init__(self, logger: logging.Logger) -> None:
        """
        Initialize TrainingLogger wrapper.
        
        Args:
            logger: Logger instance to wrap and use for training events.
        """
        self._logger = logger

    def on_dataset_summary(self, summary: Mapping[str, Any]) -> None:
        """
        Log dataset summary statistics.
        
        Args:
            summary: Dictionary of dataset statistics.
        """
        log_event(self._logger, "dataset_summary", fields=dict(summary), level="INFO")

    def on_train_start(self, cfg: Mapping[str, Any]) -> None:
        """
        Log training start event with configuration.
        
        Args:
            cfg: Training configuration dictionary.
        """
        cfg_dict = dict(cfg)
        log_event(self._logger, "train_start", fields=cfg_dict, level="INFO")

    def on_step(self, *, step: int, loss: float | None = None, lr: float | None = None) -> None:
        """
        Log training step with metrics.
        
        Args:
            step: Training step number.
            loss: Optional loss value.
            lr: Optional learning rate.
        """
        fields: dict[str, Any] = {"step": step}
        if loss is not None:
            fields["loss"] = loss
        if lr is not None:
            fields["lr"] = lr
        log_event(self._logger, "train_step", fields=fields, level="DEBUG")

    def on_train_end(self, *, output_dir: str, metrics: Mapping[str, Any] | None = None) -> None:
        """
        Log training end event with final metrics.
        
        Args:
            output_dir: Path to output directory with trained model.
            metrics: Optional dictionary of final training metrics.
        """
        fields: dict[str, Any] = {"output_dir": _json_safe(output_dir)}
        if metrics is not None:
            for k, v in metrics.items():
                fields[k] = _json_safe(v)
        log_event(self._logger, "train_end", fields=fields, level="INFO")