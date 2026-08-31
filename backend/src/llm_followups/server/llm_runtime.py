from __future__ import annotations

import asyncio
import logging
import os
import time

from llm_followups.server.runtime_types import (
    GenerationRequest,
    GenerationResult,
)

from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    set_seed,
)

from llm_followups.prompting import render_chat_prompt
from llm_followups.server.schemas import ChatMessage
from llm_followups.tuning.validate import (
    fallback_followups,
    try_repair_to_followups,
    validate_followup_list,
)
from llm_followups.utils.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _take_bullets(text: str, k: int) -> str | None:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") or stripped.startswith("* "):
            bullets.append(stripped)
        if len(bullets) >= k:
            break
    if bullets:
        return "\n".join(bullets[:k]).strip()
    return None


class LLMRuntime:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[PreTrainedModel] = None
        self._loaded = False
        self._lock = asyncio.Lock()
        self._device = "cpu"
        self._adapter_loaded = False
        self._load_path: str | None = None

    async def load(self) -> None:
        if self._loaded:
            return

        requested = getattr(self._settings, "device", "cpu")
        if requested == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested == "cuda":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved == "cpu":
                logger.warning("Requested CUDA but CUDA is not available; falling back to CPU")
        elif requested == "cpu":
            resolved = "cpu"
        else:
            resolved = "cpu"
            logger.warning("Unknown device setting '%s', defaulting to cpu", requested)

        self._device = resolved
        torch_device = torch.device(resolved)

        model_path = os.getenv("MODEL_PATH")
        load_path: str | Path
        if model_path and Path(model_path).exists():
            load_path = model_path
        else:
            load_path = self._settings.model_name

        self._load_path = str(load_path)
        logger.info("Loading model from: %s", self._load_path)

        tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
        added_tokens = False
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                added_tokens = True

        assert tokenizer.pad_token_id is not None, "pad_token_id must be set"
        assert tokenizer.eos_token_id is not None, "eos_token_id must be set"

        dtype = None
        if resolved == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model = AutoModelForCausalLM.from_pretrained(load_path, dtype=dtype)
        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))

        model.config.pad_token_id = tokenizer.pad_token_id
        if model.generation_config is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id

        logger.info("Tokenizer vocab size: %d", len(tokenizer))
        logger.info("Model vocab size: %d", model.config.vocab_size)
        logger.info("Pad token id: %s", tokenizer.pad_token_id)
        logger.info("EOS token id: %s", tokenizer.eos_token_id)
        logger.info("Runtime device: %s", self._device)

        model.to(torch_device)
        model.eval()

        if self._settings.adapter_path is not None:
            logger.warning(
                "Adapter path provided but PEFT support not yet integrated; adapter not loaded"
            )
        self._adapter_loaded = False

        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    def is_loaded(self) -> bool:
        return self._loaded

    def device_str(self) -> str:
        return self._device

    def model_name(self) -> str:
        return self._load_path or self._settings.model_name

    def adapter_loaded(self) -> bool:
        return self._adapter_loaded

    def build_prompt(self, messages: Sequence[ChatMessage]) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not initialized")

        return render_chat_prompt(
            self._tokenizer,
            messages,
            min_questions=self._settings.min_questions,
            bullet_style=self._settings.bullet_style,
            add_generation_prompt=True,
        )

    def enforce_followup_format(
        self,
        text: str,
        *,
        prompt_summary: str | None = None,
    ) -> tuple[str, bool, bool]:
        if not self._settings.enforce_format:
            return text.strip(), False, False

        k = self._settings.min_questions
        trimmed = _take_bullets(text, k)
        if trimmed is not None:
            text = trimmed

        validation = validate_followup_list(
            text,
            min_questions=k,
            bullet_style=self._settings.bullet_style,
            require_question_mark=True,
            forbid_extra_text=True,
        )
        if validation.ok:
            return validation.normalized_text or text.strip(), False, False

        repair_style = (
            self._settings.bullet_style
            if self._settings.bullet_style in ("dash", "asterisk")
            else "dash"
        )
        repaired = try_repair_to_followups(
            text,
            min_questions=k,
            bullet_style=repair_style,
        )
        if repaired is not None:
            return repaired, True, False

        fallback = fallback_followups(
            prompt_summary=prompt_summary,
            min_questions=k,
            bullet_style=repair_style,
        )
        return fallback, False, True

    def make_request(
        self,
        messages: Sequence[ChatMessage],
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
    ) -> GenerationRequest:
        eff_max_new_tokens = (
            max_new_tokens if max_new_tokens is not None else self._settings.max_new_tokens
        )
        eff_temperature = (
            temperature if temperature is not None else self._settings.temperature
        )
        eff_top_p = top_p if top_p is not None else self._settings.top_p
        eff_seed = seed if seed is not None else self._settings.seed

        if not isinstance(eff_max_new_tokens, int) or eff_max_new_tokens < 1:
            raise ValueError("max_new_tokens must be an integer >= 1")
        if eff_max_new_tokens > 4096:
            raise ValueError("max_new_tokens too large")
        if not isinstance(eff_temperature, (int, float)) or not 0.0 <= eff_temperature <= 2.0:
            raise ValueError("temperature must be a number between 0.0 and 2.0")
        if not isinstance(eff_top_p, (int, float)) or not 0.0 < eff_top_p <= 1.0:
            raise ValueError("top_p must be in range (0.0, 1.0]")
        if eff_seed is not None and not isinstance(eff_seed, int):
            raise ValueError("seed must be an integer or None")

        return GenerationRequest(
            messages=messages,
            max_new_tokens=eff_max_new_tokens,
            temperature=float(eff_temperature),
            top_p=float(eff_top_p),
            seed=eff_seed,
        )

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        if not self.is_loaded():
            await self.load()

        start = time.perf_counter()

        async with self._lock:
            if req.seed is not None:
                set_seed(req.seed)

            if self._tokenizer is None or self._model is None:
                raise RuntimeError("Tokenizer or model not initialized")

            prompt = self.build_prompt(req.messages)
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
            )
            device = torch.device(self._device)
            inputs = {name: value.to(device) for name, value in inputs.items()}
            input_len = inputs["input_ids"].shape[1]

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": req.max_new_tokens,
                "min_new_tokens": min(45, req.max_new_tokens),
                "do_sample": False,
                "repetition_penalty": 1.10,
                "no_repeat_ngram_size": 3,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            with torch.inference_mode():
                generated = self._model.generate(**inputs, **generation_kwargs)

        generated_ids = generated[:, input_len:]
        raw_text = self._tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        ).strip()
        logger.info("RAW MODEL OUTPUT: %r", raw_text)

        prompt_summary = None
        for message in reversed(req.messages):
            if message.role == "user":
                prompt_summary = message.content[:100]
                break

        final_text, used_repair, used_fallback = self.enforce_followup_format(
            raw_text,
            prompt_summary=prompt_summary,
        )
        logger.info(
            "FORMAT FLAGS: used_repair=%s used_fallback=%s",
            used_repair,
            used_fallback,
        )
        logger.info("FINAL OUTPUT: %r", final_text)

        latency_ms = int((time.perf_counter() - start) * 1000)
        return GenerationResult(
            raw_text=raw_text,
            final_text=final_text,
            used_fallback=used_fallback,
            used_repair=used_repair,
            latency_ms=latency_ms,
        )
