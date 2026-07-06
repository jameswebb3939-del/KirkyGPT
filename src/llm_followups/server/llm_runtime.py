
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence, Optional
import asyncio
import time
import logging
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed, PreTrainedTokenizerBase, PreTrainedModel

from llm_followups.utils.config import Settings
from llm_followups.tuning.validate import (
    validate_followup_list,
    try_repair_to_followups,
    fallback_followups,
)
from llm_followups.server.schemas import ChatMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _take_bullets(text: str, k: int) -> str | None:
    """
    Extract up to k bullet lines (starting with - or *) from text.
    Returns the joined bullet lines if any are found, else None.
    """
    bullets: list[str] = []
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("- ") or s.startswith("* "):
            bullets.append(s)
        if len(bullets) >= k:
            break
    if bullets:
        return "\n".join(bullets[:k]).strip()
    return None

@dataclass(frozen=True)
class GenerationRequest:
    messages: Sequence[ChatMessage]
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int | None


@dataclass(frozen=True)
class GenerationResult:
    raw_text: str
    final_text: str
    used_fallback: bool
    used_repair: bool
    latency_ms: int


class LLMRuntime:
    def __init__(self, settings: Settings) -> None:
        """
        Initialize the LLM runtime with configuration settings.
        
        Args:
            settings: Settings object containing model and generation configuration.
        """
        self._settings = settings
        self._tokenizer: Optional[PreTrainedTokenizerBase] = None
        self._model: Optional[PreTrainedModel] = None
        self._loaded: bool = False
        self._lock = asyncio.Lock()
        # resolved device ("cpu" or "cuda")
        self._device: str = "cpu"
        self._adapter_loaded: bool = False
        self._load_path: str | None = None

    async def load(self) -> None:
        """
        Load the model and tokenizer from the configured model repository.
        
        Resolves the device (CPU or CUDA), loads the model and tokenizer,
        handles pad token configuration, and attempts to load adapters if specified.
        """
        if self._loaded:
            return

        # Resolve device from settings (support 'auto', 'cpu', 'cuda', and fallback to cpu)
        settings_device = getattr(self._settings, "device", "cpu")
        resolved: str
        if settings_device == "auto":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
        elif settings_device == "cuda":
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            if resolved == "cpu":
                logger.warning("Requested CUDA but CUDA is not available; falling back to CPU")
        elif settings_device == "cpu":
            resolved = "cpu"
        else:
            # Unknown value — default to cpu but log
            resolved = "cpu"
            logger.warning("Unknown device setting '%s', defaulting to cpu", settings_device)

        self._device = resolved
        torch_device = torch.device("cuda" if resolved == "cuda" else "cpu")


        # Allow override of model path via environment variable
        model_path = os.getenv("MODEL_PATH")

        if model_path and Path(model_path).exists():
            load_path = model_path
        else:
            load_path = self._settings.model_name

        self._load_path = str(load_path)
        logger.info("Loading model from: %s", self._load_path)

        # Tokenizer: ensure pad token exists
        tokenizer = AutoTokenizer.from_pretrained(load_path, use_fast=True)
        added_tokens = False
        if tokenizer.pad_token is None:
            # fallback to eos_token if pad not set
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                # As a last resort, set pad token to '<pad>' and add to vocab if needed
                tokenizer.add_special_tokens({"pad_token": "<pad>"})
                added_tokens = True
        # After this, pad_token_id should be set
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        # Final safety
        assert tokenizer.pad_token_id is not None, "pad_token_id must be set"
        assert tokenizer.eos_token_id is not None, "eos_token_id must be set"

        # Model: CPU-first approach; avoid float16 on CPU
        model = AutoModelForCausalLM.from_pretrained(load_path)

        logger.info("Tokenizer vocab size: %d", len(tokenizer))
        logger.info("Model vocab size: %d", model.config.vocab_size)
        logger.info("Pad token id: %s", tokenizer.pad_token_id)
        logger.info("EOS token id: %s", tokenizer.eos_token_id)
        # If special tokens were added, resize embeddings
        if added_tokens:
            model.resize_token_embeddings(len(tokenizer))
        logger.info("Runtime device: %s", self._device)
        model.to(torch_device)
        model.eval()

        # Adapter support is a stub unless PEFT is integrated
        if self._settings.adapter_path is not None:
            # adapter provided but not yet supported
            logger.warning("Adapter path provided but PEFT support not yet integrated; adapter not loaded")
            self._adapter_loaded = False
        else:
            # no adapter requested
            self._adapter_loaded = False

        self._tokenizer = tokenizer
        self._model = model
        self._loaded = True

    def is_loaded(self) -> bool:
        """
        Check if the model has been loaded.
        
        Returns:
            True if model is loaded and ready for generation.
        """
        return self._loaded

    def device_str(self) -> str:
        """
        Get the resolved device string.
        
        Returns:
            Device name ("cpu" or "cuda").
        """
        return self._device

    def model_name(self) -> str:
        """
        Get the configured model name.
        
        Returns:
            Model name from settings.
        """
        return self._load_path or self._settings.model_name

    def adapter_loaded(self) -> bool:
        """
        Check if an adapter has been loaded.
        
        Returns:
            True if adapter is loaded.
        """
        return self._adapter_loaded

    def build_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """
        Build a prompt string from chat messages and system instructions.
        
        Args:
            messages: Sequence of chat messages to format.
        
        Returns:
            Formatted prompt string ready for tokenization.
        """
        # Determine bullet character preference
        bullet_pref = self._settings.bullet_style
        if bullet_pref == "asterisk":
            bullet_char = "*"
        else:
            # if 'dash' or 'either', prefer dash for stability
            bullet_char = "-"

        sys_instruction = (
            f"Return exactly {self._settings.min_questions} follow-up questions.\n"
            "Output ONLY the questions.\n"
            "The first character of the output must be '-'.\n"
            "Every line must begin with '- '.\n"
            "Every line must end with '?'.\n"
            "Do not write any introduction.\n"
            "Do not write any explanation.\n"
            "Do not write any summary.\n"
            "Do not number the questions.\n"
            "Do not leave blank lines.\n"
            "Each question must be specific to the user's request.\n"
            "Use varied wording.\n"
        )

        parts: list[str] = []
        # Wrap sys_instruction as System: ... to match training format
        parts.append(f"System: {sys_instruction}")

        # Append conversation transcript, preserving order
        for message in messages:
            role = message.role
            content = message.content.strip()
            if role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                # Unknown role — include generically
                parts.append(f"{role.capitalize()}: {content}")

        # Final assistant cue: force bullet start
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def enforce_followup_format(self, text: str, *, prompt_summary: str | None = None) -> tuple[str, bool, bool]:
        """
        Enforce format compliance for follow-up question output.
        
        Validates text as bullet-point list, attempts repair if needed,
        and falls back to generated questions if necessary.
        
        Args:
            text: Raw output text from model.
            prompt_summary: Optional summary of the original prompt for fallback generation.
        
        Returns:
            Tuple of (formatted_text, used_repair, used_fallback).
        """
        # If format enforcement is disabled, return trimmed text and no flags
        if not self._settings.enforce_format:
            return (text.strip(), False, False)

        # Truncate to first k bullets before validation (robust: use any bullets found)
        k = self._settings.min_questions
        trimmed = _take_bullets(text, k)
        if trimmed is not None:
            text = trimmed

        # Validate the raw text
        validation = validate_followup_list(
            text,
            min_questions=self._settings.min_questions,
            bullet_style=self._settings.bullet_style,
            require_question_mark=True,
            forbid_extra_text=True,
        )

        # If valid, return normalized_text (or stripped original) and flags
        if validation.ok:
            normalized = validation.normalized_text or text.strip()
            return (normalized, False, False)

        # Attempt repair: choose repair style (prefer configured style, fallback to dash)
        repair_style = (
            self._settings.bullet_style
            if self._settings.bullet_style in ("dash", "asterisk")
            else "dash"
        )

        repaired = try_repair_to_followups(text, min_questions=self._settings.min_questions, bullet_style=repair_style)
        if repaired is not None:
            return (repaired, True, False)

        # Fallback: attempt to craft guaranteed valid bullets using provided prompt_summary hint
        fallback = fallback_followups(prompt_summary=prompt_summary, min_questions=self._settings.min_questions, bullet_style=repair_style)
        return (fallback, False, True)

    def make_request(self, messages: Sequence[ChatMessage], *, max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, seed: int | None = None) -> GenerationRequest:
        """
        Create and validate a generation request from message and parameters.
        
        Args:
            messages: Chat messages to generate response for.
            max_new_tokens: Max tokens to generate (or None for default).
            temperature: Sampling temperature (or None for default).
            top_p: Nucleus sampling parameter (or None for default).
            seed: Random seed (or None for default).
        
        Returns:
            GenerationRequest with validated parameters.
        
        Raises:
            ValueError: If any parameter is out of valid range.
        """
        # Resolve effective values without mutating settings
        eff_max_new_tokens = max_new_tokens if max_new_tokens is not None else self._settings.max_new_tokens
        eff_temperature = temperature if temperature is not None else self._settings.temperature
        eff_top_p = top_p if top_p is not None else self._settings.top_p
        eff_seed = seed if seed is not None else self._settings.seed

        # Validate ranges and types
        if not isinstance(eff_max_new_tokens, int) or eff_max_new_tokens < 1:
            raise ValueError("max_new_tokens must be an integer >= 1")
        # Cap to a reasonable upper limit to avoid runaway requests
        if eff_max_new_tokens > 4096:
            raise ValueError("max_new_tokens too large")

        if not (isinstance(eff_temperature, (int, float)) and eff_temperature >= 0.0 and eff_temperature <= 2.0):
            raise ValueError("temperature must be a number between 0.0 and 2.0")

        if not (isinstance(eff_top_p, (int, float)) and 0.0 < eff_top_p <= 1.0):
            raise ValueError("top_p must be in range (0.0, 1.0]")

        # seed may be None or int
        if eff_seed is not None and not isinstance(eff_seed, int):
            raise ValueError("seed must be an integer or None")

        return GenerationRequest(messages=messages, max_new_tokens=eff_max_new_tokens, temperature=float(eff_temperature), top_p=float(eff_top_p), seed=eff_seed)

    async def generate(self, req: GenerationRequest) -> GenerationResult:
        """
        Generate follow-up questions from a generation request.
        
        Args:
            req: GenerationRequest with messages and parameters.
        
        Returns:
            GenerationResult with raw text, formatted text, and metadata.
        
        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self.is_loaded():
            await self.load()

        t0 = time.time()

        async with self._lock:
            if req.seed is not None:
                set_seed(req.seed)

            prompt = self.build_prompt(req.messages)

            if self._tokenizer is None or self._model is None:
                raise RuntimeError("Tokenizer or model not initialized")

            # Tokenize and move inputs to the configured device
            inputs = self._tokenizer(prompt, return_tensors="pt")
            device = torch.device("cuda" if self._device == "cuda" else "cpu")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Store input length before generation
            input_len = inputs.get("input_ids").shape[1]

            do_sample = False

            # Run generation with inference mode and explicit token ids
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": req.max_new_tokens,
                "min_new_tokens": min(45, req.max_new_tokens),
                "do_sample": do_sample,
                "repetition_penalty": 1.10,
                "no_repeat_ngram_size": 3,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": self._tokenizer.eos_token_id,
            }

            if do_sample:
                generation_kwargs["temperature"] = req.temperature
                generation_kwargs["top_p"] = req.top_p

            with torch.inference_mode():
                gen_output = self._model.generate(
                    **inputs,
                    **generation_kwargs,
                )

        # slice to only the newly generated tokens
        generated_ids = gen_output[:, input_len:]

        # Decode only generated portion
        raw_text = self._tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

        logger.info("RAW MODEL OUTPUT: %r", raw_text)

        # Extract prompt_summary from last user message if available
        prompt_summary: str | None = None
        for msg in reversed(req.messages):
            if msg.role == "user":
                prompt_summary = msg.content[:100]  # use first 100 chars of last user msg
                break

        final_text, used_repair, used_fallback = self.enforce_followup_format(raw_text, prompt_summary=prompt_summary)

        logger.info("FORMAT FLAGS: used_repair=%s used_fallback=%s", used_repair, used_fallback)
        logger.info("FINAL OUTPUT: %r", final_text)

        t1 = time.time()
        latency_ms = int((t1 - t0) * 1000)

        return GenerationResult(raw_text=raw_text, final_text=final_text, used_fallback=used_fallback, used_repair=used_repair, latency_ms=latency_ms)