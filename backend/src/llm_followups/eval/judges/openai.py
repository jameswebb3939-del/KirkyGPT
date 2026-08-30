from __future__ import annotations

import asyncio
import json
from typing import Any

from openai import OpenAI

from llm_followups.eval.core.models import JudgeResult


class OpenAIJudge:
    """OpenAI implementation of the generic Judge contract."""

    def __init__(self, *, client: OpenAI, model: str = "gpt-4o-mini") -> None:
        self._client = client
        self._model = model

    async def judge(
        self,
        *,
        instructions: str,
        input_text: str,
        output_text: str,
    ) -> JudgeResult:
        # The OpenAI client used by the existing project is synchronous, so keep
        # the evaluator API async without blocking the event loop.
        return await asyncio.to_thread(
            self._judge_sync,
            instructions=instructions,
            input_text=input_text,
            output_text=output_text,
        )

    def _judge_sync(
        self,
        *,
        instructions: str,
        input_text: str,
        output_text: str,
    ) -> JudgeResult:
        judge_prompt = f"""
Evaluate the candidate output according to the rubric below.

Rubric:
{instructions}

Original input:
{input_text}

Candidate output:
{output_text}

Return exactly one JSON object and no markdown:
{{"score": number, "reason": "short explanation"}}
""".strip()

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict, consistent LLM evaluation judge.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("Judge returned an empty response")

        data: dict[str, Any] = json.loads(content)
        score = data.get("score")

        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError(f"Judge returned an invalid score: {score!r}")

        reason = data.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = str(reason)

        return JudgeResult(
            score=float(score),
            reason=reason,
            metadata={"judge_provider": "openai", "judge_model": self._model},
        )
