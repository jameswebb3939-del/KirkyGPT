from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import csv
import os
from typing import Sequence, Any

from llm_followups.utils.config import Settings, get_settings
from llm_followups.tuning.validate import validate_followup_list
from llm_followups.server.schemas import ChatMessage
from llm_followups.server.llm_runtime import LLMRuntime

from dataclasses import asdict
from openai import OpenAI

client = OpenAI()
JUDGE_MODEL = "gpt-4o-mini"

@dataclass(frozen=True)
class EvalExample:
    id: int
    prompt: str
    expected_style: str | None
    source: str

@dataclass(frozen=True)
class EvalPrediction:
    id: int
    prompt: str
    response_text: str
    raw_text: str | None
    latency_ms: int
    used_repair: bool
    used_fallback: bool

@dataclass(frozen=True)
class FormatEvalResult:
    id: int
    format_valid: bool
    num_questions: int
    format_errors: list[str]
    normalized_text: str | None

@dataclass(frozen=True)
class BatchEvalResult:
    id: int
    prompt: str
    response_text: str
    format_valid: bool
    num_questions: int
    format_errors: list[str]
    latency_ms: int
    used_repair: bool
    used_fallback: bool
    trulens_scores: dict[str, float | str | None]

def load_eval_examples(path: Path, limit: int | None = None) -> list[EvalExample]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            # Assign robust ID: use row['id'] if present, else sequential
            row_id = row.get("id")
            if row_id is None:
                row_id = len(examples)
            # Support SFT-style: {"messages": [...]}
            if "messages" in row and isinstance(row["messages"], list):
                user_msgs = [m for m in row["messages"] if m.get("role") == "user"]
                if user_msgs:
                    last_user = user_msgs[-1]
                    prompt = last_user.get("content", "")
                else:
                    prompt = ""
                examples.append(EvalExample(
                    id=row_id,
                    prompt=prompt,
                    expected_style=row.get("expected_style"),
                    source=row.get("source", "user")  # Metadata only
                ))
            elif "user_message" in row:
                examples.append(EvalExample(
                    id=row_id,
                    prompt=row.get("user_message"),
                    expected_style=row.get("expected_style"),
                    source=row.get("source", "user")  # Metadata only
                ))
            elif "prompt" in row:
                examples.append(EvalExample(
                    id=row_id,
                    prompt=row.get("prompt"),
                    expected_style=row.get("expected_style"),
                    source=row.get("source", "user")  # Metadata only
                ))
            else:
                continue
            if limit is not None and len(examples) >= limit:
                break
    return examples

def build_runtime(settings: Settings, model_path: Path | None = None) -> LLMRuntime:
    if model_path is not None:
        os.environ["MODEL_PATH"] = str(model_path)
    runtime = LLMRuntime(settings=settings)
    return runtime

async def generate_prediction(runtime: LLMRuntime, example: EvalExample, max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> EvalPrediction:
    # Always use role="user" for evaluation
    message = ChatMessage(
        role="user",
        content=example.prompt
    )
    req = runtime.make_request(
        [message],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p
    )
    result = await runtime.generate(req)
    return EvalPrediction(
        id=example.id,
        prompt=example.prompt,
        response_text=result.final_text,
        raw_text=result.raw_text,
        latency_ms=result.latency_ms,
        used_repair=result.used_repair,
        used_fallback=result.used_fallback
    )

async def generate_predictions_batch(runtime: LLMRuntime, examples: Sequence[EvalExample], max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> list[EvalPrediction]:
    predictions = []
    for example in examples:
        try:
            pred = await generate_prediction(runtime, example, max_new_tokens, temperature, top_p)
            predictions.append(pred)
        except Exception as e:
            print(f"Parsing through example {example.id} failed: {e}")
            continue
    return predictions

def evaluate_format(prediction: EvalPrediction, min_questions: int = 3, bullet_style: str = "either") -> FormatEvalResult:
    result = validate_followup_list(
        text=prediction.response_text,
        min_questions=min_questions,
        bullet_style=bullet_style
    )
    return FormatEvalResult(
        id=prediction.id,
        format_valid=result.ok,
        num_questions=result.num_items,
        format_errors=result.errors,
        normalized_text=result.normalized_text
    )

def evaluate_format_batch(predictions: Sequence[EvalPrediction], min_questions: int = 3, bullet_style: str = "either") -> list[FormatEvalResult]:
    results = []
    for prediction in predictions:
        result = evaluate_format(prediction=prediction, 
                                 min_questions=min_questions, 
                                 bullet_style=bullet_style)
        results.append(result)
    return results


def _extract_score_from_json(text: str) -> float | None:
    try:
        data = json.loads(text)
        score = data.get("score")
        if isinstance(score, (int, float)):
            return float(score)
        return None
    except Exception as e:
        print(f"JSON score extraction failed: {e}")
        return None


def evaluate_coherence_with_openai(response: str) -> float | None:

    judge_prompt = f"""
                    You are evaluating an LLM response for coherence.

                    Score from 0 to 3:
                    0 = incoherent or unreadable
                    1 = weakly coherent
                    2 = mostly coherent
                    3 = very coherent, well-structured, and easy to follow

                    Return exactly one JSON object and no markdown.
                    {{"score": number, "reason": "short explanation"}}

                    Response to evaluate:
                    {response}
                    """

    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict LLM evaluation judge."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = result.choices[0].message.content
        if content is None:
            return None

        return _extract_score_from_json(content)

    except Exception as e:
        print(f"Coherence evaluation failed: {e}")
        return None


def evaluate_relevance_with_openai(prompt: str, response: str) -> float | None:

    judge_prompt = f"""
                    You are evaluating an LLM response for answer relevance.

                    Score from 0 to 3:
                    0 = not relevant to the prompt
                    1 = slightly relevant
                    2 = mostly relevant
                    3 = directly relevant and useful

                    Return exactly one JSON object and no markdown:
                    {{"score": number, "reason": "short explanation"}}

                    Original prompt:
                    {prompt}

                    Response to evaluate:
                    {response}
                    """

    try:
        result = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict LLM evaluation judge."},
                {"role": "user", "content": judge_prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )

        content = result.choices[0].message.content
        if content is None:
            return None

        return _extract_score_from_json(content)

    except Exception as e:
        print(f"Relevance evaluation failed: {e}")
        return None

def run_trulens_batch_eval(predictions: Sequence[EvalPrediction]) -> dict[int, dict[str, float | None]]:
    scores: dict[int, dict[str, float | None]] = {}

    for pred in predictions:
        coherence_score = evaluate_coherence_with_openai(
            response=pred.response_text,
        )
        relevance_score = evaluate_relevance_with_openai(
            prompt=pred.prompt,
            response=pred.response_text,
        )

        scores[pred.id] = {
            "coherence_score": coherence_score,
            "answer_relevance_score": relevance_score,
        }

    return scores

def merge_results(
    predictions: Sequence[EvalPrediction],
    format_results: Sequence[FormatEvalResult],
    trulens_results: dict[int, dict[str, float | None]],
) -> list[BatchEvalResult]:
    format_results_dict = {f.id: f for f in format_results}

    results = []

    for pred in predictions:
        fmt = format_results_dict.get(pred.id)
        scores = trulens_results.get(pred.id, {})

        results.append(BatchEvalResult(
            id=pred.id,
            prompt=pred.prompt,
            response_text=pred.response_text,
            format_valid=fmt.format_valid if fmt else False,
            num_questions=fmt.num_questions if fmt else 0,
            format_errors=fmt.format_errors if fmt else [],
            latency_ms=pred.latency_ms,
            used_repair=pred.used_repair,
            used_fallback=pred.used_fallback,
            trulens_scores=scores,
        ))

    return results


def write_results_csv(results: Sequence[BatchEvalResult], out_path: Path):
    with open(out_path, "w", newline='', encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "id", "prompt", "response_text", "format_valid", "num_questions", "format_errors", "latency_ms", "used_repair", "used_fallback", "trulens_scores"
        ])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "id": row.id,
                "prompt": row.prompt,
                "response_text": row.response_text,
                "format_valid": row.format_valid,
                "num_questions": row.num_questions,
                "format_errors": "; ".join(row.format_errors),
                "latency_ms": row.latency_ms,
                "used_repair": row.used_repair,
                "used_fallback": row.used_fallback,
                "trulens_scores": str(row.trulens_scores)
            })

def write_results_json(results: Sequence[BatchEvalResult], out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)

def summarise_results(results: Sequence[BatchEvalResult]) -> dict[str, Any]:
    count_example = len(results)
    format_valid_percentage = sum(r.format_valid for r in results) / count_example * 100 if count_example else 0
    average_latency = sum(r.latency_ms for r in results) / count_example if count_example else 0
    fallback_rate = sum(r.used_fallback for r in results) / count_example * 100 if count_example else 0
    repair_rate = sum(r.used_repair for r in results) / count_example * 100 if count_example else 0
    # Average trulens_score: just average over all float values in trulens_scores
    trulens_scores = []
    for r in results:
        for v in r.trulens_scores.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                trulens_scores.append(v)
    average_trulens_score = sum(trulens_scores) / len(trulens_scores) if trulens_scores else 0
    return {
        "count_example": count_example,
        "format_valid_percentage": format_valid_percentage,
        "average_latency": average_latency,
        "fallback_rate": fallback_rate,
        "repair_rate": repair_rate,
        "average_trulens_score": average_trulens_score
    }

async def run_batch_evaluation(data_path: Path, output_dir: Path, model_path: Path | None = None, limit: int | None = None, min_questions: int = 3, bullet_style: str = "either") -> dict[str, Any]:
    settings = get_settings()
    examples = load_eval_examples(path=data_path, limit=limit)
    runtime = build_runtime(settings, model_path)

    await runtime.load()

    predictions = await generate_predictions_batch(runtime, examples, settings.max_new_tokens, settings.temperature, settings.top_p)
    format_results = evaluate_format_batch(predictions)

    trulens_results = run_trulens_batch_eval(predictions)

    merged = merge_results(predictions, format_results, trulens_results)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"

    write_results_csv(merged, csv_path)
    write_results_json(merged, json_path)

    summary = summarise_results(merged)

    return {
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "summary": summary
    }