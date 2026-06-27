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

import pandas as pd
import json
from dataclasses import asdict
import mlflow
from mlflow.genai.scorers.trulens import Coherence, AnswerRelevance


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
    with open(path) as f:
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
        except Exception:
            # Robustness: skip failed example, could log here
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

def build_trulens_dataset(predictions: Sequence[EvalPrediction]) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        rows.append({
            "id": pred.id,
            "inputs": {"question": pred.prompt},
            "outputs": pred.response_text,
            "metadata": {"id": pred.id, "source": "user"}  # Could include more metadata if desired
        })
    return pd.DataFrame(rows)

def build_trulens_scorers(provider_name: str, metrics: Sequence[str]) -> list[Any]:
    model = f"{provider_name}:/gpt-4o-mini"
    scorers = []
    if "coherence" in metrics:
        scorers.append(Coherence(model=model))
    if "answer_relevance" in metrics:
        scorers.append(AnswerRelevance(model=model))
    return scorers
    

def run_trulens_batch_eval(dataset, scorers):
    if not scorers:
        return pd.DataFrame(dataset)

    results = mlflow.genai.evaluate(
        data=dataset,
        scorers=list(scorers),
    )

    return results.tables["eval_results"]

def merge_results(predictions: Sequence[EvalPrediction], format_results: Sequence[FormatEvalResult], trulens_results: pd.DataFrame) -> list[BatchEvalResult]:
    predictions_dict = {p.id: p for p in predictions}
    format_results_dict = {f.id: f for f in format_results}
    
    print(trulens_results.columns.tolist())
    print(trulens_results.head())

    trulens_results_dict = {int(r["id"]): r for _, r in trulens_results.iterrows()}
    
    print(trulens_results.columns)

    results = []
    for pid in predictions_dict:
        pred = predictions_dict[pid]
        fmt = format_results_dict.get(pid)
        tru = trulens_results_dict.get(pid, {})
        # Only include columns that look like scores (float/int, not bool, not prompt/response/id)
        score_fields = {k: v for k, v in dict(tru).items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in ("id",)}
        results.append(BatchEvalResult(
            id=pid,
            prompt=pred.prompt,
            response_text=pred.response_text,
            format_valid=fmt.format_valid if fmt else False,
            num_questions=fmt.num_questions if fmt else 0,
            format_errors=fmt.format_errors if fmt else [],
            latency_ms=pred.latency_ms,
            used_repair=pred.used_repair,
            used_fallback=pred.used_fallback,
            trulens_scores=score_fields
        ))
    return results

def write_results_csv(results: Sequence[BatchEvalResult], out_path: Path):
    with open(out_path, 'w', newline='') as file:
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
    with open(out_path, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)

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

async def run_batch_evaluation(data_path: Path, output_dir: Path, model_path: Path | None = None, limit: int | None = None, min_questions: int = 3, bullet_style: str = "either", metrics: Sequence[str] = ("coherence", "answer_relevance")) -> dict[str, Any]:
    settings = get_settings()
    examples = load_eval_examples(path=data_path, limit=limit)
    runtime = build_runtime(settings, model_path)

    await runtime.load()

    predictions = await generate_predictions_batch(runtime, examples, settings.max_new_tokens, settings.temperature, settings.top_p)
    format_results = [evaluate_format(prediction=pred, min_questions=min_questions, bullet_style=bullet_style) for pred in predictions]

    trulens_dataset = build_trulens_dataset(predictions)
    trulens_scorers = build_trulens_scorers(provider_name="openai", metrics=metrics)
    trulens_results = run_trulens_batch_eval(trulens_dataset, trulens_scorers)

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