from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import csv
import time
from typing import Literal, Sequence, Any

from llm_followups.utils.config import Settings, get_settings
from llm_followups.tuning.validate import ValidationResult, validate_followup_list
from llm_followups.server.schemas import ChatMessage
from llm_followups.server.llm_runtime import LLMRuntime

import pandas
from typer import prompt
import mlflow
from trulens import TruLensCoherenceScorer, TruLensAnswerRelevanceScorer


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
    with open(path) as json_file:
        j = json.load(json_file)
        examples = []
        for row in j:
            if "user_message" in row:
                examples.append(EvalExample(
                    id=row.get("id"),
                    prompt=row.get("user_message"), 
                    expected_style=row.get("expected_style"), 
                    source=row.get("source")
                ))
            elif "prompt" in row:
                examples.append(EvalExample(
                    id=row.get("id"),
                    prompt=row.get("prompt"), 
                    expected_style=row.get("expected_style"), 
                    source=row.get("source")
                ))
            else:
                continue
            if limit is not None and len(examples) >= limit:
                break
    return examples

def build_runtime(settings: Settings, model_path: Path | None = None) -> LLMRuntime:
    runtime = LLMRuntime(settings=settings)

    if model_path is not None:
        runtime = Settings(adapter_path=model_path)

    runtime.load()
    return runtime

def generate_prediction(runtime: LLMRuntime, example: EvalExample, max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> EvalPrediction:
    example = ChatMessage(
        role=example.source,
        content=example.prompt
    )

    if not max_new_tokens:
        max_new_tokens = runtime.settings.max_new_tokens
    if not temperature:
        temperature = runtime.settings.temperature
    if not top_p:
        top_p = runtime.settings.top_p
        
    generation_request = runtime.make_request()
    runtime.generate()

    return EvalPrediction(
        id=example.id,
        prompt=example.prompt,
        response_text=example.response_text,
        raw_text=example.raw_text,
        latency_ms=example.latency_ms,
        used_repair=example.used_repair,
        used_fallback=example.used_fallback
    )

def generate_predictions_batch(runtime: LLMRuntime, examples: Sequence[EvalExample], max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None) -> list[EvalPrediction]:
    predictions = []

    for example in examples:
        item = generate_prediction(runtime=runtime, 
                                   example=example, 
                                   max_new_tokens=max_new_tokens, 
                                   temperature=temperature, 
                                   top_p=top_p)
        predictions.append(item)

    return predictions

def evaluate_format(prediction: EvalPrediction, min_questions: int = 3, bullet_style: str = "either") -> FormatEvalResult:
    ok, num_items, errors, normalized_text = validate_followup_list(prediction=prediction.response_text, 
                                                                    min_questions=min_questions, 
                                                                    bullet_style=bullet_style)
    return FormatEvalResult(
        id=prediction.id,
        format_valid=ok,
        num_questions=num_items,
        format_errors=errors,
        normalized_text=normalized_text
    )

def evaluate_format_batch(predictions: Sequence[EvalPrediction], min_questions: int = 3, bullet_style: str = "either") -> list[FormatEvalResult]:
    results = []
    for prediction in predictions:
        result = evaluate_format(prediction, 
                                 min_questions=min_questions, 
                                 bullet_style=bullet_style)
        results.append(result)
    return results

def build_trulens_dataset(predictions: Sequence[EvalPrediction]) -> pandas.DataFrame:
    table = {
        "input/prompt": [p.prompt for p in predictions],
        "output/response": [p.response_text for p in predictions],
        "id": [p.id for p in predictions]
    }
    return pandas.DataFrame(table)

def build_trulens_scorers(provider_name: str, metrics: Sequence[str]) -> list[Any]:
    scorers = []
    if "coherence" in metrics:
        scorers.append(TruLensCoherenceScorer(provider=provider_name))
    if "answer_relevance" in metrics:
        scorers.append(TruLensAnswerRelevanceScorer(provider=provider_name))

    return scorers

def run_trulens_batch_eval(dataset: pandas, scorers: Sequence[Any]) -> dict[str, list[float | str | None]]:
    
    evaluate_format_batch(dataset)

    for scorer in scorers:
        s = evaluate_format(scorer)
        board.add(s)

    board = {
        "Scorer": #INCOMPLETE
    }

    return pandas.DataFrame(board)

def merge_results(predictions: Sequence[EvalPrediction], format_results: Sequence[FormatEvalResult], trulens_results: pandas):
    #Match rows by ID
    predictions_dict = {p.id: p for p in predictions}
    format_results_dict = {f.id: f for f in format_results}
    trulens_results_dict = {r["id"]: r for _, r in trulens_results.iterrows()}    

    return BatchEvalResult(
        prompt=predictions_dict[predictions.id].prompt,
        response=predictions_dict[predictions.id].response_text,
        validation_results=format_results_dict[predictions.id],
        latency_ms=predictions_dict[predictions.id].latency_ms,
        used_repair=predictions_dict[predictions.id].used_repair,
        used_fallback=predictions_dict[predictions.id].used_fallback,
        trulens_scores=trulens_results_dict[predictions.id]
    )

def write_results_csv(results: Sequence[BatchEvalResult], out_path: Path):
    with open(out_path, 'w', newline='') as file:
        for row in results:
            writer = csv.writer(file)
            writer.writerows(row)
            writer.writerow("\n")

def write_results_json(results: Sequence[BatchEvalResult], out_path: Path):
    output_dir = {
        "id":[results.id],
        "prompt":[results.prompt],
        "response_text":[results.response_text],
        "format_valid":[results.format_valid],
        "num_questions":[results.num_questions],
        "format_errors":[results.format_errors],
        "latency_ms":[results.latency_ms],
        "used_repair":[results.used_repair],
        "used_fallback":[results.used_fallback],
        "trulens_score":[results.trulens_score]
    }
    df = pandas.DataFrame(output_dir)
    with open(out_path, 'w', newline='') as f:
        fieldNames = ["id", "prompt", "response_text", "format_valid", "num_questions", "format_errors", "latency_ms", "used_repair", "used_fallback", "trulens_score"]
        writer = csv.DictWriter(f, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(df)

def summarise_results(results: Sequence[BatchEvalResult]) -> dict[str, Any]:
    count_example = len(results.id)
    format_valid_percentage = sum(result.format_valid for result in results) / len(results.format_valid) * 100
    average_latency = sum(result.latency_ms for result in results) / len(results.latency_ms) * 100
    fallback_rate = sum(result.used_fallback for result in results) / len(results.used_fallback) * 100
    repair_rate = sum(result.used_repair for result in results) / len(results.used_fallback) * 100
    average_trulens_score = sum(result.trulens_score for result in results) / len(results.trulens_score) * 100
    return {
        "count_example": count_example,
        "format_valid_percentage": format_valid_percentage,
        "average_latency": average_latency,
        "fallback_rate": fallback_rate,
        "repair_rate": repair_rate,
        "average_trulens_score": average_trulens_score
    }

def run_batch_evaluation(data_path: Path, output_dir: Path, model_path: Path | None = None, limit: int | None = None, min_questions: int = 3, bullet_style: str = "either", metrics: Sequence[str] = ("coherence", "answer_relevance")) -> dict[str, Any]:

    settings_load = get_settings(env=model_path)
    eval_example_load = load_eval_examples(path=data_path, limit=limit)
    build_Runtime = build_runtime(settings_load, data_path)
    
    generate_Prediction = generate_prediction(build_Runtime, eval_example_load, settings_load.max_new_tokens, settings_load.temperature, settings_load.top_p)
    run_batch_Evaluation = run_batch_evaluation(generate_Prediction, min_questions, bullet_style)

    build_trulens_Dataset = build_trulens_dataset(generate_Prediction)
    build_trulens_Scorers = build_trulens_scorers(provider_name="openai", metrics=metrics)
    run_trulens_batch_Eval = run_trulens_batch_eval(build_trulens_Dataset, build_trulens_Scorers)

    result_csv = write_results_csv(run_trulens_batch_Eval, output_dir)
    result_json = write_results_json(run_trulens_batch_Eval, output_dir)

    return {
        "csv_path": result_csv,
        "json_path": result_json,
        "summary": summarise_results(run_trulens_batch_Eval)
    }