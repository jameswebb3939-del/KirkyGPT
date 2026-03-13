import argparse
import json
from pathlib import Path
from typing import Sequence, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import statistics

from llm_followups.utils.config import Settings
from llm_followups.server.llm_runtime import LLMRuntime, GenerationResult
from llm_followups.server.schemas import ChatMessage

#SRC.LLM_FOLLOWUPS IS CORRECT - NOTE: DO NOT CHANGE THIS AS IT WILL BE INCORRECT OTHERWISE!!!!

from trulens.providers.openai import OpenAI


@dataclass
class EvaluationRow:
    prompt: str
    raw_output: str
    final_output: str
    relevance_score: Optional[float]
    relevance_reason: Optional[str]
    correctness_score: Optional[float]
    correctness_reason: Optional[str]
    # Optionally, include groundedness if you define a custom metric
    groundedness_score: Optional[float] = None
    groundedness_reason: Optional[str] = None

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate LLM outputs with TruLens feedbacks')
    parser.add_argument('--model', type=str, help='Model/Runtime settings')
    parser.add_argument('--prompts-file', type=str, default=None, help='Input prompt file path')
    parser.add_argument('--output', type=str, default='outputs/eval_results.json', help='Output file path')
    parser.add_argument('--limit', type=int, default=10, help='Number of examples to run')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', type=int, help="Seed for reproducibility")
    return parser

def load_eval_prompts(path: Optional[Path], limit: Optional[int] = None) -> Sequence[str]:
    prompts = []
    if path is not None:
        with open(path, 'r') as f:
            for line in f:
                prompt = line.strip()
                if prompt:
                    prompts.append(prompt)
    else:
        prompts = [
            "How do I use Docker Compose to start multiple services?",
            "Write a pytest test for a FastAPI endpoint.",
            "How can I use LocalStack for AWS emulation?",
            "What is the best way to cache data in Redis?",
            "How do I write a good README for a Python project?",
            "Explain how to fine-tune a transformer model.",
            "How do I set up a CI pipeline for Python?",
            "What are common issues with Python virtual environments?",
            "How do I debug a failing Docker container?",
            "How can I use environment variables in pytest?",
            "How do I use SQLAlchemy for database access?"
        ]
    prompts = [p for p in prompts if p.strip()]
    if limit is not None:
        prompts = prompts[:limit]
    return prompts

def make_runtime(settings: Settings) -> LLMRuntime:
    return LLMRuntime(settings)

async def generate_response(runtime: LLMRuntime, prompt: str) -> dict[str, str]:
    chat_msg = ChatMessage(role="user", content=prompt)
    req = runtime.make_request(messages=[chat_msg])
    result: GenerationResult = await runtime.generate(req)
    return {
        "prompt": prompt,
        "raw_output": result.raw_text,
        "final_output": result.final_text
    }

def build_feedbacks(provider: OpenAI) -> dict[str, Any]:
    # Return provider methods directly for existing data evaluation
    return {
        "relevance": provider.relevance_with_cot_reasons,
        "correctness": provider.correctness_with_cot_reasons
    }

async def evaluate_prompt(runtime: LLMRuntime, prompt: str, feedbacks: dict[str, Any]) -> EvaluationRow:
    resp = await generate_response(runtime, prompt)
    prompt = resp["prompt"]
    raw_output = resp["raw_output"]
    final_output = resp["final_output"]

    # Call provider methods directly for existing data evaluation
    relevance_result = feedbacks["relevance"](prompt, final_output)
    correctness_result = feedbacks["correctness"](final_output)

    return EvaluationRow(
        prompt=prompt,
        raw_output=raw_output,
        final_output=final_output,
        relevance_score=getattr(relevance_result, "score", None),
        relevance_reason=getattr(relevance_result, "reason", None),
        correctness_score=getattr(correctness_result, "score", None),
        correctness_reason=getattr(correctness_result, "reason", None)
    )


async def evaluate_prompts(runtime: LLMRuntime, prompts: Sequence[str], feedbacks: dict[str, Any], verbose: bool = False) -> list[EvaluationRow]:
    results = []
    for idx, prompt in enumerate(prompts):
        row = await evaluate_prompt(runtime, prompt, feedbacks)
        results.append(row)
        if verbose:
            print(f"[{idx+1}/{len(prompts)}] Evaluated: {prompt}")
    return results


def summarize_results(rows: list[EvaluationRow]) -> dict[str, float | int | None]:
    summary = {}
    summary["total"] = len(rows)
    relevance_scores = [row.relevance_score for row in rows if row.relevance_score is not None]
    correctness_scores = [row.correctness_score for row in rows if row.correctness_score is not None]
    summary["mean_relevance"] = statistics.mean(relevance_scores) if relevance_scores else None
    summary["mean_correctness"] = statistics.mean(correctness_scores) if correctness_scores else None
    return summary

def write_results(rows: list[EvaluationRow], summary: dict[str, float | int | None], output_path: Path) -> None:
    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    results_dict = {
        "summary": summary,
        "rows": [asdict(row) for row in rows]
    }
    with open(output_path, "w") as file:
        json.dump(results_dict, file, indent=2)
    print(f"Results written to: {output_path}")


async def async_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    settings_kwargs = {}
    if args.model:
        settings_kwargs["model_name"] = args.model
    settings = Settings(**settings_kwargs)

    # Optionally set random seed
    if args.seed is not None:
        import random
        random.seed(args.seed)

    # Create runtime
    runtime = make_runtime(settings)

    # Create provider (OpenAI example, adjust as needed)
    provider = OpenAI()

    # Build feedbacks
    feedbacks = build_feedbacks(provider)

    # Load prompts
    prompts = load_eval_prompts(Path(args.prompts_file) if args.prompts_file else None, args.limit)

    # Evaluate prompts
    rows = await evaluate_prompts(runtime, prompts, feedbacks, verbose=args.verbose)

    # Summarize results
    summary = summarize_results(rows)

    # Write output
    write_results(rows, summary, Path(args.output))

    # Print summary
    print("Evaluation Summary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    return 0

def main(argv: Optional[Sequence[str]] = None) -> int:
    return asyncio.run(async_main(argv))

if __name__ == "__main__":
    main()