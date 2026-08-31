# `llm_followups`

Local LLM follow-up-question generation, fine-tuning, guardrails, and black-box evaluation.

This project was developed as part of the **Elemental Concept Python Developer Internship Training Program — Program 5**.

The application runs a local Hugging Face causal language model behind FastAPI and a CLI, with one strict behavioural requirement:

> **Return only follow-up questions in bullet-list format.**

The project also includes deterministic output validation, repair/fallback guardrails, JSONL SFT dataset generation, Hugging Face fine-tuning, sanity inference, black-box batch evaluation, rubric-based LLM-as-a-judge scoring, OpenAI judge integration, CSV/JSON reporting, and automated tests.

## Current Status

```text
pytest -v
34 passed

pytest tests/eval -v
27 passed
```

The remaining warnings are FastAPI deprecation warnings for `@app.on_event("startup")`.

## Core Architecture

```text
APPLICATION

CLI / FastAPI
      ↓
  LLMRuntime
      ↓
Local Hugging Face Model
      ↓
Validate → Repair → Fallback
      ↓
Contract-compliant response


TRAINING

Topic templates
      ↓
SFT JSONL
      ↓
Validation
      ↓
Dataset + Tokenizer
      ↓
Trainer
      ↓
Fine-tuned checkpoint


EVALUATION

DatasetSource
      ↓
Target
      ↓
EvalPrediction
      ↓
Evaluator(s)
   ├── deterministic format
   └── rubric → Judge
      ↓
EvaluationResult
      ↓
Summary + Reporter(s)
```

The evaluation layer follows a **black-box throughout** design using replaceable contracts:

```text
DatasetSource
Target
Evaluator
Judge
Reporter
```

See [`architecture.md`](architecture.md) for the detailed architecture.

## Project Structure

```text
ElementalConceptTrainingProgram5/
│
├── data/
│   ├── sft_followups.jsonl
│   ├── sft_followups_train_v3.jsonl
│   ├── sft_followups_eval_v3.jsonl
│   └── sft_followups_v3_report.txt
│
├── outputs/
│   └── ... trained model checkpoints
│
├── scripts/
│   ├── make_sft.py
│   ├── run_train.py
│   ├── sanity_infer.py
│   └── run_trulens_eval.py
│
├── src/
│   └── llm_followups/
│       ├── client/
│       │   └── cli.py
│       ├── server/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   └── llm_runtime.py
│       ├── tuning/
│       │   ├── dataset.py
│       │   ├── train.py
│       │   ├── validate.py
│       │   └── log.py
│       ├── utils/
│       │   └── config.py
│       └── eval/
│           ├── batch_eval.py
│           ├── rubrics.py
│           ├── core/
│           │   ├── models.py
│           │   ├── protocols.py
│           │   ├── runner.py
│           │   └── summary.py
│           ├── datasets/
│           │   └── jsonl.py
│           ├── evaluators/
│           │   ├── format.py
│           │   └── rubric.py
│           ├── judges/
│           │   └── openai.py
│           ├── reporters/
│           │   ├── csv.py
│           │   └── json.py
│           └── targets/
│               └── llm_runtime.py
│
├── tests/
│   ├── conftest.py
│   ├── test_chat_format.py
│   └── eval/
│       ├── test_format_evaluator.py
│       ├── test_jsonl_source.py
│       ├── test_llm_runtime_target.py
│       ├── test_openai_judge.py
│       ├── test_pipeline.py
│       ├── test_reporters.py
│       ├── test_rubric_evaluator.py
│       ├── test_runner.py
│       └── test_summary.py
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── architecture.md
```

## Requirements

Python:

```text
Python >= 3.13
```

Latest verified local test run:

```text
Python 3.14.0
```

Main stack:

- FastAPI
- Uvicorn
- Pydantic
- httpx
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Accelerate
- OpenAI Python SDK
- pytest
- pytest-asyncio

## Installation

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Python 3.13+ can be used if 3.14 is not installed.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Hugging Face Access

Default model:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Authenticate if required:

```powershell
hf auth login
hf auth whoami
```

## Running the Server

### Windows PowerShell

```powershell
$env:PYTHONPATH="src"
$env:MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
$env:DEVICE="cpu"

python -m uvicorn llm_followups.server.main:app --reload
```

Default server:

```text
http://127.0.0.1:8000
```

### Health

```http
GET /health
```

### Chat

```http
POST /chat
```

Example:

```json
{
  "messages": [
    {"role": "user", "content": "Help me deploy a model"}
  ]
}
```

## Running the CLI

Interactive:

```powershell
python -m llm_followups.client.cli
```

One-shot:

```powershell
python -m llm_followups.client.cli --once "Explain Docker containers"
```

Local REPL commands:

```text
:reset
:history
:q
:quit
```

## Output Validation and Guardrails

Validation is implemented in:

```text
src/llm_followups/tuning/validate.py
```

The runtime enforces:

- bullet output
- minimum question count
- trailing `?`
- no extra prose
- no numbered lists
- normalized output

Reliability path:

```text
Generate
  ↓
Validate
  ↓
Repair
  ↓
Fallback
```

## Dataset Generation

Script:

```text
scripts/make_sft.py
```

Example:

```powershell
python scripts/make_sft.py --out data/sft_followups.jsonl --n 2000 --k 3 --seed 42
```

Current v3 dataset report:

```text
Examples: 2000
Strict format failures: 0
Unique question lines: 2326 / 6000
Unique assistant responses: 2000 / 2000
Unique prompts: 600 / 2000
```

## Fine-Tuning

Main files:

```text
src/llm_followups/tuning/train.py
scripts/run_train.py
```

Current training configuration uses:

```text
base model: meta-llama/Llama-3.2-1B-Instruct
dataset: data/sft_followups_train_v3.jsonl
output: outputs/llama7
max_length: 512
learning rate: 5e-6
batch size: 1
epochs: 2
device: cuda
```

Run:

```powershell
python scripts/run_train.py
```

## Sanity Inference

```powershell
python scripts/sanity_infer.py --model-path outputs/llama7 --device auto
```

This reports raw output, strict-format validity, validation errors, latency, and a final summary.

## Evaluation Framework

The evaluation package lives under:

```text
src/llm_followups/eval/
```

### Core protocols

```text
DatasetSource
Target
Evaluator
Judge
Reporter
```

### Core result models

```text
EvalExample
EvalPrediction
EvaluationResult
EvaluatedExample
JudgeResult
```

### Current target

```text
LLMRuntimeTarget
```

Wraps the existing `LLMRuntime`.

### Current evaluators

```text
FollowupFormatEvaluator
RubricEvaluator
```

### Current rubric dimensions

#### Coherence

```text
0 = incoherent or unreadable
1 = weakly coherent
2 = mostly coherent
3 = very coherent, well-structured, and easy to follow
pass threshold = 2.0
```

#### Answer relevance

```text
0 = not relevant
1 = slightly relevant
2 = mostly relevant
3 = directly relevant and useful
pass threshold = 2.0
```

### Current judge provider

```text
OpenAIJudge
```

Default judge model:

```text
gpt-4o-mini
```

The OpenAI adapter returns a provider-neutral `JudgeResult`.

### Current dataset source

```text
JSONLDatasetSource
```

Supported shapes:

```json
{"prompt": "..."}
```

```json
{"user_message": "..."}
```

and SFT-style `messages`.

### Current reporters

```text
CSVReporter
JSONReporter
```

### Current summary metrics

```text
count_example
average_latency_ms
fallback_rate
repair_rate
format_valid_percentage
per-evaluator average scores
per-evaluator pass rates
```

## Running Batch Evaluation

Current entry point:

```text
scripts/run_trulens_eval.py
```

The filename is historical; the current implementation is now a general black-box/rubric evaluation pipeline.

Set the OpenAI key for real rubric evaluation:

```powershell
$env:OPENAI_API_KEY="your-key"
```

Do not commit API keys.

Start with a small run:

```powershell
python scripts/run_trulens_eval.py `
  --data_path data/sft_followups_eval_v3.jsonl `
  --output_dir eval_results `
  --model_path outputs/llama7 `
  --limit 5 `
  --min_questions 3 `
  --bullet_style either
```

Outputs:

```text
eval_results/
├── results.csv
└── results.json
```

Inspect:

- raw output
- final output
- format validity
- coherence score
- relevance score
- repair/fallback usage
- latency
- batch summary

Then remove `--limit 5` for the full evaluation run.

## Testing

All tests:

```powershell
pytest -v
```

Verified:

```text
34 passed
```

Evaluation suite:

```powershell
pytest tests/eval -v
```

Verified:

```text
27 passed
```

The evaluation tests use fake targets, fake judges, and mocked provider clients so they do not require a real model or OpenAI network request.

## Configuration

Main configuration lives in:

```text
src/llm_followups/utils/config.py
```

Environment variables include:

```text
MODEL_NAME
SERVER_HOST
SERVER_PORT
ENDPOINT_CHAT
ENDPOINT_HEALTH
REQUEST_TIMEOUT_S
MAX_NEW_TOKENS
TEMPERATURE
TOP_P
SEED
DEVICE
ADAPTER_PATH
ENFORCE_FORMAT
MIN_QUESTIONS
BULLET_STYLE
```

## Dependency Management

`pyproject.toml` declares project/dev dependencies.

`requirements.txt` can be refreshed from the current virtual environment with:

```powershell
python -m pip freeze > requirements.txt
```

Then inspect:

```powershell
git diff requirements.txt
```

The current evaluation setup requires `openai` and `pytest-asyncio`.

## Known Limitations

- FastAPI `@app.on_event("startup")` is deprecated in favour of lifespan handlers.
- Evaluation is currently sequential.
- OpenAI is currently the only concrete judge provider.
- PEFT adapter loading is not implemented.
- Runtime generation is currently greedy (`do_sample = False`).
- Prompting is strongly dash-bullet oriented even though validation can support other styles.
- Generic rubric evaluation does not yet enforce score bounds.
- `run_trulens_eval.py` should eventually be renamed to reflect the broader evaluation framework.

## Extension Points

The black-box design supports future implementations such as:

### Evaluators

```text
ExactMatchEvaluator
JSONSchemaEvaluator
RegexEvaluator
SemanticSimilarityEvaluator
SafetyEvaluator
```

### Judges

```text
AnthropicJudge
LocalModelJudge
HTTPJudge
```

### Targets

```text
HTTP agent
RAG system
LangGraph workflow
remote model API
local model
```

### Dataset sources

```text
CSV
PostgreSQL
S3
API
```

### Reporters

```text
MLflow
database
S3
HTML
dashboard
```

## Further Documentation

For the detailed design, component responsibilities, data flows, extension points, and technical debt, see:

[`architecture.md`](architecture.md)

## Author

**Shyam Popat**

Python Developer Intern / Software Developer Intern  
Elemental Concept 2016 Ltd
