# **llm_followups — Production-Ready Async LLM Backend with Validation, CLI & Training Pipeline**

A complete **Python 3.13+ asynchronous LLM backend application** built using:

* FastAPI
* Pydantic v2
* Async HTTP client
* HuggingFace Transformers
* Structured validation pipeline
* Custom follow-up question formatting enforcement
* CLI client
* Training + dataset pipeline
* Full pytest test suite

This project demonstrates:

* Clean layered architecture (`server/`, `tuning/`, `utils/`)
* Strict schema validation via Pydantic
* Custom output validation logic (`validate_followup_list`)
* Repair + fallback logic for malformed model outputs
* Async runtime with streaming support
* Dataset preprocessing + tokenization pipeline
* Training loop with HuggingFace Trainer
* Structured logging
* CLI validation tool for SFT JSONL
* Fully working pytest suite

Completed as part of the **Elemental Concept Python Developer Internship Training Program — Program 5**.

---

## **Core Concept**

This project enforces a strict rule:

> The assistant must return only follow-up questions in bullet format.

Example required format:

```
- What environment are you deploying this to?
- Do you need GPU acceleration?
- Should responses be streamed or returned in full?
```

The backend:

* Generates responses
* Validates output format
* Attempts repair if malformed
* Falls back safely if validation fails
* Logs validation results

This mimics real-world production LLM guardrails.

---

## **Features**

| Category   | Description                            |
| ---------- | -------------------------------------- |
| Backend    | FastAPI async API server               |
| Runtime    | Async LLM generation with Transformers |
| Validation | Custom bullet-list follow-up validator |
| Repair     | Auto-repair malformed outputs          |
| Fallback   | Safe fallback question generator       |
| CLI        | Interactive chat client                |
| Dataset    | JSONL SFT format validation            |
| Training   | Tokenization + Trainer pipeline        |
| Logging    | Structured JSON or plain logging       |
| Testing    | pytest + schema + output tests         |

---

## **Project Structure**

```
llm_followups/
├── pyproject.toml
├── README.md
│
├── data/
│   └── sft_followups.jsonl
│
├── src/
│   └── llm_followups/
│       ├── server/
│       │   ├── main.py
│       │   ├── schemas.py
│       │   └── llm_runtime.py
│       │
│       ├── tuning/
│       │   ├── dataset.py
│       │   ├── train.py
│       │   ├── validate.py
│       │   └── log.py
│       │
│       └── utils/
│           └── config.py
│
├── tests/
│   ├── conftest.py
│   └── test_chat_format.py
│
└── scripts/
    └── validate_jsonl.py
```

---

## **Architecture Overview**

### 1️⃣ Server Layer

* `schemas.py` — Pydantic request/response models
* `llm_runtime.py` — Model loading + generation + repair logic
* `main.py` — FastAPI app

Handles:

* `/chat`
* `/health`

---

### 2️⃣ Validation Layer

`validate_followup_list()` enforces:

* Minimum number of questions
* Bullet style (`-` or `*`)
* No numbered lists
* No extra prose
* Question marks required

Returns:

```
ValidationResult(
    ok: bool,
    num_items: int,
    errors: list[str]
)
```

---

### 3️⃣ Repair Logic

If validation fails:

1. Attempt structured repair
2. Re-validate
3. If still invalid → fallback questions

Ensures production safety.

---

### 4️⃣ Dataset + Training Pipeline

Includes:

* `DatasetConfig`
* Text normalization
* Tokenization
* Length statistics
* HuggingFace `Trainer`
* TrainConfig validation

Supports SFT-style JSONL training data.

---

### 5️⃣ JSONL Validator CLI

```
python scripts/validate_jsonl.py data/sft_followups.jsonl
```

Checks:

* Valid JSON
* Schema correctness
* Proper bullet formatting
* No numbered lists
* Minimum questions

---

## **Running the Server**

```bash
uvicorn llm_followups.server.main:app --reload
```

Health check:

```
GET /health
```

Chat request:

```json
{
  "messages": [
    {"role": "user", "content": "Help me deploy a model"}
  ]
}
```

---

## **Running the CLI Client**

```bash
python -m llm_followups.cli
```

Interactive chat session with:

* `:reset`
* `:history`
* `:quit`

---

## **Running Tests**

```bash
pytest -v
```

Example:

```
collected 5 items
5 passed in 0.11s
```

Tests validate:

* JSONL correctness
* ChatRequest schema
* Assistant bullet formatting
* No numbered lists
* Min question count enforced

---

## **Tech Stack**

* Python 3.13+
* FastAPI
* Pydantic v2
* HuggingFace Transformers
* PyTorch
* pytest + pytest-asyncio
* Structured logging
* Async HTTP (httpx)
* Black / Ruff

---

## **Production Concepts Demonstrated**

* Guardrail enforcement
* Output validation
* Automatic repair loops
* Deterministic fallback
* Clean async architecture
* Schema-first design
* CLI + API integration
* Training + inference in same codebase

---

## **Outcome**

This project demonstrates:

* Production-safe LLM backend patterns
* Strict output format enforcement
* Clean modular architecture
* Full test coverage for data correctness
* CLI + API integration
* Training + inference workflow

A significant progression from Program 4 — moving from database architecture to **LLM production architecture**.

---

## **Author**

**Shyam Popat**
Intern Python Developer — Elemental Concept 2016 Ltd
GitHub: **sp2023lab**