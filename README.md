# **llm_followups — Local LLM Server + CLI (Follow-Up Questions Only)**

A complete **Python 3.13.9** project that:

* Runs a **local LLM server** using Hugging Face Transformers
* Provides a **CLI chat client**
* Fine-tunes a model to return **only follow-up questions**
* Includes a **dataset + training pipeline**
* Includes **end-to-end pytest tests**

Completed as part of the **Elemental Concept Python Developer Internship Training Program — Program 5**.

---

## **Core Concept**

This project enforces a strict rule:

> The assistant must return only follow-up questions in bullet format.

Example required output:

```
- What environment are you deploying this to?
- What constraints should be considered for this setup?
- What output format do you expect from the system?
```

The system:

* Generates responses using a local LLM
* Validates formatting strictly
* Attempts repair if invalid
* Falls back safely if needed

---

## **Default Model**

```
meta-llama/Llama-3.2-1B-Instruct
```

---

## **Features**

| Category   | Description                      |
| ---------- | -------------------------------- |
| Backend    | FastAPI async API server         |
| CLI        | Interactive chat client          |
| LLM        | HuggingFace Transformers         |
| Validation | Strict bullet-format enforcement |
| Repair     | Auto-fix malformed outputs       |
| Fallback   | Safe fallback questions          |
| Dataset    | JSONL SFT dataset generation     |
| Training   | HuggingFace Trainer pipeline     |
| Testing    | pytest + API tests               |

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
│       ├── client/
│       │   └── cli.py
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
    ├── make_sft.py
    └── run_train.py
```

---

## **Setup**

### Python version

```
Python 3.13.9
```

### Create virtual environment

#### Windows (PowerShell)

```powershell
py -3.13 -m venv venv
venv\Scripts\Activate.ps1
```

#### Windows (cmd)

```cmd
py -3.13 -m venv venv
venv\Scripts\activate.bat
```

#### macOS / Linux

```bash
python3.13 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Freeze dependencies

```bash
pip freeze > requirements.txt
```

---

## **Running the Server**

```bash
python -m uvicorn llm_followups.server.main:app --reload
```

### Health check

```
GET /health
```

### Chat request

```json
{
  "messages": [
    {"role": "user", "content": "Help me deploy a model"}
  ]
}
```

---

## **Running the CLI**

```bash
python -m llm_followups.client.cli
```

Optional:

```bash
python -m llm_followups.client.cli --once "Explain Docker containers"
```

---

## **Generate Dataset**

```bash
PYTHONPATH=src python scripts/make_sft.py --out data/sft_followups.jsonl --n 300 --seed 42
```

---

## **Run Training**

```bash
PYTHONPATH=src python scripts/run_train.py
```

---

## **Run Tests**

```bash
pytest -v
```

Tests include:

* API `/health` endpoint
* API `/chat` endpoint
* Follow-up question format validation
* JSONL dataset structure

---

## **Tech Stack**

* Python 3.13.9
* FastAPI
* Pydantic v2
* HuggingFace Transformers
* PyTorch
* pytest + pytest-asyncio
* httpx

---

## **Key Concepts Demonstrated**

* LLM guardrails and validation
* Structured output enforcement
* Repair + fallback mechanisms
* Async API design
* CLI + API integration
* Dataset + fine-tuning workflow

---

## **Author**

**Shyam Popat**
Intern Python Developer — Elemental Concept 2016 Ltd
GitHub: **sp2023lab**
