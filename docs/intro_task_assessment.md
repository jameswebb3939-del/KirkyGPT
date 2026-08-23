# Intro Task Assessment — LLM Follow-Up Questions Project

## Overview

This project implements a **local LLM server and CLI client** that enforces a strict output rule:

> The assistant must return only follow-up questions in bullet format.

The system is built using **FastAPI, HuggingFace Transformers, and pytest**, and includes a full **dataset generation and fine-tuning pipeline**.

---

## How Requirements Are Met

### 1. Local LLM Server

* Implemented using FastAPI
* Exposes:

  * `GET /health`
  * `POST /chat`
* Loads model locally via HuggingFace Transformers
* Default model:

  ```
  meta-llama/Llama-3.2-1B-Instruct
  ```

---

### 2. CLI Chat Client

* Located at:

  ```
  src/llm_followups/client/cli.py
  ```
* Supports:

  * Interactive mode
  * One-shot queries (`--once`)
* Communicates with the FastAPI server

---

### 3. Follow-Up Question Enforcement

* Implemented via `validate_followup_list()`

* Enforces:

  * Bullet format (`-`)
  * Minimum number of questions
  * Each line ends with `?`
  * No extra prose
  * No numbered lists

* Includes:

  * Repair logic for malformed outputs
  * Fallback generation if validation fails

---

### 4. Fine-Tuning Pipeline

* Dataset generated via:

  ```
  scripts/make_sft.py
  ```
* Format:

  * JSONL
  * Chat-style messages
* Training:

  ```
  scripts/run_train.py
  ```
* Uses HuggingFace Trainer

---

### 5. Testing

* Implemented with pytest
* Includes:

  * API endpoint tests (`/health`, `/chat`)
  * Output format validation
  * Dataset validation

---

### 6. Project Structure & Packaging

* Uses modern Python packaging:

  * `pyproject.toml`
  * `src/` layout
* Dependencies managed via `requirements.txt`
* Clean modular structure:

  * `server/`
  * `client/`
  * `tuning/`
  * `utils/`

---

## Key Design Decisions

### Strict Output Validation

The system validates all model outputs before returning them.
This ensures consistent formatting and mimics production LLM guardrails.

### Repair + Fallback Strategy

If validation fails:

1. Attempt repair
2. Re-validate
3. Fallback to safe questions

This guarantees reliable responses.

### Separation of Concerns

* Server handles API + runtime
* Client handles interaction
* Tuning handles dataset + training
* Validation is isolated and reusable

---

## Summary

This project demonstrates:

* A complete LLM application (training + inference)
* Production-style validation and guardrails
* Clean architecture and modular design
* End-to-end functionality (CLI → API → model → validation)

It meets all requirements of the task while maintaining clarity, correctness, and reliability.
