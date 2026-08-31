FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/hf-cache

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
    && python -m pip install \
        --no-cache-dir \
        --index-url \
        https://download.pytorch.org/whl/cpu \
        torch==2.10.0 \
    && python -m pip install \
        --no-cache-dir \
        . \
    && python -m pip install \
        --no-cache-dir \
        protobuf \
        sentencepiece \
        tiktoken

COPY scripts ./scripts
COPY data ./data

RUN mkdir -p \
    /app/outputs \
    /app/eval_results \
    /hf-cache

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=300s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "llm_followups.server.main:app", "--host", "0.0.0.0", "--port", "8000"]