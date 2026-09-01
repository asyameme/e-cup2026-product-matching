FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    CE_BATCH_SIZE=128 \
    CE_MAX_LENGTH=256 \
    CE_CPU_THREADS=20

WORKDIR /app

RUN apt-get -o Acquire::Check-Date=false \
        -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir \
    "numpy==2.1.3" \
    "pandas==2.2.3" \
    "pyarrow==18.1.0" \
    "torch==2.8.0" \
    "transformers==4.57.1" \
    "sentencepiece>=0.2.0"

COPY model /app/model
COPY run_submission.py /app/run_submission.py
