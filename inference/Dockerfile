FROM asyameme/ecup26-matching-catboost:1.6

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    CE_BATCH_SIZE=128 \
    CE_MAX_LENGTH=256 \
    CE_CPU_THREADS=20

WORKDIR /app

RUN python -m pip install --no-cache-dir \
    "torch==2.8.0" \
    "transformers==4.57.1" \
    "sentencepiece>=0.2.0"

COPY model /app/model
COPY run_submission.py /app/run_submission.py
