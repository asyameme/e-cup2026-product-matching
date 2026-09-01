from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


SOLUTION_ROOT = Path(__file__).resolve().parent
MODEL_DIR = SOLUTION_ROOT / "model"
MAX_LENGTH = int(os.environ.get("CE_MAX_LENGTH", "256"))
DEFAULT_BATCH_SIZE = int(os.environ.get("CE_BATCH_SIZE", "128"))


def _json_attributes(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def product_text(name: Any, category: Any, attributes: Any) -> str:
    name_text = "" if name is None else str(name)
    category_text = "<NULL>" if category is None else str(category)
    attrs = _json_attributes(attributes)
    attr_text = " ".join(f"{key}: {value}" for key, value in attrs.items())
    return f"Name: {name_text} Category: {category_text} Attributes: {attr_text}"


def load_item_texts(items_path: Path) -> dict[int, str]:
    items = pd.read_parquet(
        items_path,
        columns=["id", "name", "attributes", "category"],
    )
    texts: dict[int, str] = {}
    for row in items.itertuples(index=False):
        texts[int(row.id)] = product_text(row.name, row.category, row.attributes)
    return texts


def load_model() -> tuple[Any, Any, torch.device]:
    required = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "sentencepiece.bpe.model",
    ]
    missing = [name for name in required if not (MODEL_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cross-encoder model is incomplete in {MODEL_DIR}: {missing}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        thread_count = int(os.environ.get("CE_CPU_THREADS", "8"))
        torch.set_num_threads(max(1, thread_count))

    print(f"Loading cross-encoder from {MODEL_DIR}", flush=True)
    print(f"Inference device: {device}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    ).to(device)
    model.eval()

    # The model was fine-tuned in fp32. Use fp16 only when CUDA is available.
    if device.type == "cuda" and os.environ.get("CE_FP16", "1") == "1":
        model = model.half()

    return tokenizer, model, device


def score_pairs(
    pairs: list[tuple[str, str]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    scores = np.empty(len(pairs), dtype=np.float32)
    started = time.perf_counter()

    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            stop = min(start + batch_size, len(pairs))
            left = [pair[0] for pair in pairs[start:stop]]
            right = [pair[1] for pair in pairs[start:stop]]
            encoded = tokenizer(
                left,
                right,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits

            if logits.ndim == 2 and logits.shape[1] == 1:
                batch_scores = torch.sigmoid(logits[:, 0])
            elif logits.ndim == 2 and logits.shape[1] >= 2:
                batch_scores = torch.softmax(logits, dim=1)[:, 1]
            else:
                batch_scores = torch.sigmoid(logits.reshape(-1))

            scores[start:stop] = batch_scores.float().cpu().numpy()

            if start == 0 or stop == len(pairs) or stop % 5000 == 0:
                print(
                    f"Scored {stop:,}/{len(pairs):,} pairs",
                    flush=True,
                )

    print(
        f"Scoring finished in {time.perf_counter() - started:.1f}s",
        flush=True,
    )
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", "--output-path", "-o", dest="output_path", required=True)
    parser.add_argument("--items_path", "--items-path", "-i", dest="items_path", required=True)
    parser.add_argument("--matches_path", "--matches-path", "-m", dest="matches_path", required=True)
    args = parser.parse_args()

    items_path = Path(args.items_path)
    matches_path = Path(args.matches_path)
    output_path = Path(args.output_path)

    started = time.perf_counter()
    item_texts = load_item_texts(items_path)
    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])

    pairs: list[tuple[str, str]] = []
    missing_ids: set[int] = set()
    for row in matches.itertuples(index=False):
        id1, id2 = int(row.id1), int(row.id2)
        text1 = item_texts.get(id1)
        text2 = item_texts.get(id2)
        if text1 is None:
            missing_ids.add(id1)
        if text2 is None:
            missing_ids.add(id2)
        pairs.append((text1 or "", text2 or ""))

    if missing_ids:
        raise KeyError(
            f"{len(missing_ids)} item IDs from matches are missing in items: "
            f"{sorted(missing_ids)[:10]}"
        )

    tokenizer, model, device = load_model()
    predictions = score_pairs(
        pairs,
        tokenizer,
        model,
        device,
        batch_size=DEFAULT_BATCH_SIZE,
    )

    output = matches.copy()
    output["predict"] = predictions.astype(float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    print(
        f"Wrote {len(output):,} predictions to {output_path}; "
        f"total runtime {time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
