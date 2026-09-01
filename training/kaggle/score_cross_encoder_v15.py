"""Score train/validation pairs with a fine-tuned cross-encoder.

The script reuses the text-cache and pair formatting from train_cross_encoder.py
and writes parquet files with an additional cross_encoder_score column.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from train_cross_encoder import (
    PairCollator,
    PairTextDataset,
    _barrier,
    _distributed_state,
    _is_main_process,
    build_item_text_cache,
)


def _prediction_arguments(output_dir: Path, batch_size: int) -> TrainingArguments:
    common = {
        "output_dir": str(output_dir),
        "per_device_eval_batch_size": batch_size,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
        "fp16": torch.cuda.is_available(),
        "bf16": False,
        "save_strategy": "no",
    }
    try:
        return TrainingArguments(evaluation_strategy="no", **common)
    except TypeError:
        return TrainingArguments(eval_strategy="no", **common)


def _score(trainer: Trainer, dataset: PairTextDataset) -> np.ndarray:
    prediction = trainer.predict(dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits).reshape(-1)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_pairs_path", required=True)
    parser.add_argument("--val_pairs_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_path", required=True)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()

    rank, world_size, distributed = _distributed_state()
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for cross-encoder scoring")

    if torch.cuda.is_available():
        local_rank = int(torch.cuda.current_device())
        print(
            f"Scoring device: cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)}), "
            f"rank={rank}/{world_size}",
            flush=True,
        )

    train_pairs = pd.read_parquet(args.train_pairs_path, columns=["id1", "id2", "target"])
    val_pairs = pd.read_parquet(args.val_pairs_path, columns=["id1", "id2", "target", "category"])
    item_ids = set(train_pairs["id1"].astype(int)) | set(train_pairs["id2"].astype(int))
    item_ids.update(val_pairs["id1"].astype(int))
    item_ids.update(val_pairs["id2"].astype(int))

    if _is_main_process(rank):
        build_item_text_cache(
            Path(args.data_dir) / "items.parquet",
            Path(args.cache_path),
            item_ids,
        )
    _barrier(distributed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    collator = PairCollator(tokenizer, args.max_length)
    trainer = Trainer(
        model=model,
        args=_prediction_arguments(Path(args.output_dir), args.batch_size),
        processing_class=tokenizer,
        data_collator=collator,
    )

    train_dataset = PairTextDataset(train_pairs, Path(args.cache_path))
    val_dataset = PairTextDataset(val_pairs, Path(args.cache_path))
    train_scores = _score(trainer, train_dataset)
    _barrier(distributed)
    val_scores = _score(trainer, val_dataset)
    _barrier(distributed)

    if _is_main_process(rank):
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_out = train_pairs.copy()
        train_out["cross_encoder_score"] = train_scores
        train_out.to_parquet(output_dir / "train_pairs_with_cross_encoder.parquet", index=False)

        val_out = val_pairs.copy()
        val_out["cross_encoder_score"] = val_scores
        val_out.to_parquet(output_dir / "val_pairs_with_cross_encoder.parquet", index=False)

        per_category = {}
        for category, group in val_out.groupby("category", sort=True):
            per_category[str(category)] = float(
                average_precision_score(group["target"], group["cross_encoder_score"])
            )

        report = {
            "train_pairs": len(train_out),
            "val_pairs": len(val_out),
            "overall_average_precision": float(
                average_precision_score(val_out["target"], val_out["cross_encoder_score"])
            ),
            "macro_average_precision": float(np.mean(list(per_category.values()))),
            "per_category_average_precision": per_category,
            "world_size": world_size,
        }
        (output_dir / "cross_encoder_score_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
