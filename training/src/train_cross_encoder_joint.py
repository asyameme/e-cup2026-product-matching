"""Fine-tune the existing cross-encoder on a mixed matching dataset.

This is a training-only script. It does not change the CatBoost pipeline or
the final submission runner. Item texts are kept in a disk-backed SQLite
cache so that the 3.8 GB items.parquet file is never loaded into RAM.

This variant also reports macro average precision by category and supports a
fixed max_steps budget, which is useful when the Kaggle session has a strict
time limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    TrainingArguments,
)


SEED = 42


def _distributed_state() -> tuple[int, int, bool]:
    """Return rank, world size, and whether torchrun DDP is active."""
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP was requested, but CUDA is not available")
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        torch.cuda.set_device(local_rank)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, distributed


def _is_main_process(rank: int) -> bool:
    return rank == 0


def _barrier(distributed: bool) -> None:
    if distributed:
        torch.distributed.barrier()


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


def _product_text(name: Any, category: Any, attributes: Any) -> str:
    name_text = "" if name is None else str(name)
    category_text = "<NULL>" if category is None else str(category)
    attrs = _json_attributes(attributes)
    attr_text = " ".join(f"{key}: {value}" for key, value in attrs.items())
    return f"Name: {name_text} Category: {category_text} Attributes: {attr_text}"


def _ids_signature(items_path: Path, item_ids: set[int]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for item_id in sorted(item_ids):
        digest.update(str(item_id).encode("ascii"))
        digest.update(b"\n")
    stat = items_path.stat()
    return {
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "item_count": len(item_ids),
        "ids_sha256": digest.hexdigest(),
    }


def build_item_text_cache(
    items_path: Path,
    cache_path: Path,
    item_ids: set[int],
    batch_size: int = 10_000,
) -> None:
    """Cache only the texts needed by train and validation pairs."""
    if not item_ids:
        raise ValueError("No item IDs were provided")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    signature = _ids_signature(items_path, item_ids)
    if cache_path.exists() and manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == signature:
                print(f"Reusing item text cache: {cache_path}", flush=True)
                return
        except (OSError, ValueError, json.JSONDecodeError):
            pass

    cache_path.unlink(missing_ok=True)
    manifest_path.unlink(missing_ok=True)
    connection = sqlite3.connect(cache_path)
    try:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=FILE;
            CREATE TABLE item_texts (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL,
                category TEXT NOT NULL
            ) WITHOUT ROWID;
            """
        )
        parquet = pq.ParquetFile(items_path)
        scanned = 0
        cached = 0
        for batch_number, batch in enumerate(
            parquet.iter_batches(
                columns=["id", "name", "attributes", "category"],
                batch_size=batch_size,
            ),
            start=1,
        ):
            frame = batch.to_pandas()
            rows = []
            for row in frame.itertuples(index=False):
                item_id = int(row.id)
                if item_id not in item_ids:
                    continue
                category = "<NULL>" if row.category is None else str(row.category)
                rows.append((item_id, _product_text(row.name, row.category, row.attributes), category))
            if rows:
                connection.executemany(
                    "INSERT INTO item_texts (id, text, category) VALUES (?, ?, ?)",
                    rows,
                )
                connection.commit()
                cached += len(rows)
            scanned += len(frame)
            if batch_number == 1 or batch_number % 100 == 0:
                print(
                    f"text cache: {scanned:,}/{parquet.metadata.num_rows:,} rows scanned, "
                    f"{cached:,} records cached",
                    flush=True,
                )
            del frame, rows
        connection.commit()
    finally:
        connection.close()
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    print(f"Item text cache ready: {cache_path} ({cached:,} records)", flush=True)


class PairTextDataset(Dataset):
    """Map-style dataset that reads two texts per pair from SQLite."""

    def __init__(self, pairs: pd.DataFrame, cache_path: Path):
        required = {"id1", "id2", "target"}
        missing = required - set(pairs.columns)
        if missing:
            raise ValueError(f"Pair table is missing columns: {sorted(missing)}")
        self.id1 = pairs["id1"].to_numpy(dtype=np.int64)
        self.id2 = pairs["id2"].to_numpy(dtype=np.int64)
        self.labels = pairs["target"].to_numpy(dtype=np.float32)
        self.cache_path = str(cache_path)
        self._connection: sqlite3.Connection | None = None

    def __len__(self) -> int:
        return len(self.labels)

    def _conn(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(f"file:{self.cache_path}?mode=ro", uri=True)
        return self._connection

    def _text(self, item_id: int) -> str:
        row = self._conn().execute(
            "SELECT text FROM item_texts WHERE id = ?", (int(item_id),)
        ).fetchone()
        if row is None:
            raise KeyError(f"Item ID {item_id} is missing from the text cache")
        return str(row[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "text1": self._text(self.id1[index]),
            "text2": self._text(self.id2[index]),
            "labels": float(self.labels[index]),
        }


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            [feature["text1"] for feature in features],
            [feature["text2"] for feature in features],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [feature["labels"] for feature in features], dtype=torch.float32
        )
        return encoded


def _metrics(
    prediction: EvalPrediction,
    categories: np.ndarray | None = None,
) -> dict[str, float]:
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    labels = np.asarray(prediction.label_ids).reshape(-1)
    metrics = {
        "average_precision": float(average_precision_score(labels, probabilities)),
    }
    if categories is None:
        return metrics
    categories = np.asarray(categories).reshape(-1)
    if len(categories) != len(labels):
        raise ValueError(
            f"Validation categories length {len(categories)} does not match "
            f"labels length {len(labels)}"
        )
    per_category = []
    for category in np.unique(categories):
        mask = categories == category
        if len(np.unique(labels[mask])) < 2:
            continue
        per_category.append(
            average_precision_score(labels[mask], probabilities[mask])
        )
    if not per_category:
        raise ValueError("Cannot compute macro_average_precision: no valid categories")
    metrics["macro_average_precision"] = float(np.mean(per_category))
    return metrics


def _training_arguments(args: argparse.Namespace, output_dir: Path) -> TrainingArguments:
    common = {
        "output_dir": str(output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_steps": args.max_steps,
        "weight_decay": 0.01,
        "warmup_ratio": 0.05,
        "logging_steps": 100,
        "eval_steps": args.eval_steps,
        "save_steps": args.eval_steps,
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_average_precision",
        "greater_is_better": True,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "ddp_find_unused_parameters": False,
        "seed": SEED,
        "fp16": bool(torch.cuda.is_available() and not args.bf16),
        "bf16": bool(torch.cuda.is_available() and args.bf16),
    }
    try:
        return TrainingArguments(evaluation_strategy="steps", **common)
    except TypeError:
        # Transformers 4.4x renamed evaluation_strategy to eval_strategy.
        return TrainingArguments(eval_strategy="steps", **common)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--train_pairs_path", required=True)
    parser.add_argument("--val_pairs_path", required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning_rate", type=float, default=3e-7)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=16000,
        help="Fixed optimizer-step budget; positive value overrides epochs. "
        "Values above 16000 are rejected to keep the Kaggle run time-bounded.",
    )
    parser.add_argument("--eval_steps", type=int, default=2000)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--require_cuda", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.max_steps > 16000:
        raise ValueError(
            "--max_steps must be between 1 and 16000 for the six-hour run budget"
        )

    started = time.perf_counter()
    rank, world_size, distributed = _distributed_state()
    cuda_available = torch.cuda.is_available()
    if args.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is not available. In Kaggle enable Settings -> Accelerator -> GPU "
            "before starting cross-encoder training."
        )
    if cuda_available:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        print(
            f"Training device: cuda:{local_rank} ({torch.cuda.get_device_name(local_rank)}), "
            f"rank={rank}/{world_size}, "
            f"fp16={not args.bf16}, bf16={args.bf16}",
            flush=True,
        )
        torch.backends.cuda.matmul.allow_tf32 = True
        if world_size == 1 and torch.cuda.device_count() > 1 and _is_main_process(rank):
            print(
                f"CUDA devices visible: {torch.cuda.device_count()}, but only one process is active. "
                "Use torchrun --nproc_per_node=2 to use both GPUs.",
                flush=True,
            )
    else:
        print("Training device: CPU", flush=True)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache_path) if args.cache_path else output_dir / "item_texts.sqlite"

    train_pairs = pd.read_parquet(args.train_pairs_path, columns=["id1", "id2", "target"])
    val_pairs = pd.read_parquet(args.val_pairs_path, columns=["id1", "id2", "target", "category"])
    all_item_ids = set(train_pairs["id1"].astype(int)) | set(train_pairs["id2"].astype(int))
    all_item_ids.update(val_pairs["id1"].astype(int))
    all_item_ids.update(val_pairs["id2"].astype(int))

    if _is_main_process(rank):
        print(f"Train pairs: {len(train_pairs):,}; validation pairs: {len(val_pairs):,}", flush=True)
        print(f"Unique item IDs: {len(all_item_ids):,}", flush=True)
        build_item_text_cache(data_dir / "items.parquet", cache_path, all_item_ids)
    _barrier(distributed)
    if not cache_path.exists():
        raise FileNotFoundError(f"Item text cache was not created: {cache_path}")
    del all_item_ids

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model_path,
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    if args.gradient_accumulation_steps > 1:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = PairTextDataset(train_pairs, cache_path)
    val_dataset = PairTextDataset(val_pairs, cache_path)
    collator = PairCollator(tokenizer, args.max_length)

    validation_categories = val_pairs["category"].astype(str).to_numpy()

    def compute_metrics(prediction: EvalPrediction) -> dict[str, float]:
        return _metrics(prediction, validation_categories)

    trainer = Trainer(
        model=model,
        args=_training_arguments(args, output_dir),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    print("Starting cross-encoder fine-tuning...", flush=True)
    trainer.train()
    if _is_main_process(rank):
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
    _barrier(distributed)

    prediction = trainer.predict(val_dataset)
    logits = prediction.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    per_category = {}
    for category, group in val_pairs.groupby("category", sort=True):
        indices = group.index.to_numpy()
        per_category[str(category)] = float(
            average_precision_score(val_pairs.loc[indices, "target"], probabilities[indices])
        )
    if _is_main_process(rank):
        report = {
            "train_pairs": len(train_pairs),
            "val_pairs": len(val_pairs),
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "learning_rate": args.learning_rate,
            "world_size": world_size,
            "overall_average_precision": float(average_precision_score(val_pairs["target"], probabilities)),
            "macro_average_precision": float(np.mean(list(per_category.values()))),
            "per_category_average_precision": per_category,
            "total_seconds": time.perf_counter() - started,
        }
        (output_dir / "cross_encoder_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    _barrier(distributed)


if __name__ == "__main__":
    main()
