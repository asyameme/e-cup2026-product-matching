from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catboost import CatBoostClassifier, Pool  # noqa: E402

from src.features import (  # noqa: E402
    AGGREGATE_FEATURES,
    build_item_cache,
    extract_pair_features,
    load_item_lookup_from_cache,
    load_selected_attributes,
)
from src.metrics import macro_average_precision  # noqa: E402
from src.split import SEED, split_matches  # noqa: E402


DEFAULT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "PRAUC",
    "iterations": 1800,
    "learning_rate": 0.05,
    "depth": 8,
    "random_seed": SEED,
    "l2_leaf_reg": 5.0,
    "random_strength": 0.5,
    "od_type": "Iter",
    "od_wait": 120,
    # Keep CatBoost's internal quantization/CTR buffers bounded on Kaggle.
    "thread_count": 2,
    "used_ram_limit": "20Gb",
    "max_ctr_complexity": 1,
    "verbose": 200,
    "allow_writing_files": False,
}

FEATURE_CHUNK_SIZE = 10_000


def _write_features_in_chunks(
    pairs: pd.DataFrame,
    item_cache_path: Path,
    selected_by_category: dict[str, list[dict]],
    all_feature_names: list[str],
    output_path: Path,
    chunk_size: int,
) -> list[str]:
    """Generate features in bounded-memory chunks and persist them to Parquet."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.unlink(missing_ok=True)
    writer = None
    output_feature_names: list[str] | None = None
    try:
        for start in range(0, len(pairs), chunk_size):
            chunk_pairs = pairs.iloc[start : start + chunk_size]
            chunk_item_ids = set(chunk_pairs["id1"].astype(int)) | set(chunk_pairs["id2"].astype(int))
            chunk_lookup = load_item_lookup_from_cache(item_cache_path, chunk_item_ids)
            chunk_features = extract_pair_features(chunk_pairs, chunk_lookup, selected_by_category)
            for feature in all_feature_names:
                if feature not in chunk_features:
                    chunk_features[feature] = np.int8(0)
            chunk_features = chunk_features[["id1", "id2", "target"] + all_feature_names]
            chunk_features["id1"] = chunk_features["id1"].astype(np.int64)
            chunk_features["id2"] = chunk_features["id2"].astype(np.int64)
            chunk_features["target"] = chunk_features["target"].astype(np.int8)
            numeric_columns = [
                column for column in chunk_features.columns
                if column not in {"id1", "id2", "target", "category"}
            ]
            if numeric_columns:
                chunk_features[numeric_columns] = chunk_features[numeric_columns].astype(np.float32)

            table = pa.Table.from_pandas(chunk_features, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                output_feature_names = list(chunk_features.columns)
            writer.write_table(table)
            chunk_index = start // chunk_size + 1
            if chunk_index == 1 or chunk_index % 100 == 0:
                print(
                    f"{output_path.name}: {min(start + len(chunk_pairs), len(pairs)):,}/{len(pairs):,} pairs processed",
                    flush=True,
                )
            del chunk_pairs, chunk_item_ids, chunk_lookup, chunk_features, table
            gc.collect()
    except Exception:
        if writer is not None:
            writer.close()
            writer = None
        output_path.unlink(missing_ok=True)
        raise
    finally:
        if writer is not None:
            writer.close()

    if output_feature_names is None:
        raise ValueError(f"No pairs were available for feature generation: {output_path}")
    return output_feature_names


def _train_one(
    name: str,
    train_feature_path: Path,
    val_feature_path: Path,
    feature_names: list[str],
    cat_features: list[str],
    model_path: Path,
    report_dir: Path,
    params: dict,
    item_cache_path: Path,
) -> dict:
    started = time.perf_counter()
    train_df = pd.read_parquet(train_feature_path, columns=feature_names + ["target"])
    X_train = train_df[feature_names]
    y_train = train_df["target"].to_numpy(dtype=np.int8)
    cat_indices = [feature_names.index(name) for name in cat_features]
    train_pool = Pool(X_train, y_train, cat_features=cat_indices, feature_names=feature_names)
    del train_df, X_train, y_train
    gc.collect()

    # Do not keep train and validation DataFrames in RAM at the same time.
    val_df = pd.read_parquet(val_feature_path, columns=feature_names + ["target", "id1", "id2"])
    X_val = val_df[feature_names]
    y_val = val_df["target"].to_numpy(dtype=np.int8)
    val_pool = Pool(X_val, y_val, cat_features=cat_indices, feature_names=feature_names)
    val_meta = val_df[["id1", "id2", "category", "target"]].copy()
    del val_df, X_val
    gc.collect()
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)
    training_seconds = time.perf_counter() - started
    model.save_model(model_path)

    inference_started = time.perf_counter()
    scores = model.predict_proba(val_pool)[:, 1]
    inference_seconds = time.perf_counter() - inference_started
    macro_ap, per_category = macro_average_precision(y_val, scores, val_meta["category"])
    per_category.insert(0, "experiment", name)
    per_category.to_csv(report_dir / f"{name}_category_metrics.csv", index=False)

    importance = pd.DataFrame(
        {
            "experiment": name,
            "feature": feature_names,
            "importance": model.get_feature_importance(val_pool),
        }
    ).sort_values("importance", ascending=False)
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(report_dir / f"{name}_feature_importance.csv", index=False)

    errors = val_meta.copy()
    errors["prediction"] = scores

    # Build detailed JSON only for the 100 hardest examples of each type.
    # Materializing attributes for all validation pairs causes a large RAM spike.
    hard_fp = errors[errors["target"] == 0].nlargest(100, "prediction").copy()
    hard_fn = errors[errors["target"] == 1].nsmallest(100, "prediction").copy()
    hard_ids = set(hard_fp["id1"].astype(int)) | set(hard_fp["id2"].astype(int))
    hard_ids.update(hard_fn["id1"].astype(int))
    hard_ids.update(hard_fn["id2"].astype(int))
    report_lookup = load_item_lookup_from_cache(item_cache_path, hard_ids)
    for frame in (hard_fp, hard_fn):
        frame["name1"] = frame["id1"].map(lambda item_id: report_lookup[int(item_id)]["name"])
        frame["name2"] = frame["id2"].map(lambda item_id: report_lookup[int(item_id)]["name"])
        frame["attributes1"] = frame["id1"].map(
            lambda item_id: json.dumps(report_lookup[int(item_id)]["attrs"], ensure_ascii=False, sort_keys=True)
        )
        frame["attributes2"] = frame["id2"].map(
            lambda item_id: json.dumps(report_lookup[int(item_id)]["attrs"], ensure_ascii=False, sort_keys=True)
        )
    hard_fp.to_csv(report_dir / f"{name}_hard_false_positives.csv", index=False)
    hard_fn.to_csv(report_dir / f"{name}_hard_false_negatives.csv", index=False)

    best_iteration = int(model.get_best_iteration())
    top_features = importance.head(50).to_dict(orient="records")
    per_category_records = per_category.to_dict(orient="records")
    validation_pairs = len(val_meta)
    del train_pool, val_pool, model, importance, errors, hard_fp, hard_fn, report_lookup, hard_ids, val_meta
    gc.collect()
    return {
        "experiment": name,
        "macro_ap": macro_ap,
        "best_iteration": best_iteration,
        "training_seconds": training_seconds,
        "validation_inference_seconds": inference_seconds,
        "validation_pairs_per_second": validation_pairs / inference_seconds if inference_seconds else None,
        "model_path": str(model_path),
        "top_features": top_features,
        "per_category": per_category_records,
    }, scores


def _baseline_report(val_df: pd.DataFrame, report_dir: Path) -> list[dict]:
    rows = []
    scores = {
        "name_similarity": val_df["name_similarity"].to_numpy(dtype=float),
        "model_token_heuristic": (
            val_df["model_token_match"].to_numpy(dtype=float)
            - val_df["model_token_conflict"].to_numpy(dtype=float)
        ),
    }
    for name, score in scores.items():
        macro_ap, per_category = macro_average_precision(val_df["target"], score, val_df["category"])
        per_category.insert(0, "experiment", name)
        per_category.to_csv(report_dir / f"{name}_category_metrics.csv", index=False)
        rows.append({"experiment": name, "macro_ap": macro_ap, "per_category": per_category.to_dict(orient="records")})
    return rows


def run(args: argparse.Namespace) -> dict:
    overall_started = time.perf_counter()
    data_dir = Path(args.data_dir)
    artifact_dir = Path(args.artifact_dir)
    split_dir = artifact_dir / "splits"
    feature_dir = artifact_dir / "features"
    model_dir = artifact_dir / "models"
    report_dir = artifact_dir / "reports"
    for directory in (split_dir, feature_dir, model_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_stats = split_matches(data_dir / "matches.parquet", split_dir, args.val_fraction, SEED, args.force_split)
    pair_columns = ["id1", "id2", "target"]
    train_pairs = pd.read_parquet(split_dir / "train_pairs.parquet", columns=pair_columns)
    val_pairs = pd.read_parquet(split_dir / "val_pairs.parquet", columns=pair_columns)

    feature_started = time.perf_counter()
    needed_item_ids = set(train_pairs["id1"].astype(int)) | set(train_pairs["id2"].astype(int))
    needed_item_ids.update(val_pairs["id1"].astype(int))
    needed_item_ids.update(val_pairs["id2"].astype(int))
    item_cache_path = feature_dir / "items.sqlite"
    print(f"Building disk-backed item cache for {len(needed_item_ids):,} item IDs...", flush=True)
    build_item_cache(data_dir / "items.parquet", item_cache_path, needed_item_ids)
    print(f"Item cache ready: {item_cache_path}", flush=True)
    del needed_item_ids
    selected_specs, selected_by_category = load_selected_attributes(
        Path(args.attribute_stats_path), args.min_attribute_support
    )
    aggregate_features = AGGREGATE_FEATURES
    attribute_features = [spec["feature"] for spec in selected_specs]
    model_features = aggregate_features + attribute_features
    train_feature_path = feature_dir / "train_attributes.parquet"
    val_feature_path = feature_dir / "val_attributes.parquet"
    _write_features_in_chunks(
        train_pairs,
        item_cache_path,
        selected_by_category,
        model_features,
        train_feature_path,
        args.feature_chunk_size,
    )
    del train_pairs
    gc.collect()
    _write_features_in_chunks(
        val_pairs,
        item_cache_path,
        selected_by_category,
        model_features,
        val_feature_path,
        args.feature_chunk_size,
    )
    del val_pairs, selected_by_category
    gc.collect()

    (feature_dir / "attribute_feature_mapping.json").write_text(
        json.dumps(selected_specs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    feature_seconds = time.perf_counter() - feature_started

    # Load only the validation columns needed for the lightweight baselines;
    # _train_one loads the full cache and releases DataFrames after Pool creation.
    baseline_val = pd.read_parquet(
        val_feature_path,
        columns=aggregate_features + ["target"],
    )
    results = _baseline_report(baseline_val, report_dir)
    del baseline_val
    gc.collect()
    experiment, scores = _train_one(
        "attributes",
        train_feature_path,
        val_feature_path,
        model_features,
        ["category"],
        model_dir / "catboost_attributes.cbm",
        report_dir,
        DEFAULT_PARAMS,
        item_cache_path,
    )
    del scores
    gc.collect()
    results.append(experiment)
    report = {
        "seed": SEED,
        "split": split_stats,
        "feature_generation_seconds": feature_seconds,
        "aggregate_feature_count": len(aggregate_features),
        "attribute_specific_feature_count": len(attribute_features),
        "min_attribute_support": args.min_attribute_support,
        "catboost_params": DEFAULT_PARAMS,
        "results": results,
        "total_seconds": time.perf_counter() - overall_started,
    }
    (report_dir / "catboost_experiment.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    pd.DataFrame(
        [{"experiment": result["experiment"], "macro_ap": result["macro_ap"]} for result in results]
    ).to_csv(report_dir / "experiment_summary.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=float))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--artifact_dir", default="artifacts/catboost")
    parser.add_argument("--attribute_stats_path", required=True)
    parser.add_argument("--min_attribute_support", type=int, default=200)
    parser.add_argument("--val_fraction", type=float, default=0.20)
    parser.add_argument("--feature_chunk_size", type=int, default=FEATURE_CHUNK_SIZE)
    parser.add_argument("--force_split", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
