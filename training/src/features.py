from __future__ import annotations

import json
import re
import sqlite3
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from eda_attributes import parse_attributes  # noqa: E402
from eda_matching import code_tokens, name_similarity, token_jaccard  # noqa: E402


NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)?(?:[a-zа-я]+)?(?!\w)", re.I)


def clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip()


def compact_text(value: object) -> str:
    return re.sub(r"[^0-9a-zа-я]+", "", clean_text(value))


def number_tokens(text: str) -> set[str]:
    return {clean_text(m.group(0)) for m in NUMBER_RE.finditer(text)}


def _item_record(row: object) -> dict:
    name = "" if row.name is None else str(row.name)
    attrs = parse_attributes(row.attributes)
    full_text = " ".join([name, *attrs.values()])
    return {
        "id": int(row.id),
        "category": str(row.category) if row.category is not None else "<NULL>",
        "name": name,
        "name_clean": clean_text(name),
        "name_compact": compact_text(name),
        "name_len": len(name),
        "attrs": attrs,
        "numbers": number_tokens(full_text),
        "codes": code_tokens(full_text),
    }


def load_item_lookup(
    items_path: Path,
    batch_size: int = 100_000,
    item_ids: set[int] | None = None,
) -> dict[int, dict]:
    lookup: dict[int, dict] = {}
    parquet = pq.ParquetFile(items_path)
    for batch in parquet.iter_batches(columns=["id", "name", "attributes", "category"], batch_size=batch_size):
        frame = batch.to_pandas()
        for row in frame.itertuples(index=False):
            item_id = int(row.id)
            if item_ids is not None and item_id not in item_ids:
                continue
            lookup[item_id] = _item_record(row)
    return lookup


def _cache_record(record: dict) -> str:
    """Serialize one item record without keeping derived records in RAM."""
    serializable = dict(record)
    serializable["numbers"] = sorted(record["numbers"])
    serializable["codes"] = sorted(record["codes"])
    return json.dumps(serializable, ensure_ascii=False, separators=(",", ":"))


def _uncache_record(payload: str) -> dict:
    record = json.loads(payload)
    record["numbers"] = set(record["numbers"])
    record["codes"] = set(record["codes"])
    return record


def _item_cache_signature(items_path: Path, item_ids: set[int]) -> dict:
    return {
        "source_size": items_path.stat().st_size,
        "source_mtime_ns": items_path.stat().st_mtime_ns,
        "item_count": len(item_ids),
        "item_sum": int(sum(item_ids)),
        "item_min": int(min(item_ids)) if item_ids else None,
        "item_max": int(max(item_ids)) if item_ids else None,
    }


def build_item_cache(
    items_path: Path,
    cache_path: Path,
    item_ids: set[int],
    batch_size: int = 10_000,
) -> None:
    """Create a disk-backed cache for only the items used by train/validation.

    The previous implementation kept every parsed item record in a Python
    dictionary. For a multi-million-row items file that can exceed RAM before
    CatBoost starts. SQLite keeps the records on disk and lets feature
    generation load only the IDs in the current pair chunk.
    """
    if not item_ids:
        raise ValueError("Cannot build an item cache without item IDs")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_path.with_suffix(cache_path.suffix + ".json")
    signature = _item_cache_signature(items_path, item_ids)
    if cache_path.exists() and manifest_path.exists():
        try:
            if json.loads(manifest_path.read_text(encoding="utf-8")) == signature:
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
            CREATE TABLE item_records (
                id INTEGER PRIMARY KEY,
                record_json TEXT NOT NULL
        ) WITHOUT ROWID;
            """
        )
        parquet = pq.ParquetFile(items_path)
        processed_rows = 0
        cached_rows = 0
        for batch_index, batch in enumerate(parquet.iter_batches(
            columns=["id", "name", "attributes", "category"],
            batch_size=batch_size,
        ), start=1):
            frame = batch.to_pandas()
            rows = []
            for row in frame.itertuples(index=False):
                item_id = int(row.id)
                if item_id in item_ids:
                    rows.append((item_id, _cache_record(_item_record(row))))
            if rows:
                connection.executemany(
                    "INSERT INTO item_records (id, record_json) VALUES (?, ?)",
                    rows,
                )
                connection.commit()
                cached_rows += len(rows)
            processed_rows += len(frame)
            if batch_index == 1 or batch_index % 100 == 0:
                print(
                    f"item cache: {processed_rows:,}/{parquet.metadata.num_rows:,} rows scanned, "
                    f"{cached_rows:,} records cached",
                    flush=True,
                )
            del frame, rows
        connection.commit()
    finally:
        connection.close()
    manifest_path.write_text(json.dumps(signature, indent=2), encoding="utf-8")


def load_item_lookup_from_cache(
    cache_path: Path,
    item_ids: Iterable[int],
    query_batch_size: int = 900,
) -> dict[int, dict]:
    """Load only the item records needed by one feature chunk."""
    ids = list({int(item_id) for item_id in item_ids})
    lookup: dict[int, dict] = {}
    connection = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        for start in range(0, len(ids), query_batch_size):
            batch_ids = ids[start : start + query_batch_size]
            placeholders = ",".join("?" for _ in batch_ids)
            rows = connection.execute(
                f"SELECT id, record_json FROM item_records WHERE id IN ({placeholders})",
                batch_ids,
            )
            lookup.update((int(item_id), _uncache_record(payload)) for item_id, payload in rows)
    finally:
        connection.close()
    missing = set(ids) - set(lookup)
    if missing:
        sample = sorted(missing)[:10]
        raise KeyError(f"Item cache is missing {len(missing)} IDs, examples: {sample}")
    return lookup


def load_selected_attributes(stats_path: Path, min_support: int) -> tuple[list[dict], dict[str, list[dict]]]:
    stats = pd.read_csv(stats_path)
    selected = stats[stats["pairs_with_attribute_both"] >= min_support].copy()
    selected = selected.sort_values(["category", "attribute"]).drop_duplicates(["category", "attribute"])
    specs: list[dict] = []
    by_category: dict[str, list[dict]] = {}
    for row in selected.itertuples(index=False):
        category = str(row.category)
        attribute = str(row.attribute)
        safe_category = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", category).strip("_")
        safe_attribute = re.sub(r"[^0-9A-Za-zА-Яа-я]+", "_", attribute).strip("_")
        spec = {
            "category": category,
            "attribute": attribute,
            "feature": f"attr_state__{safe_category}__{safe_attribute}",
            "support": int(row.pairs_with_attribute_both),
        }
        specs.append(spec)
        by_category.setdefault(category, []).append(spec)
    return specs, by_category


def _state(left: dict, right: dict, attribute: str) -> int:
    if attribute not in left["attrs"] or attribute not in right["attrs"]:
        return 0
    return 1 if left["attrs"][attribute] == right["attrs"][attribute] else 2


def extract_pair_features(
    pairs: pd.DataFrame,
    lookup: dict[int, dict],
    selected_by_category: dict[str, list[dict]] | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    selected_by_category = selected_by_category or {}
    for row in pairs.itertuples(index=False):
        left = lookup[int(row.id1)]
        right = lookup[int(row.id2)]
        common_keys = set(left["attrs"]).intersection(right["attrs"])
        equal_values = sum(left["attrs"][key] == right["attrs"][key] for key in common_keys)
        different_values = len(common_keys) - equal_values
        number_intersection = left["numbers"].intersection(right["numbers"])
        code_intersection = left["codes"].intersection(right["codes"])
        numbers_union = left["numbers"].union(right["numbers"])
        name_a, name_b = left["name_clean"], right["name_clean"]
        row_features = {
            "id1": int(row.id1),
            "id2": int(row.id2),
            "target": int(float(row.target) >= 0.5),
            "category": left["category"],
            "name_exact": int(left["name_compact"] == right["name_compact"]),
            "name_similarity": name_similarity(left["name"], right["name"]),
            "name_token_jaccard": token_jaccard(left["name"], right["name"]),
            "name_length_a": left["name_len"],
            "name_length_b": right["name_len"],
            "name_length_diff": abs(left["name_len"] - right["name_len"]),
            "number_match": int(bool(number_intersection)),
            "number_conflict": int(bool(left["numbers"] and right["numbers"] and not number_intersection)),
            "numbers_a_count": len(left["numbers"]),
            "numbers_b_count": len(right["numbers"]),
            "numbers_intersection_count": len(number_intersection),
            "numbers_union_count": len(numbers_union),
            "numbers_jaccard": len(number_intersection) / len(numbers_union) if numbers_union else 0.0,
            "model_token_match": int(bool(code_intersection)),
            "model_token_conflict": int(bool(left["codes"] and right["codes"] and not code_intersection)),
            "model_tokens_a_count": len(left["codes"]),
            "model_tokens_b_count": len(right["codes"]),
            "model_token_intersection_count": len(code_intersection),
            "common_attribute_keys": len(common_keys),
            "equal_attribute_values": equal_values,
            "different_attribute_values": different_values,
            "equal_value_fraction": equal_values / len(common_keys) if common_keys else 0.0,
            "different_value_fraction": different_values / len(common_keys) if common_keys else 0.0,
        }
        for spec in selected_by_category.get(left["category"], []):
            row_features[spec["feature"]] = _state(left, right, spec["attribute"])
        rows.append(row_features)
    return pd.DataFrame(rows)


AGGREGATE_FEATURES = [
    "category",
    "name_exact",
    "name_similarity",
    "name_token_jaccard",
    "name_length_a",
    "name_length_b",
    "name_length_diff",
    "number_match",
    "number_conflict",
    "numbers_a_count",
    "numbers_b_count",
    "numbers_intersection_count",
    "numbers_union_count",
    "numbers_jaccard",
    "model_token_match",
    "model_token_conflict",
    "model_tokens_a_count",
    "model_tokens_b_count",
    "model_token_intersection_count",
    "common_attribute_keys",
    "equal_attribute_values",
    "different_attribute_values",
    "equal_value_fraction",
    "different_value_fraction",
]
