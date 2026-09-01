from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SEED = 42


class UnionFind:
    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int32)
        self.size = np.ones(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = int(self.parent[x])
        return x

    def union(self, x: int, y: int) -> None:
        x, y = self.find(x), self.find(y)
        if x == y:
            return
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]


def _component_labels(matches: pd.DataFrame) -> tuple[np.ndarray, dict[int, int]]:
    ids = np.unique(np.concatenate([matches["id1"].to_numpy(), matches["id2"].to_numpy()]))
    id_to_idx = {int(item_id): i for i, item_id in enumerate(ids)}
    uf = UnionFind(len(ids))
    for id1, id2 in zip(matches["id1"], matches["id2"]):
        uf.union(id_to_idx[int(id1)], id_to_idx[int(id2)])
    roots = np.array([uf.find(i) for i in range(len(ids))], dtype=np.int32)
    item_to_root = {int(item_id): int(roots[i]) for i, item_id in enumerate(ids)}
    return roots, item_to_root


def _choose_validation_components(
    matches: pd.DataFrame,
    item_to_root: dict[int, int],
    val_fraction: float,
    seed: int,
) -> set[int]:
    component_rows: dict[int, list[int]] = defaultdict(list)
    for row_index, (id1, id2) in enumerate(zip(matches["id1"], matches["id2"])):
        root1 = item_to_root[int(id1)]
        root2 = item_to_root[int(id2)]
        if root1 != root2:
            raise AssertionError("A pair has inconsistent component labels")
        component_rows[root1].append(row_index)

    categories = matches["category"].astype(str)
    component_info: dict[int, dict] = {}
    for root, row_indices in component_rows.items():
        group = matches.iloc[row_indices]
        group_categories = set(categories.iloc[row_indices])
        if len(group_categories) != 1:
            raise ValueError(f"Component {root} contains multiple categories: {group_categories}")
        component_info[root] = {
            "category": next(iter(group_categories)),
            "pairs": len(row_indices),
            "positives": int((group["target"].to_numpy() >= 0.5).sum()),
        }

    rng = np.random.default_rng(seed)
    by_category: dict[str, list[int]] = defaultdict(list)
    for root, info in component_info.items():
        by_category[info["category"]].append(root)

    selected: set[int] = set()
    for category in sorted(by_category):
        roots = np.asarray(by_category[category], dtype=np.int64)
        rng.shuffle(roots)
        category_pairs = sum(component_info[int(root)]["pairs"] for root in roots)
        target_pairs = int(round(category_pairs * val_fraction))
        current_pairs = 0
        for root_raw in roots:
            root = int(root_raw)
            component_pairs = component_info[root]["pairs"]
            if current_pairs >= target_pairs:
                break
            before = abs(current_pairs - target_pairs)
            after = abs(current_pairs + component_pairs - target_pairs)
            if current_pairs + component_pairs <= target_pairs or after < before:
                selected.add(root)
                current_pairs += component_pairs

        # The loop above is exact for the overwhelmingly common one-edge
        # components. This fallback closes any remaining gap with the closest
        # unselected component without changing the item-disjoint guarantee.
        if current_pairs != target_pairs:
            candidates = [
                int(root)
                for root in roots
                if int(root) not in selected
            ]
            if candidates:
                best = min(candidates, key=lambda root: abs(current_pairs + component_info[root]["pairs"] - target_pairs))
                if abs(current_pairs + component_info[best]["pairs"] - target_pairs) < abs(current_pairs - target_pairs):
                    selected.add(best)

    return selected


def split_matches(
    matches_path: Path,
    output_dir: Path,
    val_fraction: float = 0.20,
    seed: int = SEED,
    force: bool = False,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train_pairs.parquet"
    val_path = output_dir / "val_pairs.parquet"
    stats_path = output_dir / "split_stats.json"
    if not force and train_path.exists() and val_path.exists() and stats_path.exists():
        return json.loads(stats_path.read_text(encoding="utf-8"))

    started = time.perf_counter()
    matches = pq.read_table(matches_path, columns=["id1", "id2", "target"]).to_pandas()
    items_path = matches_path.parent / "items.parquet"
    items = pq.read_table(items_path, columns=["id", "category"]).to_pandas().set_index("id")
    matches["category"] = matches["id1"].map(items["category"]).fillna("<NULL>").astype(str)

    _, item_to_root = _component_labels(matches)
    val_components = _choose_validation_components(matches, item_to_root, val_fraction, seed)
    row_roots = matches["id1"].map(item_to_root).to_numpy()
    val_mask = np.isin(row_roots, np.fromiter(val_components, dtype=np.int32))
    val = matches.loc[val_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    train = matches.loc[~val_mask, ["id1", "id2", "target", "category"]].reset_index(drop=True)
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)

    train_items = set(train["id1"]).union(train["id2"])
    val_items = set(val["id1"]).union(val["id2"])
    intersection = train_items.intersection(val_items)
    if intersection:
        raise AssertionError(f"Item-disjoint split failed: {len(intersection)} overlapping items")

    def split_stats(frame: pd.DataFrame) -> dict:
        return {
            "pairs": len(frame),
            "positives": int((frame["target"] >= 0.5).sum()),
            "positive_rate": float((frame["target"] >= 0.5).mean()),
            "categories": int(frame["category"].nunique()),
            "unique_items": len(set(frame["id1"]).union(frame["id2"])),
        }

    stats = {
        "seed": seed,
        "val_fraction_requested": val_fraction,
        "method": "connected_components_stratified_by_category",
        "train": split_stats(train),
        "validation": split_stats(val),
        "train_item_ids_disjoint_val_item_ids": len(intersection) == 0,
        "item_overlap_count": len(intersection),
        "elapsed_seconds": time.perf_counter() - started,
    }
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
