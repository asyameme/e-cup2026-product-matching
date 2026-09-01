"""Build a time-bounded joint dataset for cross-encoder fine-tuning.

The output contains:
  * a balanced sample of confident LLM pairs (soft targets are preserved),
  * human training pairs,
  * optional hard negatives mined by the current model,
  * one reversed copy of every selected pair.

Validation objects are excluded from every training source to keep the
existing connected-component split leakage-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42


def _category_map(items_path: Path) -> dict[int, str]:
    items = pd.read_parquet(items_path, columns=["id", "category"])
    return {
        int(item_id): "<NULL>" if pd.isna(category) else str(category)
        for item_id, category in items.itertuples(index=False, name=None)
    }


def _add_category(pairs: pd.DataFrame, categories: dict[int, str]) -> pd.DataFrame:
    result = pairs.copy()
    result["category"] = result["id1"].map(categories)
    missing = result["category"].isna()
    if missing.any():
        result.loc[missing, "category"] = result.loc[missing, "id2"].map(categories)
    result["category"] = result["category"].fillna("<NULL>").astype(str)
    return result


def _balanced_sample(
    frame: pd.DataFrame,
    count: int,
    category_column: str = "category",
    seed: int = SEED,
) -> pd.DataFrame:
    """Sample exactly count rows while favouring equal category quotas.

    If a category has fewer rows than its equal quota, its available rows are
    used and the remainder is distributed among categories with capacity.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return frame.iloc[:0].copy()
    if len(frame) < count:
        raise ValueError(
            f"Cannot sample {count:,} rows from {len(frame):,} available rows"
        )

    grouped = {
        str(category): group
        for category, group in frame.groupby(category_column, sort=True, observed=True)
    }
    names = sorted(grouped)
    if not names:
        raise ValueError("No categories available for sampling")

    capacities = {name: len(grouped[name]) for name in names}
    base = count // len(names)
    quotas = {name: min(capacities[name], base) for name in names}
    remaining = count - sum(quotas.values())

    # Fill the remainder from categories with the most unused capacity. This
    # keeps the result close to balanced even when category sizes differ.
    while remaining:
        eligible = [name for name in names if quotas[name] < capacities[name]]
        if not eligible:
            raise RuntimeError("Quota allocation failed despite enough rows")
        eligible.sort(key=lambda name: (capacities[name] - quotas[name], name), reverse=True)
        for name in eligible:
            if remaining == 0:
                break
            quotas[name] += 1
            remaining -= 1

    pieces = []
    for offset, name in enumerate(names):
        quota = quotas[name]
        if quota:
            pieces.append(
                grouped[name].sample(n=quota, random_state=seed + offset)
            )
    return (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _reverse_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    reversed_pairs = pairs.copy()
    reversed_pairs["id1"], reversed_pairs["id2"] = (
        pairs["id2"].to_numpy(copy=True),
        pairs["id1"].to_numpy(copy=True),
    )
    return reversed_pairs


def _read_hard_negatives(
    score_path: Path,
    human_train: pd.DataFrame,
    per_category: int,
    seed: int,
) -> pd.DataFrame:
    # The scorer output also contains target. Read only the pair IDs and the
    # model score so the merge below keeps the human target column name.
    scored = pd.read_parquet(
        score_path,
        columns=["id1", "id2", "cross_encoder_score"],
    )
    required = {"id1", "id2", "cross_encoder_score"}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Hard-negative score file is missing: {sorted(missing)}")

    scored = scored.merge(
        human_train[["id1", "id2", "target", "category"]],
        on=["id1", "id2"],
        how="inner",
        validate="one_to_one",
    )
    # A hard negative is a known human negative that the current model ranks
    # highly. It is duplicated deliberately to increase its training weight.
    negatives = scored[scored["target"] <= 0.0].copy()
    if negatives.empty:
        raise ValueError("No human negative pairs found in score file")
    hard = (
        negatives.sort_values("cross_encoder_score", ascending=False)
        .groupby("category", sort=True, group_keys=False, observed=True)
        .head(per_category)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return hard[["id1", "id2", "target", "category"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items-path", required=True)
    parser.add_argument("--human-train-path", required=True)
    parser.add_argument("--human-val-path", required=True)
    parser.add_argument("--llm-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--hard-negative-score-path", default=None)
    parser.add_argument("--llm-high-count", type=int, default=500_000)
    parser.add_argument("--llm-low-count", type=int, default=1_500_000)
    parser.add_argument(
        "--human-repeat",
        type=int,
        default=4,
        help="How many times to repeat the human training pairs before reversal.",
    )
    parser.add_argument("--hard-per-category", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    if args.human_repeat < 1:
        raise ValueError("--human-repeat must be at least 1")

    items_path = Path(args.items_path)
    human_train_path = Path(args.human_train_path)
    human_val_path = Path(args.human_val_path)
    llm_path = Path(args.llm_path)
    output_path = Path(args.output_path)

    categories = _category_map(items_path)
    human_train = _add_category(
        pd.read_parquet(human_train_path, columns=["id1", "id2", "target"]),
        categories,
    )
    human_val = pd.read_parquet(human_val_path, columns=["id1", "id2"])
    val_ids = set(human_val["id1"].astype("int64"))
    val_ids.update(human_val["id2"].astype("int64"))

    llm = _add_category(
        pd.read_parquet(llm_path, columns=["id1", "id2", "target"]),
        categories,
    )
    llm = llm[
        ((llm["target"] >= 0.7) | (llm["target"] <= 0.3))
        & ~llm["id1"].isin(val_ids)
        & ~llm["id2"].isin(val_ids)
    ].copy()

    high = _balanced_sample(
        llm[llm["target"] >= 0.7], args.llm_high_count, seed=args.seed
    )
    low = _balanced_sample(
        llm[llm["target"] <= 0.3], args.llm_low_count, seed=args.seed + 1
    )
    llm_selected = pd.concat([high, low], ignore_index=True)

    hard = pd.DataFrame(columns=["id1", "id2", "target", "category"])
    if args.hard_negative_score_path:
        hard = _read_hard_negatives(
            Path(args.hard_negative_score_path),
            human_train,
            args.hard_per_category,
            args.seed,
        )

    # Human labels are the closest proxy to the competition metric. Repeat
    # them deliberately so a fixed max_steps budget spends more updates on
    # human supervision while retaining LLM and hard-negative sources.
    human_effective = pd.concat(
        [human_train] * args.human_repeat,
        ignore_index=True,
    )

    sources = {
        "llm": llm_selected,
        "human": human_effective,
        "hard": hard,
    }
    pieces = []
    for frame in sources.values():
        if len(frame):
            pieces.extend([frame, _reverse_pairs(frame)])
    joint = (
        pd.concat(pieces, ignore_index=True)
        .sample(frac=1.0, random_state=args.seed)
        .reset_index(drop=True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joint[["id1", "id2", "target", "category"]].to_parquet(
        output_path, index=False, compression="zstd"
    )

    summary = {
        "output": str(output_path),
        "rows": len(joint),
        "llm_high": len(high),
        "llm_low": len(low),
        "human": len(human_train),
        "human_repeat": args.human_repeat,
        "human_effective": len(human_effective),
        "hard": len(hard),
        "reversed": True,
        "validation_ids_excluded_from_llm": True,
        "target_counts": {
            str(key): int(value)
            for key, value in joint["target"].value_counts().sort_index().items()
        },
        "category_counts": {
            str(key): int(value)
            for key, value in joint["category"].value_counts().sort_index().items()
        },
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
