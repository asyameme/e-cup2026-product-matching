from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def macro_average_precision(
    y_true: pd.Series | np.ndarray,
    y_score: pd.Series | np.ndarray,
    category: pd.Series | np.ndarray,
) -> tuple[float, pd.DataFrame]:
    """Compute the competition metric: mean AP over categories."""
    frame = pd.DataFrame(
        {
            "y_true": np.asarray(y_true, dtype=np.int8),
            "y_score": np.asarray(y_score, dtype=np.float64),
            "category": np.asarray(category),
        }
    )
    rows: list[dict] = []
    for cat, group in frame.groupby("category", sort=True):
        positives = int(group["y_true"].sum())
        ap = float(average_precision_score(group["y_true"], group["y_score"])) if positives else float("nan")
        rows.append(
            {
                "category": str(cat),
                "pairs": len(group),
                "positives": positives,
                "positive_rate": positives / len(group) if len(group) else float("nan"),
                "AP": ap,
            }
        )
    per_category = pd.DataFrame(rows)
    return float(per_category["AP"].mean()), per_category
