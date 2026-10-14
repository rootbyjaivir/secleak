"""Single-feature shortcut detection.

For each feature we fit a depth-limited decision tree on that ONE feature and report
held-out F1. A feature that alone reaches near-perfect F1 is a shortcut; comparing the
random-split score with the temporal-split score shows whether it survives distribution shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

from ..protocols import split


def shortcut_scan(ds, protocols=("random", "temporal_per_class"), max_depth=3,
                  seed=0, top_k=15, max_rows=200_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(ds))
    if len(idx) > max_rows:
        idx = np.sort(rng.choice(idx, max_rows, replace=False))
    num_cols = [c for c in ds.X.columns if pd.api.types.is_numeric_dtype(ds.X[c])]
    splits = {p: split(ds, p, seed=seed, idx=idx) for p in protocols}
    rows = []
    for c in num_cols:
        x = ds.X[c].to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=-1e9, posinf=1e9, neginf=-1e9).reshape(-1, 1)
        rec = {"feature": c}
        for p, (tr, te) in splits.items():
            t = DecisionTreeClassifier(max_depth=max_depth, random_state=seed).fit(x[tr], ds.y[tr])
            rec[f"f1_{p}"] = f1_score(ds.y[te], t.predict(x[te]), zero_division=0)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(f"f1_{protocols[0]}", ascending=False)
    return out.head(top_k).reset_index(drop=True)
