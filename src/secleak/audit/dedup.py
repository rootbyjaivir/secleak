"""Duplication, label-conflict and train/test-overlap statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sig_round(a: np.ndarray, sig: int) -> np.ndarray:
    """Round to ``sig`` significant figures (vectorised, zero-safe)."""
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.floor(np.log10(np.abs(a)))
    mag = np.where(np.isfinite(mag), mag, 0)
    dec = (sig - 1 - mag).astype(int)
    scale = 10.0 ** dec
    return np.where(a == 0, 0.0, np.round(a * scale) / scale)


def row_hashes(X: pd.DataFrame, sig: int | None = None) -> np.ndarray:
    """64-bit hash per row. ``sig`` -> near-duplicate key (significant-figure rounding)."""
    V = X.copy()
    num = V.select_dtypes(include=[np.number]).columns
    if sig is not None and len(num):
        V[num] = _sig_round(V[num].to_numpy(dtype=float), sig)
    V = V.fillna(-999999) if len(num) == len(V.columns) else V.astype(object).fillna("NA")
    return pd.util.hash_pandas_object(V, index=False).to_numpy()


def dup_report(X: pd.DataFrame, y: np.ndarray, near_sig: int = 4) -> dict:
    n = len(X)
    h = row_hashes(X)
    hn = row_hashes(X, sig=near_sig)
    s = pd.DataFrame({"h": h, "hn": hn, "y": y})
    exact_dup_rows = s.duplicated("h", keep=False).sum()
    exact_removable = s.duplicated("h", keep="first").sum()
    near_removable = s.duplicated("hn", keep="first").sum()
    g = s.groupby("h")["y"].nunique()
    conflict_keys = set(g[g > 1].index)
    conflict_rows = int(s["h"].isin(conflict_keys).sum())
    return {
        "n_rows": int(n),
        "exact_dup_rows_frac": exact_dup_rows / n,        # rows that have >=1 twin
        "exact_removable_frac": exact_removable / n,      # rows dropped by dedup(keep first)
        "near_removable_frac": near_removable / n,
        "label_conflict_groups": len(conflict_keys),
        "label_conflict_rows_frac": conflict_rows / n,
        "attack_frac": float(np.mean(y)),
    }


def dedup_mask(X: pd.DataFrame, y: np.ndarray, near_sig: int | None = None,
               drop_conflicts: bool = True) -> np.ndarray:
    """Boolean keep-mask: first occurrence only; optionally drop label-conflict groups."""
    h = row_hashes(X, sig=near_sig)
    s = pd.DataFrame({"h": h, "y": y})
    keep = ~s.duplicated("h", keep="first").to_numpy()
    if drop_conflicts:
        g = s.groupby("h")["y"].nunique()
        bad = set(g[g > 1].index)
        keep &= ~s["h"].isin(bad).to_numpy()
    return keep


def train_test_overlap(X: pd.DataFrame, tr: np.ndarray, te: np.ndarray,
                       y: np.ndarray | None = None, near_sig: int | None = None) -> dict:
    """Fraction of test rows whose (near-)identical feature vector appears in train.

    With ``y`` also returns how often the leaked twin carries the SAME label (i.e. the
    amount of accuracy a lookup table would get for free).
    """
    h = row_hashes(X, sig=near_sig)
    train_map = pd.Series(y[tr] if y is not None else 0, index=h[tr])
    train_map = train_map[~train_map.index.duplicated(keep="first")]
    hit = pd.Series(h[te]).isin(train_map.index).to_numpy()
    out = {"test_rows_in_train_frac": float(hit.mean())}
    if y is not None and hit.any():
        mapped = train_map.reindex(h[te][hit]).to_numpy()
        out["lookup_acc_on_hits"] = float((mapped == y[te][hit]).mean())
    return out
