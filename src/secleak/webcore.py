"""UI-independent core of the web app: audit an arbitrary uploaded flow table.

Kept free of any web-framework import so it can be unit-tested and reused from notebooks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit.card import build_card, card_markdown
from .data import frame_to_dataset

LABEL_HINTS = ("label", "attack_cat", "class", "target", "attack", "category")
TIME_HINTS = ("timestamp", "time", "stime", "ltime", "date", "ts")
BENIGN_HINTS = {"benign", "normal", "0", "legitimate", "background"}


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def guess_label_col(df: pd.DataFrame) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for h in LABEL_HINTS:
        if h in low:
            return low[h]
    return None


def guess_time_col(df: pd.DataFrame) -> str | None:
    low = {c.lower(): c for c in df.columns}
    for h in TIME_HINTS:
        if h in low:
            return low[h]
    return None


def guess_benign_values(df: pd.DataFrame, label_col: str) -> list[str]:
    vals = df[label_col].dropna().astype(str).str.strip().unique().tolist()
    hit = [v for v in vals if v.lower() in BENIGN_HINTS]
    return hit or [sorted(vals, key=lambda v: -(df[label_col].astype(str).str.strip() == v).sum())[0]]


def audit_frame(df: pd.DataFrame, label_col: str | None = None, benign_values: list[str] | None = None,
                time_col: str | None = None, max_rows: int = 200_000, name: str = "uploaded") -> dict:
    """Return {'card': dict, 'markdown': str, 'warnings': [str]} for an uploaded table."""
    df = clean_columns(df)
    warnings: list[str] = []
    label_col = label_col or guess_label_col(df)
    if label_col is None or label_col not in df.columns:
        raise ValueError("Could not find a label column; pass label_col explicitly.")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    if len(df) > max_rows:
        idx = np.sort(np.random.default_rng(0).choice(len(df), max_rows, replace=False))
        df = df.iloc[idx].reset_index(drop=True)
        warnings.append(f"Subsampled to {max_rows:,} rows (time order preserved); duplicate "
                        "statistics on a subsample understate full-data duplication.")
    benign = benign_values or guess_benign_values(df, label_col)
    time_col = time_col or guess_time_col(df)
    cfg = dict(label_col=label_col, benign_values=benign, drop_cols=[])
    if time_col and time_col in df.columns:
        parsed = pd.to_datetime(df[time_col], errors="coerce")
        if parsed.notna().mean() > 0.9:
            cfg.update(time_col=time_col, time_kind="iso", drop_cols=[time_col])
            df[time_col] = parsed
        else:
            warnings.append(f"Column {time_col!r} did not parse as timestamps; using row order as a time proxy.")
    if "time_kind" not in cfg:
        cfg["time_kind"] = "file_order"
        df["__file_idx"], df["__row_idx"] = 0, np.arange(len(df))
        warnings.append("No usable timestamps: temporal results use ROW ORDER as a proxy.")
    # numeric features only; text columns (IPs, categories) are dropped, not encoded
    keep = [c for c in df.columns if c in (label_col, "__file_idx", "__row_idx", cfg.get("time_col"))
            or pd.api.types.is_numeric_dtype(df[c])]
    dropped = [c for c in df.columns if c not in keep]
    if dropped:
        warnings.append(f"Dropped {len(dropped)} non-numeric columns: {', '.join(map(str, dropped[:8]))}"
                        + (" ..." if len(dropped) > 8 else ""))
    ds = frame_to_dataset(df[keep], name, cfg)
    if len(np.unique(ds.y)) < 2:
        raise ValueError(f"Only one class after mapping benign values {benign}; adjust the benign label(s).")
    card = build_card(ds)
    return {"card": card, "markdown": card_markdown(card), "warnings": warnings,
            "label_col": label_col, "benign_values": benign, "time_col": cfg.get("time_col")}
