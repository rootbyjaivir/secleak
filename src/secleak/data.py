"""Config-driven dataset loading -> a uniform ``Dataset`` object."""
from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "datasets.yaml"


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame                 # feature columns only
    y: np.ndarray                   # 1 = attack, 0 = benign
    time: np.ndarray                # float seconds (or proxy ordinal); length n
    time_is_proxy: bool = False     # True if no real timestamps exist
    meta: dict = field(default_factory=dict)
    labels: np.ndarray | None = None  # raw (stripped) label strings, e.g. "DoS Hulk"; for diagnostics

    def __len__(self) -> int:
        return len(self.y)


def load_config(path: str | os.PathLike | None = None) -> dict:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def _parse_time(series: pd.Series, kind: str) -> np.ndarray:
    if kind == "epoch_s":
        return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    if kind == "cicids_dmy":
        # Formats differ PER FILE (e.g. Monday has seconds, others do not), so parse per element:
        # try with seconds first, then fall back to minutes only for the rows that failed.
        ts = pd.to_datetime(series, format="%d/%m/%Y %H:%M:%S", errors="coerce")
        miss = ts.isna()
        if miss.any():
            ts[miss] = pd.to_datetime(series[miss], format="%d/%m/%Y %H:%M", errors="coerce")
        # HEURISTIC: the export has no AM/PM; captures ran ~09:00-17:00, so hours < 8
        # are afternoon. Verify against the pcaps before relying on within-day order.
        fix = ts.dt.hour < 8
        ts = ts.where(~fix, ts + pd.Timedelta(hours=12))
        # unit-safe conversion (pandas >= 3 may infer s/us/ns resolution; NaT -> NaN)
        return ((ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)
    if kind == "iso":
        ts = pd.to_datetime(series, errors="coerce")
        return ts.map(lambda v: v.timestamp() if pd.notna(v) else np.nan).to_numpy(dtype=float)
    raise ValueError(f"unknown time_kind {kind!r}")


def _read_csvs(spec: dict, root: Path, max_rows_per_file: int | None) -> pd.DataFrame:
    files = sorted(glob.glob(str(root / spec["glob"])))
    if not files:
        raise FileNotFoundError(
            f"No files match {spec['glob']!r} under {root}. See docs/DATASETS.md.")
    frames = []
    for i, f in enumerate(files):
        kw = dict(encoding=spec.get("encoding", "utf-8"), low_memory=False,
                  nrows=max_rows_per_file)
        if spec.get("header") == "none":
            kw.update(header=None, names=spec["names"])
        df = pd.read_csv(f, **kw)
        df.columns = [str(c).strip() for c in df.columns]
        df["__file_idx"] = i
        df["__row_idx"] = np.arange(len(df))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_dataset(name: str, root: str | os.PathLike = ".", config: dict | None = None,
                 max_rows_per_file: int | None = None) -> Dataset:
    cfg = (config or load_config())[name]
    root = Path(root)
    df = _read_csvs(cfg, root, max_rows_per_file)
    return frame_to_dataset(df, name, cfg)


def frame_to_dataset(df: pd.DataFrame, name: str, cfg: dict) -> Dataset:
    """Turn a raw frame + preset into a Dataset (also used for synthetic data)."""
    df = df.copy()
    label_col = cfg["label_col"]
    # Some raw files (e.g. CICIDS2017 Thursday-Morning-WebAttacks) end with thousands of fully empty rows.
    internal = [c for c in df.columns if str(c).startswith("__")]
    blank = df.drop(columns=internal).isna().all(axis=1).to_numpy()
    n_blank = int(blank.sum())
    if n_blank:
        print(f"[secleak] {name}: dropping {n_blank:,} completely empty rows", file=sys.stderr)
        df = df.loc[~blank].reset_index(drop=True)
    lab = df[label_col]
    if "benign_values" in cfg:
        y = (~lab.astype(str).str.strip().isin(cfg["benign_values"])).astype(int).to_numpy()
    else:
        y = (pd.to_numeric(lab, errors="coerce") == cfg.get("attack_value", 1)).astype(int).to_numpy()

    kind = cfg.get("time_kind", "file_order")
    if kind == "file_order":
        t = (df["__file_idx"].to_numpy() * 1e9 + df["__row_idx"].to_numpy()).astype(float)
        proxy = True
    else:
        t = _parse_time(df[cfg["time_col"]], kind)
        proxy = False
    n_bad_time = int(np.isnan(t).sum())
    if n_bad_time:
        frac = n_bad_time / len(df)
        msg = f"[secleak] {name}: {n_bad_time:,} of {len(df):,} rows ({frac:.2%}) have an unparseable timestamp"
        if frac > 0.001:   # never drop a meaningful share of the data silently
            raise ValueError(msg + " -- fix the time format in configs/datasets.yaml / _parse_time")
        print(msg + "; dropping them", file=sys.stderr)
        keep = ~np.isnan(t)
        df, y, t = df.loc[keep].reset_index(drop=True), y[keep], t[keep]

    # Raw labels for per-label diagnostics. Some datasets keep the attack type in a separate column
    # (UNSW-NB15 `attack_cat`, blank for normal traffic; BoT-IoT `category`): use it when configured.
    cat_col = cfg.get("category_col")
    if cat_col and cat_col in df.columns:
        fill = str(cfg.get("category_fill", "unknown"))
        labels = df[cat_col].fillna(fill).astype(str).str.strip().replace({"": fill}).to_numpy()
        labels = np.where((y == 1) & (labels == fill), "attack_uncategorised", labels)
        mixed = pd.Series(y).groupby(labels).nunique()
        if (mixed > 1).any():
            print(f"[secleak] {name}: labels with both benign and attack rows: "
                  f"{sorted(mixed[mixed > 1].index)}", file=sys.stderr)
    else:
        labels = df[label_col].astype(str).str.strip().to_numpy()
    drop = set(cfg.get("drop_cols", [])) | {label_col, "__file_idx", "__row_idx"}
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan)
    # constant string columns / free text are useless; keep it simple & deterministic
    return Dataset(name=name, X=X.reset_index(drop=True), y=y, time=t, time_is_proxy=proxy,
                   meta={"n_raw": len(df), "n_blank_dropped": n_blank, "n_bad_time_dropped": n_bad_time,
                         "time_kind": kind}, labels=labels)


def harmonise(ds: Dataset, mapping: dict) -> pd.DataFrame:
    """Return the shared-feature view of a dataset for cross-dataset tests."""
    out = {}
    for new, (col, scale) in mapping.items():
        out[new] = pd.to_numeric(ds.X[col], errors="coerce").to_numpy(dtype=float) * float(scale)
    return pd.DataFrame(out)


def subsample(ds: Dataset, max_rows: int, seed: int = 0) -> Dataset:
    """Uniform random subsample that PRESERVES temporal order."""
    if max_rows is None or len(ds) <= max_rows:
        return ds
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(ds), max_rows, replace=False))
    return Dataset(ds.name, ds.X.iloc[idx].reset_index(drop=True), ds.y[idx], ds.time[idx],
                   ds.time_is_proxy, {**ds.meta, "subsampled_to": max_rows},
                   None if ds.labels is None else ds.labels[idx])
