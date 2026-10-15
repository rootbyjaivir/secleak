"""Rolling-origin evaluation with a size- and label-matched "leaky" control.

Why: the single 70/30 temporal split is deterministic, so its confidence interval only reflects model
seeds. Here every raw label is cut into B time-ordered blocks. For each origin k (first_origin..B-2; the last block has no future to leak from):

  rolling_past    : train on blocks < k (the past only), test on block k
  blocked_matched : train on a random sample drawn from all OTHER blocks (past AND future) with exactly the
                    same number of rows per label as the past-only training set, test on the same block k

Both fit on the same amount of data per label and are scored on the same test rows, so
delta = F1(blocked_matched) - F1(rolling_past) isolates what is gained by seeing the future, i.e. the leakage
a random/blocked CV would add. The spread of the delta over origins is the (honest) uncertainty over test windows.
Fitting is untuned (fixed sklearn defaults from make_model) to keep many folds affordable.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .data import Dataset
from .experiments import _cap, cap_train
from .metrics import score
from .models import make_model


def _labels(ds: Dataset) -> np.ndarray:
    return ds.labels if ds.labels is not None else np.where(ds.y == 1, "ATTACK", "BENIGN")


def assign_blocks(ds: Dataset, n_blocks: int) -> np.ndarray:
    """Per raw label, sort by time and cut into n_blocks equal-count, time-ordered blocks (0 = earliest)."""
    lab = _labels(ds)
    blocks = np.zeros(len(ds), dtype=int)
    for name in np.unique(lab):
        idx = np.flatnonzero(lab == name)
        idx = idx[np.argsort(ds.time[idx], kind="stable")]
        blocks[idx] = (np.arange(len(idx)) * n_blocks) // len(idx)
    return blocks


def make_fold(ds: Dataset, blocks: np.ndarray, k: int, rng: np.random.Generator,
              max_train_rows: int | None = None, max_test_rows: int | None = None,
              cap_mode: str = "uniform") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lab = _labels(ds)
    past = np.flatnonzero(blocks < k)
    test = np.flatnonzero(blocks == k)
    past = cap_train(ds, past, max_train_rows, rng, cap_mode)
    test = _cap(test, max_test_rows, rng)
    others = np.flatnonzero(blocks != k)
    ctrl = []
    for name, need in zip(*np.unique(lab[past], return_counts=True)):
        pool = others[lab[others] == name]
        ctrl.append(rng.choice(pool, min(int(need), len(pool)), replace=False))
    return np.sort(past), np.sort(np.concatenate(ctrl)), np.sort(test)


def rolling(ds: Dataset, n_blocks: int = 10, first_origin: int = 4, models: Iterable[str] = ("rf",),
            seeds: Iterable[int] = (0,), max_train_rows: int | None = 200_000,
            max_test_rows: int | None = None, cap_mode: str = "uniform", verbose: bool = True) -> pd.DataFrame:
    # the last block has no future blocks, so its matched control would be the same rows as the past-only
    # training set (delta = 0 by construction): origins stop at n_blocks - 2.
    if not 1 <= first_origin < n_blocks - 1:
        raise ValueError("need 1 <= first_origin < n_blocks - 1")
    blocks = assign_blocks(ds, n_blocks)
    rows = []
    for k in range(first_origin, n_blocks - 1):
        for seed in seeds:
            rng = np.random.default_rng(1000 * seed + k)
            past, ctrl, test = make_fold(ds, blocks, k, rng, max_train_rows, max_test_rows, cap_mode)
            t = ds.time[test]
            span = (pd.to_datetime(t.min(), unit="s").isoformat(), pd.to_datetime(t.max(), unit="s").isoformat()) \
                if not ds.time_is_proxy else (str(t.min()), str(t.max()))
            for mname in models:
                for proto, tr in (("rolling_past", past), ("blocked_matched", ctrl)):
                    t0 = time.time()
                    m = make_model(mname, seed, n_jobs=-1).fit(ds.X.iloc[tr], ds.y[tr])
                    sc = score(ds.y[test], m.predict(ds.X.iloc[test]))
                    rows.append(dict(origin=k, seed=seed, model=mname, protocol=proto, n_train=len(tr),
                                     n_test=len(test), test_start=span[0], test_end=span[1], **sc))
                    if verbose:
                        print(f"[rolling] k={k} seed={seed} {mname:<6} {proto:<15} F1={sc['f1_attack']:.4f} "
                              f"prec={sc['precision']:.3f} rec={sc['recall']:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, n_boot: int = 10_000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Per model: F1 of each protocol and the paired delta, averaged over seeds within an origin, then
    summarised over origins (bootstrap 95% CI resampling ORIGINS, min/max shown)."""
    per_origin = (df.pivot_table(index=["model", "origin"], columns="protocol", values="f1_attack", aggfunc="mean")
                  .reset_index())
    per_origin["delta"] = per_origin["blocked_matched"] - per_origin["rolling_past"]
    rng = np.random.default_rng(seed)
    out = []
    for mname, g in per_origin.groupby("model"):
        d = g["delta"].to_numpy()
        boots = rng.choice(d, (n_boot, len(d)), replace=True).mean(axis=1)
        out.append(dict(model=mname, n_origins=len(d), rolling_past_f1=g["rolling_past"].mean(),
                        blocked_matched_f1=g["blocked_matched"].mean(), delta_mean=d.mean(),
                        ci_lo=np.percentile(boots, 2.5), ci_hi=np.percentile(boots, 97.5),
                        delta_min=d.min(), delta_max=d.max()))
    return {"per_origin": per_origin, "summary": pd.DataFrame(out)}


def write_rolling(df: pd.DataFrame, out: str | Path, name: str, n_blocks: int, first_origin: int) -> Path:
    from .diagnose import _md
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rolling_folds.csv", index=False)
    res = summarize(df)
    res["per_origin"].to_csv(out / "rolling_per_origin.csv", index=False)
    res["summary"].to_csv(out / "rolling_summary.csv", index=False)
    spans = df.groupby("origin").agg(test_start=("test_start", "first"), test_end=("test_end", "first"),
                                     n_test=("n_test", "first")).reset_index()
    lines = [f"# Rolling-origin results: {name}", "",
             f"Each raw label is cut into {n_blocks} time-ordered blocks; origins k = {first_origin}..{n_blocks - 2} "
             "(train on blocks < k, test on block k). `blocked_matched` trains on the SAME number of rows per "
             "label drawn from all other blocks (past and future) and is scored on the same test block, so "
             "`delta` is the F1 gained purely from seeing the future. Untuned models; one seed set.", "",
             "## Summary over origins (bootstrap 95% CI resampling origins; few origins => wide CI)", "",
             _md(res["summary"], "{:.3f}"), "",
             "## Per origin (mean over seeds)", "", _md(res["per_origin"], "{:.3f}"), "",
             "## Test windows", "", _md(spans, "{:.3f}"), "",
             "> The delta spread across origins reflects different test windows, not sampling uncertainty "
             "over unseen datasets. Blocks are cut per label, so a test block's time span is the union of each "
             "label's k-th time slice (labels confined to different days make the span wide). Later origins have "
             "less future data to leak from, so the delta naturally shrinks. Rare labels may be absent from some "
             "blocks.", ""]
    path = out / "rolling_report.md"
    path.write_text("\n".join(lines))
    return path
