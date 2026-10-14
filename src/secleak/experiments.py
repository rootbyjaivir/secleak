"""Run the protocol x model x seed matrix and the cross-dataset test."""
from __future__ import annotations

import time
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from .audit.dedup import train_test_overlap
from .data import Dataset, harmonise
from .metrics import score
from .models import GRIDS, MODEL_NAMES, make_model
from .protocols import PROTOCOLS, split, validation_split


def _cap(idx: np.ndarray, k: int | None, rng: np.random.Generator) -> np.ndarray:
    if k is None or len(idx) <= k:
        return idx
    return np.sort(rng.choice(idx, k, replace=False))


def cap_label_floor(idx: np.ndarray, labels: np.ndarray, k: int | None, rng: np.random.Generator,
                    floor: int = 5000) -> np.ndarray:
    """Training cap that never starves rare labels.

    Every row of a label with <= `floor` rows is kept; the remaining budget is filled by uniform sampling
    from the larger labels. Without this, a uniform cap leaves e.g. Infiltration (36 flows) with ~0-3
    training rows, so a "recall 0" says nothing about drift. NOTE: it shifts class priors towards rare
    labels; apply it identically to every protocol and say so in the paper."""
    if k is None or len(idx) <= k:
        return idx
    _, inv, cnt = np.unique(labels[idx], return_inverse=True, return_counts=True)
    small = cnt <= floor
    keep_small = np.flatnonzero(small[inv])
    big = np.flatnonzero(~small[inv])
    if len(keep_small) >= k:                      # degenerate: too many rare rows, fall back to uniform
        return np.sort(rng.choice(idx, k, replace=False))
    budget = k - len(keep_small)
    chosen = big if budget >= len(big) else rng.choice(big, budget, replace=False)
    return np.sort(idx[np.concatenate([keep_small, chosen])])


def cap_train(ds: Dataset, idx: np.ndarray, k: int | None, rng: np.random.Generator,
              mode: str = "uniform") -> np.ndarray:
    if mode == "uniform":
        return _cap(idx, k, rng)
    if mode == "label_floor":
        if ds.labels is None:
            raise ValueError("cap_mode='label_floor' needs raw labels (Dataset.labels)")
        return cap_label_floor(idx, ds.labels, k, rng)
    raise ValueError(f"unknown cap_mode {mode!r}; choose 'uniform' or 'label_floor'")


def _tune(ds: Dataset, protocol: str, tr: np.ndarray, model_name: str, seed: int, X):
    """Pick hyper-parameters on a validation split built with the SAME protocol."""
    grid = GRIDS.get(model_name)
    if not grid:
        return {}
    try:
        t2, v2 = validation_split(ds, protocol, tr, seed=seed)
    except Exception:
        return {}
    if len(np.unique(ds.y[t2])) < 2 or len(v2) == 0:
        return {}
    best, best_f1 = {}, -1.0
    for params in grid:
        m = make_model(model_name, seed).set_params(**params)
        m.fit(X.iloc[t2], ds.y[t2])
        f = f1_score(ds.y[v2], m.predict(X.iloc[v2]), zero_division=0)
        if f > best_f1:
            best, best_f1 = params, f
    return best


def run_matrix(ds: Dataset, protocols: Iterable[str] = PROTOCOLS,
               models: Iterable[str] = MODEL_NAMES, seeds: Iterable[int] = (0, 1, 2),
               test_size: float = 0.3, tune: bool = True,
               max_train_rows: int | None = 200_000, max_test_rows: int | None = 200_000,
               cap_mode: str = "uniform", verbose: bool = True) -> pd.DataFrame:
    rows = []
    X = ds.X
    for protocol in protocols:
        for seed in seeds:
            tr, te = split(ds, protocol, test_size=test_size, seed=seed)
            ov_exact = train_test_overlap(X, tr, te, y=ds.y)
            ov_near = train_test_overlap(X, tr, te, y=ds.y, near_sig=4)
            rng = np.random.default_rng(seed)
            tr_c, te_c = cap_train(ds, tr, max_train_rows, rng, cap_mode), _cap(te, max_test_rows, rng)
            for mname in models:
                t0 = time.time()
                params = _tune(ds, protocol, tr_c, mname, seed, X) if tune else {}
                m = make_model(mname, seed).set_params(**params)
                m.fit(X.iloc[tr_c], ds.y[tr_c])
                pred = m.predict(X.iloc[te_c])
                rec = dict(dataset=ds.name, protocol=protocol, model=mname, seed=seed,
                           n_train=len(tr_c), n_test=len(te_c),
                           time_is_proxy=ds.time_is_proxy,
                           overlap_exact=ov_exact["test_rows_in_train_frac"],
                           overlap_near=ov_near["test_rows_in_train_frac"],
                           params=str(params), **score(ds.y[te_c], pred))
                rows.append(rec)
                if verbose:
                    print(f"[{ds.name}] {protocol:<19} {mname:<7} seed={seed} "
                          f"F1={rec['f1_attack']:.4f} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def run_cross(ds_a: Dataset, ds_b: Dataset, map_a: dict, map_b: dict,
              models: Iterable[str] = MODEL_NAMES, seeds: Iterable[int] = (0, 1, 2),
              max_rows: int | None = 200_000, verbose: bool = True) -> pd.DataFrame:
    """Train on A, test on B over the harmonised feature set (both directions)."""
    rows = []
    for (src, dst, ms, md) in ((ds_a, ds_b, map_a, map_b), (ds_b, ds_a, map_b, map_a)):
        Xs, Xd = harmonise(src, ms), harmonise(dst, md)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            i_s = _cap(np.arange(len(src)), max_rows, rng)
            i_d = _cap(np.arange(len(dst)), max_rows, rng)
            for mname in models:
                m = make_model(mname, seed).fit(Xs.iloc[i_s], src.y[i_s])
                pred = m.predict(Xd.iloc[i_d])
                rec = dict(dataset=f"{src.name}->{dst.name}", protocol="cross_dataset",
                           model=mname, seed=seed, n_train=len(i_s), n_test=len(i_d),
                           time_is_proxy=src.time_is_proxy or dst.time_is_proxy,
                           overlap_exact=np.nan, overlap_near=np.nan, params="{}",
                           **score(dst.y[i_d], pred))
                rows.append(rec)
                if verbose:
                    print(f"[{rec['dataset']}] {mname} seed={seed} F1={rec['f1_attack']:.4f}", flush=True)
    return pd.DataFrame(rows)
