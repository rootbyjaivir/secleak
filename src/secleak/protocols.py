"""Evaluation protocols. Each returns (train_idx, test_idx) as sorted integer arrays.

P1  random            : the de-facto norm in the literature (stratified random split)
P2  random_dedup      : P1 after removing exact duplicates & label-conflicting groups
P3  temporal_global   : earliest (1-test_size) of the timeline -> train, latest -> test
P4  temporal_per_class: same cut *within each class* (keeps class balance; for datasets
                        where attack types occupy separate time windows, e.g. CICIDS2017)
P4b temporal_per_label: same cut *within each raw attack label* (BENIGN, DoS Hulk, ...). Every label appears
                        in both train and test, so the drop isolates drift WITHIN an attack type from the
                        arrival of attack types the model never saw. (Added after the first CICIDS2017 run
                        showed that P4's test attacks are only the latest campaigns; see PREREGISTRATION.)
P5  cross_dataset     : handled in experiments.py (needs two datasets)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .audit.dedup import dedup_mask

PROTOCOLS = ["random", "random_dedup", "temporal_global", "temporal_per_class", "temporal_per_label"]


def _stratified_random(y: np.ndarray, test_size: float, rng: np.random.Generator,
                       idx: np.ndarray | None = None):
    idx = np.arange(len(y)) if idx is None else idx
    tr, te = [], []
    for c in np.unique(y[idx]):
        ci = rng.permutation(idx[y[idx] == c])
        k = int(round(len(ci) * test_size))
        te.append(ci[:k]); tr.append(ci[k:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def split_random(ds, test_size=0.3, seed=0, idx=None):
    return _stratified_random(ds.y, test_size, np.random.default_rng(seed), idx)


def split_random_dedup(ds, test_size=0.3, seed=0, idx=None):
    base = np.arange(len(ds)) if idx is None else idx
    keep = dedup_mask(ds.X.iloc[base], ds.y[base], drop_conflicts=True)
    return _stratified_random(ds.y, test_size, np.random.default_rng(seed), base[keep])


def split_temporal_global(ds, test_size=0.3, seed=0, idx=None, gap_frac=0.0):
    base = np.arange(len(ds)) if idx is None else idx
    order = base[np.argsort(ds.time[base], kind="stable")]
    n = len(order)
    cut = int(round(n * (1 - test_size)))
    gap = int(round(n * gap_frac))
    return np.sort(order[: cut - gap]), np.sort(order[cut:])


def split_temporal_per_class(ds, test_size=0.3, seed=0, idx=None, gap_frac=0.0):
    base = np.arange(len(ds)) if idx is None else idx
    tr, te = [], []
    for c in np.unique(ds.y[base]):
        ci = base[ds.y[base] == c]
        ci = ci[np.argsort(ds.time[ci], kind="stable")]
        cut = int(round(len(ci) * (1 - test_size)))
        gap = int(round(len(ci) * gap_frac))
        tr.append(ci[: cut - gap]); te.append(ci[cut:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


def split_temporal_per_label(ds, test_size=0.3, seed=0, idx=None, gap_frac=0.0):
    if ds.labels is None:
        raise ValueError("temporal_per_label needs raw labels (Dataset.labels); reload the dataset")
    base = np.arange(len(ds)) if idx is None else idx
    tr, te = [], []
    for lab in np.unique(ds.labels[base]):
        ci = base[ds.labels[base] == lab]
        ci = ci[np.argsort(ds.time[ci], kind="stable")]
        cut = int(round(len(ci) * (1 - test_size)))
        gap = int(round(len(ci) * gap_frac))
        tr.append(ci[: cut - gap]); te.append(ci[cut:])
    return np.sort(np.concatenate(tr)), np.sort(np.concatenate(te))


_SPLITTERS = {
    "random": split_random,
    "random_dedup": split_random_dedup,
    "temporal_global": split_temporal_global,
    "temporal_per_class": split_temporal_per_class,
    "temporal_per_label": split_temporal_per_label,
}


def split(ds, protocol: str, test_size: float = 0.3, seed: int = 0, idx=None, **kw):
    if protocol not in _SPLITTERS:
        raise ValueError(f"unknown protocol {protocol!r}; choose from {PROTOCOLS}")
    return _SPLITTERS[protocol](ds, test_size=test_size, seed=seed, idx=idx, **kw)


def validation_split(ds, protocol: str, train_idx: np.ndarray, val_size: float = 0.25,
                     seed: int = 0):
    """Carve a validation set out of TRAIN using the SAME protocol, so hyper-parameter
    tuning cannot smuggle in the very leakage the protocol is meant to remove."""
    return split(ds, protocol, test_size=val_size, seed=seed, idx=train_idx)
