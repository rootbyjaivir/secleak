"""Literature-census analysis: prevalence of risky evaluation practices (+ coder agreement).

Input CSV columns (see census/codebook.md):
  paper_id, year, venue, dataset, split_method, dedup, temporal_aware, cross_dataset,
  llm_based, code_available, reported_f1, metric
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FLAGS = ["dedup", "temporal_aware", "cross_dataset", "code_available", "llm_based"]


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    m = a.notna() & b.notna()
    a, b = a[m].astype(str), b[m].astype(str)
    if len(a) == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum((a == c).mean() * (b == c).mean() for c in set(a) | set(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def analyze(path: str, out_dir: str, second: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    n = len(df)
    rows = []
    for f in FLAGS:
        if f in df:
            k = int((df[f].astype(str).str.lower().isin(["1", "yes", "true", "y"])).sum())
            lo, hi = wilson(k, n)
            rows.append(dict(practice=f, k=k, n=n, pct=100 * k / n, ci_lo=100 * lo, ci_hi=100 * hi))
    if "split_method" in df:
        vc = df["split_method"].str.lower().value_counts()
        for name, k in vc.items():
            lo, hi = wilson(int(k), n)
            rows.append(dict(practice=f"split={name}", k=int(k), n=n, pct=100 * k / n,
                             ci_lo=100 * lo, ci_hi=100 * hi))
    res = pd.DataFrame(rows)
    res.to_csv(out / "prevalence.csv", index=False)
    if "reported_f1" in df and "dataset" in df:
        s = df.groupby("dataset")["reported_f1"].describe()[["count", "mean", "50%", "min", "max"]]
        s.to_csv(out / "reported_f1_by_dataset.csv")
    if second:
        d2 = pd.read_csv(second)
        j = df.merge(d2, on="paper_id", suffixes=("_a", "_b"))
        agr = {c: cohen_kappa(j[f"{c}_a"], j[f"{c}_b"]) for c in FLAGS + ["split_method"]
               if f"{c}_a" in j}
        pd.Series(agr, name="cohen_kappa").to_csv(out / "agreement.csv")
    print(res.to_string(index=False))
    return res
