"""Diagnostics that explain WHY performance changes between protocols.

1. composition      : attack share and per-label counts in train/test for each protocol, plus the F1 of the
                      trivial "always predict attack" classifier (2p/(1+p)). A model near that value has
                      learned nothing useful on that split.
2. per-label recall : fraction of each raw label (BENIGN, DoS Hulk, ...) classified correctly. Shows whether a
                      temporal drop comes from benign drift (false alarms), unseen attack types, or both.
3. gap sweep        : temporal_per_class with an increasing gap between the end of train and the start of
                      test. If F1 keeps falling as the gap widens, near-boundary (same-session) flows were
                      helping the model.
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
from .protocols import split


def always_attack_f1(p: float) -> float:
    """F1 (attack = positive) of predicting 'attack' for every row when the attack share is p."""
    return 2 * p / (1 + p) if p > 0 else 0.0


def diagnose(ds: Dataset, protocols: Iterable[str] = ("random", "temporal_per_class", "temporal_per_label", "temporal_global"),
             models: Iterable[str] = ("rf",), seeds: Iterable[int] = (0, 1),
             gaps: Iterable[float] = (0.0, 0.01, 0.05, 0.10), test_size: float = 0.3,
             max_train_rows: int | None = 200_000, max_test_rows: int | None = 200_000,
             cap_mode: str = "uniform", verbose: bool = True) -> dict[str, pd.DataFrame]:
    X = ds.X
    labels = ds.labels if ds.labels is not None else np.where(ds.y == 1, "ATTACK", "BENIGN")
    comp, comp_lab, rec, gap_rows = [], [], [], []

    for protocol in protocols:
        for seed in seeds:
            tr, te = split(ds, protocol, test_size=test_size, seed=seed)
            rng = np.random.default_rng(seed)
            tr, te = cap_train(ds, tr, max_train_rows, rng, cap_mode), _cap(te, max_test_rows, rng)
            p = float(ds.y[te].mean())
            comp.append(dict(protocol=protocol, seed=seed, n_train=len(tr), n_test=len(te),
                             train_attack_frac=float(ds.y[tr].mean()), test_attack_frac=p,
                             always_attack_f1=always_attack_f1(p)))
            for part, idx in (("train", tr), ("test", te)):
                for lab, n in zip(*np.unique(labels[idx], return_counts=True)):
                    comp_lab.append(dict(protocol=protocol, seed=seed, part=part, label=lab, n=int(n)))
            for mname in models:
                t0 = time.time()
                m = make_model(mname, seed, n_jobs=-1).fit(X.iloc[tr], ds.y[tr])
                pred = m.predict(X.iloc[te])
                yt, lt = ds.y[te], labels[te]
                for lab in np.unique(lt):
                    mk = lt == lab
                    rec.append(dict(protocol=protocol, model=mname, seed=seed, label=lab,
                                    kind="attack" if yt[mk][0] == 1 else "benign",
                                    n_train=int((labels[tr] == lab).sum()), n_test=int(mk.sum()),
                                    correct_frac=float((pred[mk] == yt[mk]).mean())))
                if verbose:
                    sc = score(yt, pred)
                    print(f"[diagnose] {protocol:<19} {mname:<6} seed={seed} F1={sc['f1_attack']:.4f} "
                          f"prec={sc['precision']:.3f} rec={sc['recall']:.3f} "
                          f"(always-attack F1={always_attack_f1(p):.3f}) {time.time()-t0:.0f}s", flush=True)

    for gap in gaps:
        for seed in seeds:
            tr, te = split(ds, "temporal_per_class", test_size=test_size, seed=seed, gap_frac=gap)
            rng = np.random.default_rng(seed)
            tr, te = cap_train(ds, tr, max_train_rows, rng, cap_mode), _cap(te, max_test_rows, rng)
            for mname in models:
                m = make_model(mname, seed, n_jobs=-1).fit(X.iloc[tr], ds.y[tr])
                sc = score(ds.y[te], m.predict(X.iloc[te]))
                gap_rows.append(dict(gap_frac=gap, model=mname, seed=seed, n_train=len(tr), n_test=len(te), **sc))
                if verbose:
                    print(f"[diagnose] gap={gap:<5} {mname:<6} seed={seed} F1={sc['f1_attack']:.4f}", flush=True)

    return {"composition": pd.DataFrame(comp), "label_counts": pd.DataFrame(comp_lab),
            "per_label": pd.DataFrame(rec), "gap_sweep": pd.DataFrame(gap_rows)}


def _md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    def cell(col, v):
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return "-"
            return f"{v:,.0f}" if str(col).startswith("n_") else floatfmt.format(v)
        return str(v)
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(cell(c, r[c]) for c in cols) + " |")
    return "\n".join(out)


def write_diagnostics(res: dict[str, pd.DataFrame], out: str | Path, name: str) -> Path:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for k, v in res.items():
        v.to_csv(out / f"{k}.csv", index=False)
    comp = res["composition"].groupby("protocol", as_index=False).mean(numeric_only=True).drop(columns=["seed"])
    sections = []
    for mname, g in res["per_label"].groupby("model"):      # NEVER pool models: their failure modes differ
        t = g.groupby(["kind", "label", "protocol"], as_index=False).agg(
            correct_frac=("correct_frac", "mean"), n_test=("n_test", "mean"))
        piv = t.pivot_table(index=["kind", "label"], columns="protocol", values="correct_frac").reset_index()
        ref = t[t.protocol.isin(["temporal_per_label", "random"])].groupby("label").n_test.max()
        piv.insert(2, "n_test", piv["label"].map(ref))
        piv["label"] = [f"{lab} (n<100)" if n < 100 else lab for lab, n in zip(piv["label"], piv["n_test"])]
        sections += [f"### {mname}", "", _md(piv), ""]
    pl_md = "\n".join(sections)
    cnt = (res["label_counts"].query("seed == @res['label_counts'].seed.min()")
           .pivot_table(index="label", columns=["protocol", "part"], values="n", aggfunc="sum").fillna(0))
    gap = res["gap_sweep"].groupby(["gap_frac", "model"], as_index=False).agg(
        f1_attack=("f1_attack", "mean"), precision=("precision", "mean"), recall=("recall", "mean"))
    lines = [f"# Diagnostics: {name}", "",
             "## 1. Split composition and trivial baseline",
             "`always_attack_f1` is the F1 of predicting 'attack' for every row. A model near this value on a "
             "split has learned nothing useful there.", "", _md(comp), "",
             "## 2. Fraction of each label classified correctly",
             "For attack labels this is recall; for BENIGN it is specificity (1 - false-alarm rate). "
             "One table per model. `n_test` = test flows of that label (temporal_per_label / random split); "
             "labels with n<100 are too small to interpret. Missing cell = label absent from that test set.",
             "", pl_md,
             "## 3. Gap sweep (temporal_per_class, gap between end of train and start of test)", "",
             _md(gap), "",
             "> Temporal splits are deterministic, so seeds only vary the model. These tables describe one "
             "fixed partition of one dataset, not sampling uncertainty over data.", ""]
    path = out / "diagnostics.md"
    path.write_text("\n".join(lines))
    return path
