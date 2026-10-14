"""Aggregate results -> tables (markdown/LaTeX), deltas with CIs, and figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

METRIC = "f1_attack"


def _ci95(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def summarize(df: pd.DataFrame, metric: str = METRIC) -> pd.DataFrame:
    g = df.groupby(["dataset", "protocol", "model"])[metric]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["ci95"] = g.apply(lambda s: _ci95(s.to_numpy())).to_numpy()
    return out


def deltas(df: pd.DataFrame, baseline: str = "random", metric: str = METRIC,
           n_boot: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Per dataset/model/protocol: mean (baseline - protocol) with paired bootstrap CI over seeds."""
    rng = np.random.default_rng(seed)
    rows = []
    piv = df.pivot_table(index=["dataset", "model", "seed"], columns="protocol", values=metric)
    if baseline not in piv.columns:
        return pd.DataFrame()
    for p in [c for c in piv.columns if c != baseline]:
        for (d, m), sub in piv.groupby(level=[0, 1]):
            diff = (sub[baseline] - sub[p]).dropna().to_numpy()
            if len(diff) == 0:
                continue
            boots = rng.choice(diff, (n_boot, len(diff))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append(dict(dataset=d, model=m, protocol=p, n_seeds=len(diff),
                             drop_mean=diff.mean(), ci_lo=lo, ci_hi=hi,
                             rel_drop=diff.mean() / max(sub[baseline].mean(), 1e-12)))
    return pd.DataFrame(rows)


def headline(df: pd.DataFrame, baseline: str = "random", metric: str = METRIC) -> pd.DataFrame:
    """The paper's headline: mean F1 drop per dataset & protocol, averaged over models."""
    d = deltas(df, baseline, metric)
    if d.empty:
        return d
    return (d.groupby(["dataset", "protocol"])
              .agg(mean_drop=("drop_mean", "mean"), min_drop=("drop_mean", "min"),
                   max_drop=("drop_mean", "max"), rel_drop=("rel_drop", "mean"))
              .reset_index())


def plot_protocols(df: pd.DataFrame, out: str | Path, metric: str = METRIC):
    s = summarize(df, metric)
    agg = s.groupby(["dataset", "protocol"])["mean"].mean().unstack("protocol")
    order = [p for p in ["random", "random_dedup", "temporal_global", "temporal_per_class",
                         "temporal_per_label", "cross_dataset"] if p in agg.columns]
    agg = agg[order]
    ax = agg.plot(kind="bar", figsize=(8, 4.2), width=0.8, rot=0)
    ax.set_ylabel(f"{metric} (mean over models/seeds)")
    ax.set_ylim(0, 1.02)
    ax.legend(title="protocol", fontsize=8)
    ax.set_title("Reported (random split) vs. leak-controlled performance")
    plt.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200)
    plt.close()


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating))
                                       else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_report(results_csv: str | Path, out_dir: str | Path) -> Path:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    s, d, h = summarize(df), deltas(df), headline(df)
    s.to_csv(out / "summary.csv", index=False)
    d.to_csv(out / "deltas.csv", index=False)
    plot_protocols(df, out / "f1_by_protocol.png")
    md = ["# SecLeak results", "", "## Headline: mean F1 drop vs. random split", "",
          to_markdown(h) if not h.empty else "_(no baseline)_", "",
          "## Per-model summary (mean, 95% CI over seeds)", "", to_markdown(s), "",
          "## Paired deltas (baseline - protocol), bootstrap 95% CI over seeds", "",
          to_markdown(d) if not d.empty else "_(none)_", "",
          "> Seeds vary model initialisation and the random split; deterministic temporal splits "
          "only vary the model, so CIs there are narrower than the true sampling uncertainty. "
          "Use >=10 seeds for the paper."]
    if "time_is_proxy" in df.columns and df["time_is_proxy"].any():
        md.append("\n> WARNING: at least one dataset has no real timestamps; its 'temporal' "
                  "protocols use row order as a proxy.")
    (out / "report.md").write_text("\n".join(md))
    return out / "report.md"
