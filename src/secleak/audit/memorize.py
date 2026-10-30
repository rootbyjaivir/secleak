"""LLM memorisation probes for tabular benchmarks (after Bordt et al., 'Elephants Never Forget').

Three probes, each run on REAL material and on a matched CONTROL the model cannot have seen:

  header   : continue the CSV header given its first k column names
             (control: same task, column order randomly permuted)
  row      : given header + r consecutive rows, emit the next row
             (control: per-column shuffled values -> realistic marginals, no true rows)
  feature  : given a row with one field masked, recover the field
             (control: same, on the shuffled-value rows)

Memorisation evidence = real score significantly above control score (permutation test).
A bare 'accuracy' with no control is uninterpretable: a model can guess easy fields.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np


class LLMClient(Protocol):
    def complete(self, prompt: str, max_tokens: int = 400) -> str: ...


class AnthropicClient:
    """Thin wrapper; needs `pip install anthropic` and ANTHROPIC_API_KEY."""

    def __init__(self, model: str | None = None, temperature: float = 0.0):
        import anthropic  # lazy import
        self.client = anthropic.Anthropic()
        self.model = model or os.environ.get("SECLEAK_MODEL", "claude-sonnet-5-5")
        self.temperature = temperature

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        r = self.client.messages.create(model=self.model, max_tokens=max_tokens,
                                        messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


class CallableClient:
    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def complete(self, prompt: str, max_tokens: int = 400) -> str:
        return self.fn(prompt)


@dataclass
class Example:
    kind: str          # header | row | feature
    condition: str     # real | control
    prompt: str
    truth: str


# ---------------------------------------------------------------- example builders
def read_raw_lines(path: str, n_max: int = 200_000, encoding: str = "cp1252") -> list[str]:
    out = []
    with open(path, encoding=encoding, errors="replace") as fh:
        for i, line in enumerate(fh):
            out.append(line.rstrip("\r\n"))
            if i >= n_max:
                break
    return out


def _shuffle_columns(rows: list[list[str]], rng) -> list[list[str]]:
    arr = np.array(rows, dtype=object)
    for j in range(arr.shape[1]):
        arr[:, j] = arr[rng.permutation(arr.shape[0]), j]
    return arr.tolist()


def build_examples(lines: list[str], fname: str, n: int = 40, k_header: int = 8,
                   m_header: int = 8, r_rows: int = 5, seed: int = 0) -> list[Example]:
    rng = np.random.default_rng(seed)
    header, body = lines[0], lines[1:]
    cols = header.split(",")
    ex: list[Example] = []

    # --- header probe (one example per condition: it is deterministic, so repeat for variants)
    for cond in ("real", "control"):
        c = list(cols)
        if cond == "control":
            c = [c[i] for i in rng.permutation(len(c))]
        starts = range(0, max(1, len(c) - k_header - m_header + 1),
                       max(1, (len(c) - k_header - m_header) // 5 or 1))
        for s in list(starts)[:5]:
            given, nxt = c[s:s + k_header], c[s + k_header:s + k_header + m_header]
            if len(nxt) < 2:
                continue
            p = (f"Below are {k_header} consecutive column names from the header of the CSV file "
                 f"`{fname}`. Output ONLY the next {len(nxt)} column names, comma-separated, "
                 f"preserving exact spelling and spacing.\n\n{','.join(given)}")
            ex.append(Example("header", cond, p, ",".join(nxt)))

    # --- row / feature probes
    pool_idx = rng.choice(len(body) - r_rows - 1, size=min(n, len(body) - r_rows - 1), replace=False)
    for i in pool_idx:
        win = [b.split(",") for b in body[i:i + r_rows + 1]]
        if any(len(w) != len(cols) for w in win):
            continue
        real_ctx, real_next = win[:-1], win[-1]
        # control: shuffle values column-wise over a larger pool so rows are unseen combinations
        pool = [b.split(",") for b in body[i:i + 50] if len(b.split(",")) == len(cols)]
        if len(pool) < r_rows + 2:
            continue
        ctrl = _shuffle_columns(pool, rng)
        ctrl_ctx, ctrl_next = ctrl[:r_rows], ctrl[r_rows]
        for cond, ctx, nxt in (("real", real_ctx, real_next), ("control", ctrl_ctx, ctrl_next)):
            ctx_s = "\n".join(",".join(r) for r in ctx)
            ex.append(Example("row", cond,
                f"These are consecutive lines from the CSV file `{fname}`. Output ONLY the next "
                f"line of the file (all {len(cols)} fields, comma-separated).\n\n{header}\n{ctx_s}",
                ",".join(nxt)))
            j = int(rng.integers(0, len(cols)))
            masked = list(nxt); truth = masked[j]; masked[j] = "???"
            ex.append(Example("feature", cond,
                f"This is a line from the CSV file `{fname}` with one field replaced by ???. "
                f"Output ONLY the original value of ??? (column `{cols[j]}`).\n\n{header}\n"
                f"{','.join(masked)}", truth))
    return ex


# ---------------------------------------------------------------- scoring
def _norm(s: str) -> str:
    return s.strip().strip("`").strip()


def score_example(e: Example, answer: str) -> float:
    a = answer.strip().splitlines()[0] if answer.strip() else ""
    if e.kind == "feature":
        return float(_norm(a) == _norm(e.truth))
    got, tru = a.split(","), e.truth.split(",")
    if e.kind == "header":     # position-wise exact match, whitespace-normalised
        return float(np.mean([_norm(g) == _norm(t) for g, t in zip(got, tru)] + [False] * max(0, len(tru) - len(got))))
    # row: fraction of fields matching exactly
    return float(np.mean([_norm(g) == _norm(t) for g, t in zip(got, tru)] + [False] * max(0, len(tru) - len(got))))


def evaluate(client: LLMClient, examples: list[Example], max_tokens: int = 400,
             verbose: bool = False) -> list[dict]:
    out = []
    for i, e in enumerate(examples):
        try:
            ans = client.complete(e.prompt, max_tokens=max_tokens)
        except Exception as exc:                     # API hiccup: record, do not crash the run
            ans = ""
            if verbose:
                print("API error:", exc)
        out.append(dict(kind=e.kind, condition=e.condition, score=score_example(e, ans)))
        if verbose and i % 10 == 0:
            print(f"{i+1}/{len(examples)}", flush=True)
    return out


def compare(results: list[dict], n_perm: int = 10_000, seed: int = 0) -> list[dict]:
    """Per probe: mean(real) - mean(control), one-sided permutation p, bootstrap 95% CI."""
    rng = np.random.default_rng(seed)
    rep = []
    for kind in sorted({r["kind"] for r in results}):
        real = np.array([r["score"] for r in results if r["kind"] == kind and r["condition"] == "real"])
        ctrl = np.array([r["score"] for r in results if r["kind"] == kind and r["condition"] == "control"])
        if len(real) == 0 or len(ctrl) == 0:
            continue
        obs = real.mean() - ctrl.mean()
        allv = np.concatenate([real, ctrl])
        cnt = 0
        for _ in range(n_perm):
            rng.shuffle(allv)
            cnt += (allv[:len(real)].mean() - allv[len(real):].mean()) >= obs
        p = (cnt + 1) / (n_perm + 1)
        boots = [rng.choice(real, len(real)).mean() - rng.choice(ctrl, len(ctrl)).mean()
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rep.append(dict(probe=kind, n_real=len(real), n_control=len(ctrl),
                        mean_real=real.mean(), mean_control=ctrl.mean(),
                        diff=obs, ci_lo=lo, ci_hi=hi, p_one_sided=p))
    return rep
