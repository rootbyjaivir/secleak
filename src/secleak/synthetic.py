"""Synthetic flow-like dataset with *planted* integrity problems.

Used by the demo and tests so the whole pipeline can be validated without downloading
multi-GB datasets. Planted problems (all controllable):

* concept drift     : the attack signature in ``f0`` flips sign late in the timeline
                      -> large random-vs-temporal gap
* exact duplicates  : rows copied across the timeline -> random-split leakage
* label conflicts   : identical features, opposite label -> label noise
* shortcut feature  : ``dst_port`` predicts the label with probability ``shortcut_strength``
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic(n: int = 20000, attack_frac: float = 0.3, dup_frac: float = 0.15,
                   conflict_frac: float = 0.01, drift: bool = True,
                   shortcut_strength: float = 0.75, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0, 1, n))                       # normalised time
    y = (rng.uniform(size=n) < attack_frac).astype(int)

    f = rng.normal(0, 1, (n, 6))
    sign = np.where((t > 0.72) & drift, -1.0, 1.0)          # signature flips late
    f[:, 0] += y * 2.0 * sign                               # drifting signal
    f[:, 1] += y * 0.6                                      # weak stable signal
    f[:, 2] -= y * 0.4

    df = pd.DataFrame(f, columns=[f"f{i}" for i in range(6)])
    df["flow_duration"] = np.exp(rng.normal(3, 1.2, n) + 0.3 * y)
    df["fwd_pkts"] = rng.poisson(5 + 2 * y, n)
    df["bwd_pkts"] = rng.poisson(4 + 1 * y, n)

    benign_ports = np.array([80, 443, 53, 22, 8080])
    attack_ports = np.array([23, 2323, 7547, 5555, 37215])
    port = np.where(y == 1, rng.choice(attack_ports, n), rng.choice(benign_ports, n))
    other = np.where(y == 1, rng.choice(benign_ports, n), rng.choice(attack_ports, n))
    df["dst_port"] = np.where(rng.uniform(size=n) > shortcut_strength, other, port)

    df["timestamp"] = pd.Timestamp("2017-07-03") + pd.to_timedelta(t * 5 * 24 * 3600, unit="s")
    df["label"] = np.where(y == 1, "ATTACK", "BENIGN")

    feats = [c for c in df.columns if c not in ("timestamp", "label")]
    k = int(dup_frac * n)
    if k:                                                    # duplicates keep their own timestamp
        src, dst = rng.integers(0, n, k), rng.integers(0, n, k)
        df.loc[dst, feats] = df.loc[src, feats].to_numpy()
        df.loc[dst, "label"] = df.loc[src, "label"].to_numpy()
    kc = int(conflict_frac * n)
    if kc:
        src, dst = rng.integers(0, n, kc), rng.integers(0, n, kc)
        df.loc[dst, feats] = df.loc[src, feats].to_numpy()
        df.loc[dst, "label"] = np.where(df.loc[src, "label"].to_numpy() == "ATTACK",
                                        "BENIGN", "ATTACK")
    return df
