import tempfile
import numpy as np
import pandas as pd

from secleak.data import frame_to_dataset, subsample
from secleak.diagnose import always_attack_f1, diagnose, write_diagnostics
from secleak.synthetic import make_synthetic

CFG = dict(label_col="label", benign_values=["BENIGN"], time_col="timestamp",
           time_kind="iso", drop_cols=["timestamp"])


def _multilabel_ds(n=3000):
    df = make_synthetic(n=n, seed=1)
    atk = df["label"] == "ATTACK"
    late = df["timestamp"] > df["timestamp"].median()
    df.loc[atk & ~late, "label"] = "Attack A"       # attack types live in different time windows
    df.loc[atk & late, "label"] = "Attack B"
    return frame_to_dataset(df, "syn_multi", CFG)


def test_always_attack_f1_formula():
    assert abs(always_attack_f1(0.5) - 2 / 3) < 1e-12
    assert always_attack_f1(0.0) == 0.0
    assert abs(always_attack_f1(1.0) - 1.0) < 1e-12


def test_raw_labels_kept_and_aligned_after_subsample():
    ds = _multilabel_ds()
    assert set(ds.labels) == {"BENIGN", "Attack A", "Attack B"}
    sub = subsample(ds, 1000, seed=0)
    assert len(sub.labels) == len(sub.y) == 1000
    assert ((sub.labels != "BENIGN").astype(int) == sub.y).all()   # labels still match y


def test_diagnose_outputs_are_consistent():
    ds = _multilabel_ds()
    res = diagnose(ds, models=["rf"], seeds=[0], gaps=[0.0, 0.1], verbose=False)
    comp = res["composition"]
    assert set(comp.protocol) == {"random", "temporal_per_class", "temporal_per_label", "temporal_global"}
    p = comp.test_attack_frac.to_numpy()
    assert np.allclose(comp.always_attack_f1, 2 * p / (1 + p))
    pl = res["per_label"]
    assert (pl.correct_frac.between(0, 1)).all()
    assert set(pl[pl.kind == "benign"].label) == {"BENIGN"}
    # global temporal cut: the late attack type must dominate the test window, early type may be absent
    tl = res["label_counts"].query("protocol == 'temporal_global' and part == 'test'").set_index("label").n
    assert tl.get("Attack B", 0) > tl.get("Attack A", 0)
    gs = res["gap_sweep"].sort_values("gap_frac")
    assert len(gs) == 2 and gs.n_train.iloc[0] > gs.n_train.iloc[1]    # bigger gap -> smaller train


def test_write_diagnostics_creates_report():
    ds = _multilabel_ds(2000)
    res = diagnose(ds, models=["rf"], seeds=[0], gaps=[0.0], verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        path = write_diagnostics(res, tmp, "syn_multi")
        txt = path.read_text()
        for h in ("Split composition", "classified correctly", "Gap sweep", "always_attack_f1", "### rf", "n_test"):
            assert h in txt


def test_temporal_per_label_split_is_within_each_label_and_ordered():
    from secleak.protocols import split
    ds = _multilabel_ds(4000)
    tr, te = split(ds, "temporal_per_label", test_size=0.3)
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(ds)
    for lab in np.unique(ds.labels):
        t_tr, t_te = ds.time[tr][ds.labels[tr] == lab], ds.time[te][ds.labels[te] == lab]
        assert len(t_tr) > 0 and len(t_te) > 0              # every label present on both sides
        assert t_tr.max() <= t_te.min()                     # test strictly later than train, per label
        n = (ds.labels == lab).sum()
        assert abs(len(t_te) / n - 0.3) < 0.02


def test_temporal_per_label_requires_raw_labels():
    import pytest
    from secleak.protocols import split
    ds = _multilabel_ds(500); ds.labels = None
    with pytest.raises(ValueError):
        split(ds, "temporal_per_label")


def test_cap_label_floor_keeps_rare_labels_and_respects_budget():
    from secleak.experiments import cap_label_floor
    rng = np.random.default_rng(0)
    labels = np.array(["BENIGN"] * 9000 + ["DoS"] * 6000 + ["Rare"] * 30)
    idx = np.arange(len(labels))
    out = cap_label_floor(idx, labels, 8000, rng, floor=5000)
    assert len(out) == 8000 and len(np.unique(out)) == 8000 and (np.diff(out) > 0).all()
    assert (labels[out] == "Rare").sum() == 30                      # all rare rows kept
    assert (labels[out] == "BENIGN").sum() > 0 and (labels[out] == "DoS").sum() >= 0
    assert cap_label_floor(idx, labels, None, rng) is idx           # no cap -> unchanged
