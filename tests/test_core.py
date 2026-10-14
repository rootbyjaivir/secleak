import numpy as np
import pandas as pd

from secleak.audit.dedup import dedup_mask, dup_report, row_hashes, train_test_overlap
from secleak.audit.shortcut import shortcut_scan
from secleak.data import frame_to_dataset
from secleak.protocols import PROTOCOLS, split, validation_split
from secleak.synthetic import make_synthetic

CFG = dict(label_col="label", benign_values=["BENIGN"], time_col="timestamp",
           time_kind="iso", drop_cols=["timestamp"])


def _ds(**kw):
    return frame_to_dataset(make_synthetic(n=4000, seed=1, **kw), "syn", CFG)


def test_splits_are_disjoint_and_cover_both_classes():
    ds = _ds()
    for p in PROTOCOLS:
        tr, te = split(ds, p, seed=0)
        assert len(np.intersect1d(tr, te)) == 0, p
        assert len(np.unique(ds.y[tr])) == 2 and len(np.unique(ds.y[te])) == 2, p


def test_temporal_global_never_trains_on_the_future():
    ds = _ds()
    tr, te = split(ds, "temporal_global")
    assert ds.time[tr].max() <= ds.time[te].min()


def test_temporal_per_class_respects_order_within_class():
    ds = _ds()
    tr, te = split(ds, "temporal_per_class")
    for c in (0, 1):
        assert ds.time[tr[ds.y[tr] == c]].max() <= ds.time[te[ds.y[te] == c]].min()


def test_gap_embargo_creates_separation():
    ds = _ds()
    tr, te = split(ds, "temporal_global", gap_frac=0.05)
    assert len(tr) + len(te) < len(ds)


def test_validation_split_stays_inside_train_and_same_protocol():
    ds = _ds()
    tr, _ = split(ds, "temporal_global")
    t2, v2 = validation_split(ds, "temporal_global", tr)
    assert set(t2) | set(v2) <= set(tr)
    assert ds.time[t2].max() <= ds.time[v2].min()


def test_dedup_counts_planted_duplicates():
    X = pd.DataFrame({"a": [1.0, 1.0, 2.0, 3.0, 3.0], "b": [0, 0, 1, 1, 1]})
    y = np.array([0, 0, 1, 0, 1])           # rows 3,4 identical but conflicting labels
    r = dup_report(X, y)
    assert abs(r["exact_removable_frac"] - 2 / 5) < 1e-9
    assert r["label_conflict_groups"] == 1
    keep = dedup_mask(X, y, drop_conflicts=True)
    assert keep.tolist() == [True, False, True, False, False]


def test_near_duplicate_key_collapses_tiny_differences():
    X = pd.DataFrame({"a": [1.00001, 1.00002, 5.0]})
    assert len(set(row_hashes(X))) == 3
    assert len(set(row_hashes(X, sig=4))) == 2


def test_overlap_detects_leak_in_random_but_not_when_deduped():
    ds = _ds(dup_frac=0.3)
    tr, te = split(ds, "random", seed=0)
    leak = train_test_overlap(ds.X, tr, te, y=ds.y)["test_rows_in_train_frac"]
    tr2, te2 = split(ds, "random_dedup", seed=0)
    clean = train_test_overlap(ds.X, tr2, te2, y=ds.y)["test_rows_in_train_frac"]
    assert leak > 0.1 and clean == 0.0


def test_shortcut_scan_finds_planted_port_shortcut():
    ds = frame_to_dataset(make_synthetic(n=4000, shortcut_strength=0.99, seed=2), "syn", CFG)
    top = shortcut_scan(ds, top_k=3)
    assert "dst_port" in top["feature"].tolist()[:2]


def test_drift_creates_random_vs_temporal_gap():
    from secleak.models import make_model
    from sklearn.metrics import f1_score
    ds = _ds()
    scores = {}
    for p in ("random", "temporal_per_class"):
        tr, te = split(ds, p, seed=0)
        m = make_model("gbdt").fit(ds.X.iloc[tr], ds.y[tr])
        scores[p] = f1_score(ds.y[te], m.predict(ds.X.iloc[te]))
    assert scores["random"] - scores["temporal_per_class"] > 0.1
