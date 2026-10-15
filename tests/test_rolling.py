import tempfile
import numpy as np
import pandas as pd

from secleak.rolling import assign_blocks, make_fold, rolling, summarize, write_rolling
from test_diagnose import _multilabel_ds


def test_blocks_are_time_ordered_per_label_and_balanced():
    ds = _multilabel_ds(4000)
    b = assign_blocks(ds, 5)
    for lab in np.unique(ds.labels):
        m = ds.labels == lab
        order = np.argsort(ds.time[m], kind="stable")
        assert (np.diff(b[m][order]) >= 0).all()              # blocks never go backwards in time
        counts = np.bincount(b[m], minlength=5)
        assert counts.max() - counts.min() <= 1               # equal-count blocks


def test_fold_past_precedes_test_and_control_is_size_and_label_matched():
    ds = _multilabel_ds(4000)
    b = assign_blocks(ds, 5)
    rng = np.random.default_rng(0)
    past, ctrl, test = make_fold(ds, b, 3, rng)
    assert set(past).isdisjoint(test) and set(ctrl).isdisjoint(test)    # control never sees the test block
    assert (b[past] < 3).all() and (b[test] == 3).all()
    for lab in np.unique(ds.labels[past]):
        assert (ds.labels[ctrl] == lab).sum() == (ds.labels[past] == lab).sum()
        assert ds.time[past][ds.labels[past] == lab].max() <= ds.time[test][ds.labels[test] == lab].min()
    assert (b[ctrl] > 3).any()                                          # control really contains future rows


def test_rolling_end_to_end_reports_positive_leak_on_drifting_data():
    ds = _multilabel_ds(6000)
    df = rolling(ds, n_blocks=5, first_origin=2, models=["rf"], seeds=[0], verbose=False)
    assert set(df.protocol) == {"rolling_past", "blocked_matched"} and set(df.origin) == {2, 3}
    res = summarize(df)
    s = res["summary"].iloc[0]
    assert s.n_origins == 2 and s.ci_lo <= s.delta_mean <= s.ci_hi
    assert s.delta_mean > 0                                             # synthetic drift: seeing the future helps
    with tempfile.TemporaryDirectory() as tmp:
        txt = write_rolling(df, tmp, "syn", 5, 2).read_text()
        assert "Summary over origins" in txt and "Test windows" in txt


def test_last_block_is_never_an_origin_and_bad_origins_are_rejected():
    import pytest
    ds = _multilabel_ds(3000)
    with pytest.raises(ValueError):
        rolling(ds, n_blocks=5, first_origin=4, models=["logreg"], seeds=[0], verbose=False)
    df = rolling(ds, n_blocks=5, first_origin=1, models=["logreg"], seeds=[0], verbose=False)
    assert df.origin.max() == 3


def test_explain_origin_breakdowns_are_consistent():
    from secleak.rolling import explain_origin, write_explain
    ds = _multilabel_ds(5000)
    b = assign_blocks(ds, 5)
    res = explain_origin(ds, 5, 3, model="rf", seed=0)
    n_test = (b == 3).sum()
    assert res["meta"]["n_test"] == n_test and res["labels"].n_test.sum() == n_test
    n_ben = ((b == 3) & (ds.y == 0)).sum()
    assert res["fp_by_day"].n_benign.sum() == n_ben                      # every benign test row bucketed once
    assert (res["fp_by_day"].false_alarms <= res["fp_by_day"].n_benign).all()
    d = res["drift"].shift_in_pooled_std.abs().to_numpy()
    assert len(d) > 0 and (np.diff(d) <= 1e-12).all()                    # sorted by absolute shift
    with tempfile.TemporaryDirectory() as tmp:
        txt = write_explain(res, tmp, "syn").read_text()
        assert "False alarms on benign flows by day" in txt and "covariate shift" in txt


def test_drift_ranking_is_not_dominated_by_near_constant_features():
    from secleak.rolling import explain_origin
    ds = _multilabel_ds(5000)
    ds.X["almost_const"] = 0.0
    ds.X.loc[ds.X.index[-40:], "almost_const"] = 1.0        # flips on in the last 40 rows only
    res = explain_origin(ds, 5, 3, model="rf", seed=0)
    assert res["drift"].shift_in_pooled_std.abs().max() < 50  # old past-std scaling gave ~1e8 here
