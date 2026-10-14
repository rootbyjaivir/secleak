import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from secleak.data import load_config, load_dataset


def _write_cic(root: Path, n=400, seed=0):
    d = root / "data/raw/CICIDS2017/TrafficLabelling"
    d.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    hours = rng.integers(8, 17, n)               # 8..16 -> exported WITHOUT am/pm (13..16 -> 1..4)
    h12 = np.where(hours > 12, hours - 12, hours)
    mins = rng.integers(0, 60, n)
    ts = [f"3/7/2017 {h}:{m:02d}" for h, m in zip(h12, mins)]
    df = pd.DataFrame({
        "Flow ID": [f"id{i}" for i in range(n)], " Source IP": "1.1.1.1", " Source Port": 1234,
        " Destination IP": "2.2.2.2", " Destination Port": rng.choice([80, 443, 22], n),
        " Protocol": 6, " Timestamp": ts, " Flow Duration": rng.integers(1, 10**6, n),
        " Total Fwd Packets": rng.integers(1, 20, n), " Total Backward Packets": rng.integers(0, 20, n),
        "Total Length of Fwd Packets": rng.integers(0, 5000, n),
        " Total Length of Bwd Packets": rng.integers(0, 5000, n),
        " Label": rng.choice(["BENIGN", "DoS Hulk"], n)})
    df[" Flow Duration"] = df[" Flow Duration"].astype(float)
    df.loc[0, " Flow Duration"] = np.inf
    df.to_csv(d / "Tuesday.csv", index=False, encoding="cp1252")


def test_cicids_loader_strips_columns_fixes_ampm_and_drops_ids():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _write_cic(root)
        ds = load_dataset("cicids2017_tl", root=root)
        assert "Flow ID" not in ds.X and "Source IP" not in ds.X and "Label" not in ds.X
        assert "Destination Port" in ds.X and not ds.time_is_proxy
        assert set(np.unique(ds.y)) == {0, 1}
        assert np.isfinite(ds.X.select_dtypes("number").to_numpy()[1:]).all()   # inf -> NaN
        hrs = pd.to_datetime(ds.time, unit="s").hour
        assert hrs.min() >= 8                    # afternoon hours restored to 13..16 (not 1..4)


def test_unsw_headerless_loader_and_proxy_time_for_ml_csv():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cfg = load_config()
        d = root / "data/raw/UNSW-NB15"; d.mkdir(parents=True)
        n, k = 200, len(cfg["unsw_nb15_raw"]["names"])
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 100, (n, k)).astype(object)
        names = cfg["unsw_nb15_raw"]["names"]
        arr[:, names.index("Stime")] = 1_400_000_000 + np.arange(n)
        arr[:, names.index("Label")] = rng.integers(0, 2, n)
        for c in ("srcip", "dstip", "proto", "state", "service", "attack_cat"):
            arr[:, names.index(c)] = "x"
        pd.DataFrame(arr).to_csv(d / "UNSW-NB15_1.csv", header=False, index=False)
        ds = load_dataset("unsw_nb15_raw", root=root)
        assert len(ds) == n and not ds.time_is_proxy and "Stime" not in ds.X

        d2 = root / "data/raw/CICIDS2017/MachineLearningCVE"; d2.mkdir(parents=True)
        pd.DataFrame({" Flow Duration": [1, 2, 3], " Label": ["BENIGN", "DoS", "BENIGN"]}) \
            .to_csv(d2 / "a.csv", index=False)
        ds2 = load_dataset("cicids2017_ml", root=root)
        assert ds2.time_is_proxy and np.all(np.diff(ds2.time) > 0)


def test_mixed_timestamp_formats_and_blank_rows_are_not_silently_lost():
    """Regression: Monday file has seconds, others don't; Thursday-Morning ends with blank rows."""
    import pytest
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp); _write_cic(root, n=300)
        d = root / "data/raw/CICIDS2017/TrafficLabelling"
        # file with seconds-format timestamps (like Monday-WorkingHours)
        sec = pd.read_csv(d / "Tuesday.csv", encoding="cp1252")
        sec[" Timestamp"] = "03/07/2017 09:15:58"
        sec.to_csv(d / "Monday.csv", index=False, encoding="cp1252")
        # file with trailing fully-empty rows
        blank = sec.copy()
        blank[" Timestamp"] = "6/7/2017 9:15"
        pad = pd.DataFrame(np.nan, index=range(50), columns=blank.columns)
        pd.concat([blank, pad]).to_csv(d / "Thursday.csv", index=False, encoding="cp1252")
        ds = load_dataset("cicids2017_tl", root=root)
        assert len(ds) == 900                       # nothing lost: 300 + 300 + 300 (50 blanks dropped)
        assert ds.meta["n_blank_dropped"] == 50 and ds.meta["n_bad_time_dropped"] == 0
        # genuinely bad timestamps in >0.1% of rows must raise, never drop silently
        bad = pd.read_csv(d / "Tuesday.csv", encoding="cp1252"); bad[" Timestamp"] = "garbage"
        bad.to_csv(d / "Tuesday.csv", index=False, encoding="cp1252")
        with pytest.raises(ValueError):
            load_dataset("cicids2017_tl", root=root)


def test_category_col_supplies_raw_labels_and_fills_blank_normals():
    from secleak.data import frame_to_dataset
    df = pd.DataFrame({"f": np.arange(8.0), "Label": [0, 0, 1, 1, 1, 0, 1, 1],
                       "attack_cat": [np.nan, " ", "Fuzzers ", "DoS", "Fuzzers", np.nan, np.nan, "DoS"],
                       "Stime": np.arange(8) * 10.0})
    cfg = dict(label_col="Label", attack_value=1, category_col="attack_cat", category_fill="Normal",
               time_col="Stime", time_kind="epoch_s", drop_cols=["attack_cat", "Stime"])
    ds = frame_to_dataset(df, "unsw_like", cfg)
    assert list(ds.labels) == ["Normal", "Normal", "Fuzzers", "DoS", "Fuzzers", "Normal",
                               "attack_uncategorised", "DoS"]
    assert list(ds.y) == [0, 0, 1, 1, 1, 0, 1, 1]
    assert "attack_cat" not in ds.X.columns            # category must never leak into the features
