import pytest

from secleak.synthetic import make_synthetic
from secleak.webcore import audit_frame, guess_benign_values, guess_label_col


def _df():
    df = make_synthetic(n=4000, seed=1)
    return df


def test_autodetect_and_audit():
    df = _df()
    res = audit_frame(df, max_rows=3000)
    assert res["label_col"] == guess_label_col(df)
    assert 0 <= res["card"]["duplication"]["exact_removable_frac"] <= 1
    assert "Audit card" in res["markdown"]


def test_no_timestamp_warns_proxy():
    df = _df().drop(columns=["timestamp"])
    res = audit_frame(df, max_rows=3000)
    assert any("ROW ORDER" in w for w in res["warnings"])


def test_single_class_error():
    df = _df(); df["label"] = "BENIGN"
    with pytest.raises(ValueError):
        audit_frame(df)


def test_benign_guess():
    assert "BENIGN" in guess_benign_values(_df(), "label")
