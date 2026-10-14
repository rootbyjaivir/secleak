import pandas as pd
from secleak.census import cohen_kappa, wilson


def test_wilson_interval_contains_proportion():
    lo, hi = wilson(30, 100)
    assert lo < 0.30 < hi and 0 <= lo and hi <= 1


def test_kappa_perfect_and_chance():
    a = pd.Series(["y", "n", "y", "n"])
    assert cohen_kappa(a, a) == 1.0
    assert cohen_kappa(a, pd.Series(["y", "y", "n", "n"])) <= 0.0
