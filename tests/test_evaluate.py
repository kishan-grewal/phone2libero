import numpy as np

from phone2libero.evaluate import paired_diff, rate, wilson


def test_wilson_known_value():
    lo, hi = wilson(50, 100)
    assert abs(lo - 40.4) < 0.1 and abs(hi - 59.6) < 0.1


def test_paired_diff():
    rng = np.random.default_rng(1)
    a = {(s, t, e): bool(rng.random() < 0.6) for s in range(3) for t in range(10) for e in range(10)}
    d, lo, hi, n = paired_diff(a, a, list(range(10)))
    assert (d, lo, hi, n) == (0.0, 0.0, 0.0, 300)
    b = {k: True for k in a}
    d, lo, hi, _ = paired_diff(b, a, list(range(10)))
    assert abs(d - (100 - rate(a, list(range(10)))[0])) < 1e-9 and lo <= d <= hi
