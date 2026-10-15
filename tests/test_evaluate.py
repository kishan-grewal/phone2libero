import numpy as np

from phone2libero.evaluate import paired_diff, rate


def outcomes(seeds: int, p: np.ndarray, rng) -> dict:
    return {(s, t, e): bool(rng.random() < p[t, e]) for s in range(seeds) for t in range(10) for e in range(10)}


def test_paired_diff():
    rng = np.random.default_rng(1)
    a = outcomes(3, np.full((10, 10), 0.6), rng)
    d, lo, hi, n = paired_diff(a, a, list(range(10)))
    assert (d, lo, hi, n) == (0.0, 0.0, 0.0, 300)
    b = {k: True for k in a}
    d, lo, hi, _ = paired_diff(b, a, list(range(10)))
    assert abs(d - (100 - rate(a, list(range(10)))[0])) < 1e-9 and lo <= d <= hi


def test_repeating_scenes_over_more_seeds_does_not_narrow_the_interval():
    # Seeds that repeat a scene's outcome add no information about other scenes; an episode-level interval
    # would shrink by sqrt(10) here.
    rng = np.random.default_rng(2)
    one = outcomes(1, rng.random((10, 10)), rng)
    ten = {(s, t, e): v for (_, t, e), v in one.items() for s in range(10)}
    tasks = list(range(10))
    assert rate(ten, tasks) == rate(one, tasks)
    never = {k: False for k in ten}
    assert paired_diff(ten, never, tasks)[:3] == paired_diff(one, never, tasks)[:3]
