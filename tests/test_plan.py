"""The retargeting map on the 50 recorded takes: the bowl path starts at the sim bowl and ends on the sim plate."""

import numpy as np
import pytest

from phone2libero.plan import HZ, SIM_PLACE_DZ, T_BLEND, SimScene, make_plan
from phone2libero.takes import load_takes

TAKES = load_takes()
OFFSET = np.array([0.0, 0.045, 0.008])


def scene(**kw) -> SimScene:
    return SimScene(
        bowl_c0=np.array([0.05, -0.10, 0.90]),
        plate_c=np.array([0.0, 0.20, 0.88]),
        eef0=np.array([-0.20, 0.0, 1.17]),
        grasp_offset=OFFSET,
        **kw,
    )


def test_all_takes_load():
    assert len(TAKES) == 50
    for t in TAKES:
        assert np.allclose(np.diff(t.t), 1 / HZ)
        ev = t.events
        assert ev["close"] < ev["lift"] < ev["placed"] < ev["release"]


@pytest.mark.parametrize("take", TAKES, ids=lambda t: t.name)
def test_path_starts_at_bowl_and_ends_on_plate(take):
    sc = scene()
    p = make_plan(take, sc)
    assert np.allclose(p.pos[0], sc.eef0)  # starts where the robot is
    assert np.linalg.norm(p.pos[p.steps["close"]] - (sc.bowl_c0 + OFFSET)) < 0.01  # grasps at the bowl
    placed = p.pos[p.steps["placed"]] - OFFSET
    assert np.linalg.norm(placed[:2] - sc.plate_c[:2]) < 0.01  # sets it down on the plate centre
    assert abs(placed[2] - (sc.bowl_c0[2] + SIM_PLACE_DZ)) < 0.015
    closed = p.grip > 0
    assert closed[p.steps["close"]] and closed[p.steps["placed"]] and not closed[p.steps["release"]]
    assert not closed[: int(T_BLEND * HZ)].any()


def test_raised_start_ends_at_plate_level():
    place_z = 0.80
    p = make_plan(TAKES[0], scene(place_z=place_z))
    assert abs(p.pos[p.steps["placed"], 2] - OFFSET[2] - place_z) < 0.015


@pytest.mark.parametrize("take", TAKES[::7], ids=lambda t: t.name)
def test_enclosed_start_rises_before_moving(take):
    sc = scene(clear_lift=0.14)
    p = make_plan(take, sc)
    held = p.pos[p.steps["close"] : p.steps["release"]] - OFFSET
    moved = np.linalg.norm(held[:, :2] - sc.bowl_c0[:2], axis=1) > 1e-9
    first_move = np.argmax(moved)
    assert moved.any() and held[first_move, 2] - sc.bowl_c0[2] >= 0.14 - 0.08 - 1e-9
