"""Scene estimator pieces that the checks rely on: the camera model, scene_from's raised-start rule, float32
geometry under bf16 autocast, and split loading."""

import numpy as np
import pytest
import torch

from phone2libero import scene as S
from phone2libero.plan import SIM_PLACE_DZ

# A bowl and a plate where they sit at reset (task 2, init state 10), metres
BOWL, PLATE = np.array([-0.0747, 0.0102, 0.9269]), np.array([0.0532, 0.1924, 0.9108])


def test_camera_round_trip():
    p = np.stack([BOWL, PLATE])
    pix = S.project(p)
    assert ((pix > 0) & (pix < S.IMG)).all()
    back = S.backproject(torch.tensor(pix[:, 0]), torch.tensor(pix[:, 1]), torch.tensor(p[:, 2])).numpy()
    assert np.abs(back - p).max() < 1e-9


@pytest.mark.parametrize("bowl_lo, raised", [(0.8985, False), (1.1270, True)])
def test_scene_from_raised_start(bowl_lo, raised):
    plate_lo = 0.9025
    c0 = BOWL + [0, 0, bowl_lo - 0.8985]
    y = np.r_[c0, PLATE, bowl_lo, plate_lo]
    sc = S.scene_from(y, np.array([-0.2, 0.0, 1.17]), task=9, side=1)
    assert np.array_equal(sc.bowl_c0, c0) and np.array_equal(sc.plate_c, PLATE)
    if raised:  # placed relative to the plate, as replay.scene_of does
        assert sc.place_z == pytest.approx(plate_lo + (c0[2] - bowl_lo) + SIM_PLACE_DZ)
    else:
        assert sc.place_z is None


def test_geometry_stays_float32_under_autocast():
    torch.manual_seed(0)
    model = S._net()(res=64).eval()
    agent = torch.randint(0, 256, (2, 256, 256, 3), dtype=torch.uint8)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        y, pix = model(agent, torch.tensor([0, 9]))
    assert y.dtype == torch.float32
    # each predicted centre lies on its predicted pixel's ray: bf16 would miss by ~0.4 cm (~1 px)
    for k, sl in ((0, slice(0, 3)), (1, slice(3, 6))):
        assert np.abs(S.project(y[:, sl].double().numpy()) - pix[:, k].double().numpy()).max() < 1e-2


def test_load_split(tmp_path):
    rng = np.random.default_rng(0)
    shards = []
    for t, n in ((0, 3), (4, 5)):
        z = {
            "agent": rng.integers(0, 256, (n, 8, 8, 3), dtype=np.uint8),
            "wrist": rng.integers(0, 256, (n, 4, 4, 3), dtype=np.uint8),
            "y": rng.normal(size=(n, 8)),
            "task": np.full(n, t),
        }
        np.savez(tmp_path / f"t{t}_i10.npz", **z)
        shards.append(z)
    d = S.load_split(tmp_path, res=4, wrist=False)
    assert "wrist" not in d
    assert np.array_equal(d["y"], np.concatenate([z["y"] for z in shards]))
    agent = np.concatenate([z["agent"] for z in shards]).reshape(8, 4, 2, 4, 2, 3).mean((2, 4))
    assert np.array_equal(d["agent"], (agent + 0.5).astype(np.uint8))
