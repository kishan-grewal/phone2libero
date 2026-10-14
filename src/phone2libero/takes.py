"""The recorded takes (data/takes, written by the capture project) as plain arrays. No sim, no lerobot."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TAKES = Path("data/takes")
VIDEOS = {"G1_gap16": 16, "G2_gap12": 12, "G3_gap2": 2}  # video -> start tick: bowl-to-plate gap in cm


@dataclass
class Take:
    """One take at 20 Hz. Real-world frame: x along the bowl-plate line (+ toward the plate, plate centre = 0),
    z above the card, cm. Time 0 is when the hand enters the frame; events are seconds on the same clock."""

    name: str  # e.g. "G1_gap16/05"
    video: str
    tick: int
    t: np.ndarray
    hand: np.ndarray  # (n, 2) pinch point (thumb-index midpoint) x, z
    bowl_x: np.ndarray
    bowl_base_z: np.ndarray
    events: dict[str, float]  # close, lift, placed, release (and hand_leaves)
    start_bowl_x: float
    final_bowl_x: float
    rest_base_z: float
    final_base_z: float
    flags: list[str]  # quality notes from the capture report (e.g. "lift high"); informational only


def _col(rows: list[dict], key: str) -> np.ndarray:
    a = np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])
    bad = np.isnan(a)  # hand dropouts: < 1% of frames, none inside the pinch windows; filled linearly
    if bad.any() and (~bad).any():
        a[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), a[~bad])
    return a


def load_takes(root: Path = TAKES) -> list[Take]:
    flags = json.loads((root / "summary.json").read_text())["videos"]
    out = []
    for video, tick in VIDEOS.items():
        res = json.loads((root / video / "results.json").read_text())
        for k, tk in enumerate(res["takes"], 1):
            with open(root / video / f"take_{k:02d}.csv") as f:
                rows = list(csv.DictReader(f))
            out.append(
                Take(
                    name=f"{video}/{k:02d}",
                    video=video,
                    tick=tick,
                    t=_col(rows, "time_s"),
                    hand=np.stack([_col(rows, "hand_x_cm"), _col(rows, "hand_z_cm")], 1),
                    bowl_x=_col(rows, "bowl_x_cm"),
                    bowl_base_z=_col(rows, "bowl_base_z_cm"),
                    events=tk["events_s"],
                    start_bowl_x=tk["start_bowl_x_cm"],
                    final_bowl_x=tk["final_bowl_x_cm"],
                    rest_base_z=tk["rest_base_z_cm"],
                    final_base_z=tk["final_base_z_cm"],
                    flags=flags[video]["flags"][k - 1],
                )
            )
    return out
