"""Per-take processing: hand segments -> takes -> events (close, lift, placed, release) -> 20 Hz trajectories.

Coordinates (cm): x along the object line, positive toward the plate's far (right) side; z height above the
floor card; both relative to the plate centre (measured from the plate's outline). Depth y (toward the
backdrop) is not reported per take: one side camera cannot measure it well (see bowl.fit_rim); the bowl
is assumed to travel in a straight line from its start position on the object line to the plate centre.
"""

from __future__ import annotations

import numpy as np

from . import bowl as B
from . import calib as C

THUMB_TIP, INDEX_TIP, WRIST = 4, 8, 0
RESAMPLE_HZ = 20
SEGMENT_GAP_S = 1.0  # hand detections closer than this belong to one hand-in-frame segment
LIFT_CM = 1.0  # lift start: bowl base 1 cm above its resting height (same rule as the LIBERO demo timing)
CONTACT_MARGIN_CM = 1.5  # contact: pinch point within this of its pre-lift plateau distance to the right rim
OPEN_CM = 2.5  # release: after placement, the pinch point moves this much farther from the bowl centre
DESPIKE_CM = 1.5  # bowl samples this far from the 9-frame running median are fit glitches


# --- setup ----------------------------------------------------------------------------------------


# --- segmentation -----------------------------------------------------------------------------------


def hand_segments(hands: list[dict]) -> list[tuple[int, int]]:
    t = np.array([h["t"] for h in hands])
    idx = [i for i, h in enumerate(hands) if h["hand"] is not None]
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > SEGMENT_GAP_S:
            segs.append((start, a))
            start = b
    segs.append((start, idx[-1]))
    return segs


def classify_segments(segs, bowl_x: np.ndarray, t: np.ndarray, plate_x: float) -> list[dict]:
    """A take moves the bowl from off the plate onto it; a reset does the reverse; else 'adjust'."""
    out = []
    for k, (a, e) in enumerate(segs):
        before = np.median(bowl_x[max(0, a - 15) : a]) if a > 0 else bowl_x[0]
        after = np.median(bowl_x[e + 1 : e + 16]) if e + 1 < len(bowl_x) else bowl_x[-1]
        on_before, on_after = abs(before - plate_x) < 5, abs(after - plate_x) < 5
        kind = "take" if (not on_before and on_after) else "reset" if (on_before and not on_after) else "adjust"
        out.append(
            {
                "segment": k,
                "start": int(a),
                "end": int(e),
                "t_start": float(t[a]),
                "t_end": float(t[e]),
                "kind": kind,
                "bowl_x_before": float(before),
                "bowl_x_after": float(after),
            }
        )
    return out


# --- per-take processing ---------------------------------------------------------------------------


def carry_depth(x: np.ndarray, x_start: float, y_start: float, x_end: float, y_end: float) -> np.ndarray:
    """Straight-line carry in plan view: depth follows progress along x from start to the plate centre."""
    s = np.clip((x - x_start) / (x_end - x_start), 0, 1)
    return y_start + s * (y_end - y_start)


def first_run(mask: np.ndarray, start: int, min_len: int) -> int | None:
    run = 0
    for i in range(start, len(mask)):
        run = run + 1 if mask[i] else 0
        if run >= min_len:
            return i - min_len + 1
    return None


def process_take(take: dict, hands: list[dict], bowl_recs: list[dict], scene: dict) -> dict:
    cam: C.Camera = scene["camera"]
    R = scene["bowl_rim_radius_cm"]
    px, py = scene["plate_centre_xy_cm"]
    a, e = take["start"], take["end"]
    # include 1 s after the hand leaves so the bowl's final rest is in the record
    t_all = np.array([h["t"] for h in hands])
    e2 = min(len(hands) - 1, int(np.searchsorted(t_all, t_all[e] + 1.0)))
    idx = np.arange(a, e2 + 1)
    t = t_all[idx]

    # bowl: refit each rim-tracked frame at the carry-path depth
    raw = np.array([bowl_recs[i]["centre"] for i in idx])
    x_start = float(np.median([bowl_recs[i]["centre"][0] for i in range(max(0, a - 15), a)]))
    y_start = 0.0  # fallback: the bowl starts on the object line
    # origin is the plate centre: measure where the bowl rests before the hand arrives
    start_pose = B.rest_pose(cam, bowl_recs, range(max(0, a - 30), a), R)
    if start_pose is not None:
        x_start, y_start = start_pose["centre"][0], start_pose["centre"][1]
    centres, sources, rms = [], [], []
    for i, c in zip(idx, raw):
        r = bowl_recs[i]
        y = float(carry_depth(np.array([c[0]]), x_start, y_start, px, py)[0])
        if r["source"] == "rim" and len(r.get("arc_px", [])) >= 15:
            for _ in range(2):  # x barely depends on y; two passes settle the depth along the path
                x, info = B.fit_rim(cam, np.array(r["arc_px"]), R, np.array(c), y=y)
                y = float(carry_depth(np.array([x[0]]), x_start, y_start, px, py)[0])
            centres.append(x)
            rms.append(info["rms_cm"])
        else:
            centres.append(np.array([c[0], y, c[2]]))
            rms.append(np.nan)
        sources.append(r["source"])
    centres = np.array(centres)
    # despike: short fit glitches (e.g. latching onto the plate edge or the hand during the descent)
    # 9-frame running median: glitches last up to 3 frames (~0.1 s), shorter than any real motion change
    med = np.stack([np.median(np.stack([np.roll(centres[:, j], s) for s in range(-4, 5)]), 0) for j in range(3)], 1)
    spike = np.linalg.norm((centres - med)[:, [0, 2]], axis=1) > DESPIKE_CM
    spike[:4] = spike[-4:] = False
    good = ~spike
    for j in (0, 2):
        centres[spike, j] = np.interp(np.where(spike)[0], np.where(good)[0], centres[good, j])
    sources = [("despiked" if sp else s) for s, sp in zip(sources, spike)]
    bowl_x = centres[:, 0] - px
    bowl_rim_z = centres[:, 2]
    bowl_base_z = bowl_rim_z - C.BOWL_H
    rest_base = float(np.median(bowl_base_z[:10]))

    # hand: thumb tip, index tip, wrist on the vertical plane at the bowl's depth
    hand = np.full((len(idx), 3, 3), np.nan)
    for k, (i, y) in enumerate(zip(idx, centres[:, 1])):
        h = hands[i]["hand"]
        if h is not None:
            lm = np.array(h["landmarks_px"])[[THUMB_TIP, INDEX_TIP, WRIST]]
            hand[k] = cam.to_plane(lm, y=y)
    pinch = 0.5 * (hand[:, 0] + hand[:, 1])
    aperture = np.linalg.norm(hand[:, 0] - hand[:, 1], axis=1)
    right_rim = np.stack([centres[:, 0] + R, centres[:, 2]], 1)
    d_rim = np.linalg.norm(pinch[:, [0, 2]] - right_rim, axis=1)
    d_bowl = np.linalg.norm(pinch[:, [0, 2]] - centres[:, [0, 2]], axis=1)

    # events
    i_lift = first_run(bowl_base_z > rest_base + LIFT_CM, 0, 3)
    i_close = None
    if i_lift is not None:
        # The pinch point settles at a grasp-specific distance from the right-rim point (fingertips sit
        # 1-3 cm below the rim). Contact = the run, leading into the lift, within a margin of that plateau.
        w0 = int(np.searchsorted(t, t[i_lift] - 0.4))
        plateau = np.nanmedian(d_rim[w0 : i_lift + 1])
        near = np.nan_to_num(d_rim, nan=1e9) < plateau + CONTACT_MARGIN_CM
        i_close = i_lift
        while i_close > 0 and near[i_close - 1]:
            i_close -= 1
    # at rest on the plate: over it, not moving vertically, AND down at its final resting height (a pause at
    # the top of the descent, before going straight down onto the plate, is also slow and over the plate)
    final_base = float(np.median(bowl_base_z[-10:]))
    on_plate = (
        (np.abs(centres[:, 0] - px) < 4) & (np.abs(np.gradient(bowl_rim_z, t)) < 3) & (bowl_base_z < final_base + 1.5)
    )
    i_placed = first_run(on_plate, i_lift or 0, 8)
    i_release = None
    if i_placed is not None:
        ref = np.nanmedian(d_bowl[i_placed : i_placed + 5])
        away = np.nan_to_num(d_bowl, nan=1e9) > ref + OPEN_CM
        i_release = first_run(away, i_placed, 3)
    i_leave = e - a
    events = {
        "hand_enters": 0,
        "close": i_close,
        "lift": i_lift,
        "placed": i_placed,
        "release": i_release,
        "hand_leaves": i_leave,
    }
    ev_t = {k: (float(t[v] - t[0]) if v is not None else None) for k, v in events.items()}
    phases = {}
    if None not in (i_close, i_lift, i_release):
        phases = {
            "approach": ev_t["close"] - ev_t["hand_enters"],
            "close": ev_t["lift"] - ev_t["close"],
            "carry": ev_t["release"] - ev_t["lift"],
            "release": ev_t["hand_leaves"] - ev_t["release"],
            # carry split: moving the bowl vs holding it on the plate before letting go
            "carry_transport": ev_t["placed"] - ev_t["lift"],
            "carry_settle": ev_t["release"] - ev_t["placed"],
        }
    gripper = np.zeros(len(idx), bool)
    if i_close is not None and i_release is not None:
        gripper[i_close:i_release] = True

    # hand detection inside the pinch window (from the bowl, independent of the hand tracker):
    # 0.3 s before lift start until the bowl is placed
    det = np.array([hands[i]["hand"] is not None for i in idx])
    pinch_rate = None
    if i_lift is not None and i_placed is not None:
        w0 = int(np.searchsorted(t, t[i_lift] - 0.3))
        pinch_rate = float(det[w0 : i_placed + 1].mean())

    # resample to 20 Hz on the take's own clock (hand enters = 0)
    tt = t - t[0]
    grid = np.arange(0, tt[-1] + 1e-9, 1 / RESAMPLE_HZ)

    def interp(v):
        ok = ~np.isnan(v)
        return np.interp(grid, tt[ok], v[ok]) if ok.sum() > 1 else np.full_like(grid, np.nan)

    rows = {
        "time_s": grid,
        "hand_x_cm": interp(pinch[:, 0] - px),
        "hand_z_cm": interp(pinch[:, 2]),
        "wrist_x_cm": interp(hand[:, 2, 0] - px),
        "wrist_z_cm": interp(hand[:, 2, 2]),
        "gripper_closed": np.interp(grid, tt, gripper.astype(float)) > 0.5,
        "bowl_x_cm": interp(bowl_x),
        "bowl_z_cm": interp(bowl_base_z + C.BOWL_H / 2),
        "bowl_base_z_cm": interp(bowl_base_z),
        "thumb_index_aperture_cm": interp(aperture),
    }
    # hand samples past the last detection are extrapolation: blank them
    last_det = tt[det][-1]
    for k in ("hand_x_cm", "hand_z_cm", "wrist_x_cm", "wrist_z_cm", "thumb_index_aperture_cm"):
        rows[k][grid > last_det] = np.nan

    src = np.array(sources)
    start_gap_rim = -(C.PLATE_D / 2) - (x_start - px + R)
    return {
        "take": take,
        "t0_video_s": float(t[0]),
        "events_s": ev_t,
        "phases_s": phases,
        "start_bowl_x_cm": float(x_start - px),
        "start_bowl_y_cm": float(y_start - py),
        "start_pose_measured": start_pose is not None,
        "start_gap_to_bowl_rim_cm": float(start_gap_rim),
        "rest_base_z_cm": rest_base,
        "peak_base_lift_cm": float(np.nanmax(bowl_base_z) - rest_base),
        "peak_base_z_cm": float(np.nanmax(bowl_base_z)),
        "final_bowl_x_cm": float(np.median(bowl_x[-10:])),
        "final_base_z_cm": float(np.median(bowl_base_z[-10:])),
        "bowl_sources": {s: int((src == s).sum()) for s in ("rim", "despiked", "coarse", "hold")},
        "bowl_sources_lift_to_placed": (
            {s: int((src[i_lift : i_placed + 1] == s).sum()) for s in ("rim", "despiked", "coarse", "hold")}
            if i_lift is not None and i_placed is not None
            else None
        ),
        "bowl_fit_rms_cm_median": float(np.nanmedian(rms)),
        "max_bowl_step_cm": float(np.nanmax(np.linalg.norm(np.diff(centres[:, [0, 2]], axis=0), axis=1))),
        "contact_plateau_cm": float(plateau) if i_lift is not None else None,
        "hand_detection_rate": float(det[: e - a + 1].mean()),
        "hand_detection_rate_pinch_window": pinch_rate,
        "per_frame": {
            "t": tt.tolist(),
            "bowl": centres.tolist(),
            "pinch": pinch.tolist(),
            "hand": hand.tolist(),
            "d_rim": d_rim.tolist(),
            "gripper": gripper.tolist(),
            "source": sources,
            "frames": idx.tolist(),
        },
        "resampled": {k: np.asarray(v).tolist() for k, v in rows.items()},
    }
