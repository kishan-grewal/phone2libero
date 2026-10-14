"""Process the recording session: calibrate, track the hand and bowl, cut takes, write 20 Hz trajectories.

  python -m phone_capture.session --videos G1_gap16 G2_gap12 G3_gap2 [--overlay]

Inputs: data/raw/<video>.mp4 and the MediaPipe model (data/models/hand_landmarker.task).
Outputs: data/takes/<video>/take_NN.csv + results.json and data/takes/summary.json (what the retargeting reads);
plots, an optional overlay video, caches and report.md in outputs/capture/.

Calibration: all videos share one view (background registration to G1: < 2 px), so the camera is fitted once,
to G1's empty opening frame (calib.fit_camera_plate). The plate outline also comes from that frame and is
carried to the other videos by a background homography (in G3 the bowl touches the plate at the start).
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from . import bowl as B
from . import calib as C
from . import objects as O
from . import outputs as OUT
from . import pipeline as P
from .hands import track_hands
from .video import frame_at, iter_frames, video_info

RAW = Path("data/raw")
TAKES = Path("data/takes")
WORK = Path("outputs/capture")
VIDEOS = {"G1_gap16": 16, "G2_gap12": 12, "G3_gap2": 2}  # video -> start tick (bowl-to-plate gap, cm)
REF_VIDEO = "G1_gap16"
F_TEST = 1465.6  # focal length (px) from a tape-measure calibration of the same phone lens and settings
EMPTY_T = 1.0  # s: an empty frame (no hand) at the start of every video
# Regions that differ between the empty frames (bowl at different ticks, hand at the right edge): excluded
# from the background homography.
BG_EXCLUDE = [(100, 450, 950, 950), (1700, 0, 1920, 1080)]
LIBERO_PHASES = {"approach": 2.3, "close": 0.7, "carry": 2.2, "release": 0.4}  # s, median 5-shot demo


def calibrate_view(img: np.ndarray, plate_contour: np.ndarray) -> tuple[C.Camera, dict]:
    c0 = np.array([-15.0, -60.0, 33.0])  # rough lens position, as a starting point only
    return C.fit_camera_plate(
        C.card_end_points(img),
        C.backdrop_edge_points(img),
        C.backdrop_marks(img),
        plate_contour,
        C.Camera(F_TEST, C.look_at(c0, [-15, 0, 5]), c0.tolist()),
        F_TEST,
    )


def reference() -> dict:
    """Camera, plate outline and bowl rim radius from G1's empty frame (bowl and plate well apart there)."""
    f = WORK / "reference.json"
    if f.exists():
        return json.loads(f.read_text())
    img = frame_at(str(RAW / f"{REF_VIDEO}.mp4"), EMPTY_T)
    _, plate = sorted(O.components(O.white_mask(img))[:2], key=lambda c: c["centroid"][0])
    cam, info = calibrate_view(img, plate["contour"])
    c = B.coarse_detect(cam, img, plate["mask"], 8.0)
    radius = B.calibrate_radius(cam, img, B.plate_outline_mask(plate["mask"]), c[:2], C.BOWL_H)
    ref = {
        "plate_contour_px": plate["contour"].tolist(),
        "bowl_rim_radius_cm": radius,
        "camera": asdict(cam),
        "calibration": info,
    }
    f.write_text(json.dumps(ref))
    return ref


def setup(video: str, cache: Path) -> dict:
    """Camera, plate mask and the bowl's resting pose for one video (cached in <cache>/scene.json)."""
    img = frame_at(str(RAW / f"{video}.mp4"), EMPTY_T)
    ref = reference()
    H, h_info = C.background_homography(frame_at(str(RAW / f"{REF_VIDEO}.mp4"), EMPTY_T), img, BG_EXCLUDE)
    contour = cv2.perspectiveTransform(np.array(ref["plate_contour_px"], np.float64).reshape(-1, 1, 2), H).reshape(
        -1, 2
    )
    plate_mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(plate_mask, [contour.astype(np.int32)], 1)
    f = cache / "scene.json"
    if f.exists():
        d = json.loads(f.read_text())
        d["camera"], d["plate_mask"] = C.Camera(**d["camera"]), plate_mask
        return d
    # One camera serves the session (G1's) unless this view moved by 3 px or more.
    shift = float(np.linalg.norm(h_info["shift_px_median"]))
    if shift < 3.0:
        cam, cinfo = C.Camera(**ref["camera"]), ref["calibration"]
    else:
        cam, cinfo = calibrate_view(img, contour)
    radius = ref["bowl_rim_radius_cm"]
    marks = C.backdrop_marks(img)
    rest, _ = B.track_frame(
        cam, img, B.plate_outline_mask(plate_mask), radius, B.coarse_detect(cam, img, plate_mask, radius)
    )
    d = {
        "camera": asdict(cam),
        "calibration": cinfo,
        "camera_summary": C.camera_summary(cam),
        "plate_centre_xy_cm": [0.0, 0.0],  # the world origin
        "bowl_rim_radius_cm": radius,
        "bowl_rest_centre": rest.tolist(),
        "homography_from_G1": h_info,
        "checks": {
            "marks_z_cm": cam.to_plane(marks, y=cinfo["backdrop_y_cm"])[:, 2].tolist(),
            "marks_z_expected_cm": cinfo["mark_z_cm"],
            "plate_outline": O.rim_circle(cam, contour, C.PLATE_OUTLINE_Z, "all"),
        },
        "video": video_info(str(RAW / f"{video}.mp4")),
    }
    f.write_text(json.dumps(d, indent=1, default=float))
    d["camera"], d["plate_mask"] = cam, plate_mask
    return d


def run_video(video: str, overlay: bool) -> None:
    work, takes_dir = WORK / video, TAKES / video
    cache = work / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    takes_dir.mkdir(parents=True, exist_ok=True)
    path = str(RAW / f"{video}.mp4")
    scene = setup(video, cache)
    hf = cache / "hands.json"
    hands = json.loads(hf.read_text()) if hf.exists() else track_hands(path, hf)
    bf = cache / "bowl_raw.json"
    if bf.exists():
        recs = json.loads(bf.read_text())
    else:
        recs = B.track_video(
            scene["camera"],
            iter_frames(path),
            scene["plate_mask"],
            scene["bowl_rim_radius_cm"],
            np.array(scene["bowl_rest_centre"]),
        )
        bf.write_text(json.dumps(recs))

    t = np.array([h["t"] for h in hands])
    bx = np.array([r["centre"][0] for r in recs])
    segs = P.classify_segments(P.hand_segments(hands), bx, t, 0.0)
    results = [P.process_take(s, hands, recs, scene) for s in segs if s["kind"] == "take"]
    for k, res in enumerate(results, 1):
        OUT.write_csv(res["resampled"], takes_dir / f"take_{k:02d}.csv")
        OUT.plot_take(res, k, work / f"take_{k:02d}.png")
    det = np.array([h["hand"] is not None for h in hands])
    summary = {
        "video": video,
        "tick": VIDEOS[video],
        "scene": {k: v for k, v in scene.items() if k not in ("camera", "plate_mask")},
        "segments": segs,
        "hand_detection": {
            "frames_total": len(hands),
            "frames_detected": int(det.sum()),
            "within_segments": float(np.mean([det[s["start"] : s["end"] + 1].mean() for s in segs])),
        },
        "takes": [{k: v for k, v in r.items() if k not in ("per_frame", "resampled")} for r in results],
    }
    (takes_dir / "results.json").write_text(json.dumps(summary, indent=1, default=float))
    if overlay:
        bowl_all = np.array([r["centre"] for r in recs], float)
        labels = [{"text": "no hand", "gripper": None} for _ in hands]
        for s in segs:
            for i in range(s["start"], s["end"] + 1):
                labels[i] = {"text": f"segment {s['segment'] + 1}: {s['kind']}", "gripper": None}
        for k, res in enumerate(results, 1):
            pf, ev = res["per_frame"], res["events_s"]
            for j, (i, tt) in enumerate(zip(pf["frames"], pf["t"])):
                bowl_all[i] = pf["bowl"][j]
                phase = (
                    "approach"
                    if ev["close"] is None or tt < ev["close"]
                    else "close"
                    if tt < (ev["lift"] or 1e9)
                    else "carry"
                    if tt < (ev["release"] or 1e9)
                    else "release"
                )
                g, b = pf["gripper"][j], np.array(pf["bowl"][j])
                labels[i] = {
                    "text": f"{video} take {k} | {phase} | gripper {'CLOSED' if g else 'open'} | "
                    f"bowl x {b[0]:+.1f} base z {b[2] - C.BOWL_H:+.1f} cm",
                    "gripper": g,
                }
        OUT.overlay_video(
            path, work / "overlay.mp4", scene["camera"], hands, bowl_all, scene["bowl_rim_radius_cm"], labels
        )
    print(f"{video}: {len(segs)} hand segments, {len(results)} takes", flush=True)


def take_flags(video: str, k: int, t: dict) -> list[str]:
    """Quality notes for one take (thresholds in the report). Informational: the sim replay decides what is used."""
    with open(TAKES / video / f"take_{k:02d}.csv") as f:
        rows = list(csv.DictReader(f))

    def col(key):
        return np.array([float(x[key]) if x[key] != "" else np.nan for x in rows])

    tt, hx, hz, bx, bz = col("time_s"), col("hand_x_cm"), col("hand_z_cm"), col("bowl_x_cm"), col("bowl_base_z_cm")
    ev, ph = t["events_s"], t["phases_s"]
    flags = []
    if ev["lift"] is not None and ev["placed"] is not None:
        sel = (tt >= ev["lift"] + 0.3) & (tt <= ev["placed"] - 0.3)
        rel = np.stack([hx - bx, hz - bz], 1)[sel]
        rel = rel[~np.isnan(rel).any(1)]
        # A regrip is a sudden jump of the pinch point relative to the bowl; a slow drift of a few cm is the
        # hand tilting during the carry.
        jump = np.linalg.norm(rel[10:] - rel[:-10], axis=1).max() if len(rel) > 10 else 0.0
        if jump > 4:
            flags.append(f"regrip? (hand jumps {jump:.1f} cm relative to the bowl within 0.5 s mid-carry)")
    if t["peak_base_lift_cm"] > 18:
        flags.append(f"lift high ({t['peak_base_lift_cm']:.0f} cm)")
    if t["peak_base_lift_cm"] < 9:
        flags.append(f"lift low ({t['peak_base_lift_cm']:.0f} cm)")
    if ph and ph["close"] > 1.7:
        flags.append(f"slow close ({ph['close']:.1f} s)")
    if ph and ph["carry_settle"] > 2.2:
        flags.append(f"long hold on plate ({ph['carry_settle']:.1f} s)")
    if abs(t["start_bowl_y_cm"]) > 1.5:
        flags.append(f"start {t['start_bowl_y_cm']:+.1f} cm off the line")
    if abs(t["final_bowl_x_cm"]) > 2.5 or not (-0.5 <= t["final_base_z_cm"] <= 1.5):
        flags.append("placement off centre")
    if not ph:
        flags.append("events missing")
    return flags


def report() -> None:
    """data/takes/summary.json (per-take flags, read by the retargeting) and outputs/capture/report.md."""
    vids = {v: json.loads((TAKES / v / "results.json").read_text()) for v in VIDEOS}
    ref = json.loads((WORK / "reference.json").read_text())
    cs = C.camera_summary(C.Camera(**ref["camera"]))
    cal = ref["calibration"]
    marks = vids[REF_VIDEO]["scene"]["checks"]
    L = [
        "# Recording session: tracking report",
        "",
        "## Calibration (fitted to G1's empty frame)",
        "",
        "| Check | Result | Expected |",
        "|---|---|---|",
        f"| Lens position (cm) | {-cs['centre'][0]:.1f} left, {-cs['centre'][1]:.1f} in front, "
        f"{cs['centre'][2]:.1f} up "
        "| ~16-18 / 50-52 / 33 (hand-measured) |",
        f"| Card ends (free in the fit) | x = {cal['card_left_end_x_cm']:.1f} / {cal['card_right_end_x_cm']:.1f} "
        "| -44.4 / 15.0 (layout) |",
        f"| Backdrop depth (free) | {cal['backdrop_y_cm']:.1f} cm behind the plate centre | ~18 |",
        f"| Bowl rim radius | {ref['bowl_rim_radius_cm']:.2f} cm | 7.8 (rim 15.6 cm across) |",
        f"| Lift-guide marks back-projected | {', '.join(f'{z:.1f}' for z in marks['marks_z_cm'])} cm "
        f"| {', '.join(f'{z:g}' for z in marks['marks_z_expected_cm'])} |",
        "",
        "## Takes",
        "",
        "| Video | Tick | Takes | Start gap (cm) | Peak base lift (cm) | Clean takes |",
        "|---|---|---|---|---|---|",
    ]
    summary = {"camera": cs, "videos": {}}
    phases = {k: [] for k in LIBERO_PHASES}
    flagged = []
    for v, tick in VIDEOS.items():
        tk = vids[v]["takes"]
        fl = [take_flags(v, k, t) for k, t in enumerate(tk, 1)]
        flagged += [(v, k, f) for k, f in enumerate(fl, 1) if f]
        gaps = [t["start_gap_to_bowl_rim_cm"] for t in tk]
        lifts = [t["peak_base_lift_cm"] for t in tk]
        for t in tk:
            for p in phases:
                if t["phases_s"]:
                    phases[p].append(t["phases_s"][p])
        L.append(
            f"| {v} | {tick} | {len(tk)} | {np.median(gaps):.1f} ({min(gaps):.1f} to {max(gaps):.1f}) | "
            f"{np.median(lifts):.1f} ({min(lifts):.1f} to {max(lifts):.1f}) | {sum(not f for f in fl)} |"
        )
        summary["videos"][v] = {"n_takes": len(tk), "flags": fl, "start_gap_cm": gaps, "peak_lift_cm": lifts}
    L += [
        "",
        "## Phase durations (s, median and range over all takes)",
        "",
        "| Phase | Session | LIBERO 5-shot |",
        "|---|---|---|",
    ]
    for p, lib in LIBERO_PHASES.items():
        a = np.array(phases[p])
        L.append(f"| {p} | {np.median(a):.2f} ({a.min():.1f}-{a.max():.1f}) | {lib} |")
    L += ["", "## Quality flags (informational; the sim replay decides what is used)", ""]
    L += [f"- {v} take {k}: {'; '.join(f)}" for v, k, f in flagged]
    (WORK / "report.md").write_text("\n".join(L) + "\n")
    (TAKES / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n".join(L))


def main(videos: tuple[str, ...] = tuple(VIDEOS), overlay: bool = False) -> None:
    """Process the given videos, then write the session summary.

    Args:
        videos: recordings in data/raw (without .mp4).
        overlay: also render an annotated overlay video per recording (slow).
    """
    WORK.mkdir(parents=True, exist_ok=True)
    for v in videos:
        run_video(v, overlay)
    report()


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
