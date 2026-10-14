"""Plots, overlay video and per-take files for the capture pipeline."""

from __future__ import annotations

import csv
from pathlib import Path

import av
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import calib as C  # noqa: E402
from .objects import white_mask  # noqa: E402
from .video import iter_frames  # noqa: E402

# Reference data-viz palette (validated categorical slots 1-2), surfaces and ink.
HAND, BOWL = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#d9d8d2"


def write_csv(rows: dict, path: Path) -> None:
    keys = list(rows)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(len(rows[keys[0]])):
            w.writerow(
                [
                    (
                        ""
                        if isinstance(rows[k][i], float) and np.isnan(rows[k][i])
                        else int(rows[k][i])
                        if isinstance(rows[k][i], (bool, np.bool_))
                        else f"{rows[k][i]:.3f}"
                        if isinstance(rows[k][i], float)
                        else rows[k][i]
                    )
                    for k in keys
                ]
            )


def plot_take(res: dict, k: int, path: Path) -> None:
    r = res["resampled"]
    t = np.array(r["time_s"])
    closed = np.array(r["gripper_closed"], bool)
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, label in (
        (axes[0], "x", "x along line (cm, plate centre = 0)"),
        (axes[1], "z", "height above card (cm)"),
    ):
        ax.set_facecolor(SURFACE)
        # gripper-closed band(s)
        edges = np.flatnonzero(np.diff(np.r_[0, closed.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.axvspan(t[a], t[b - 1], color=BAND, lw=0, zorder=0)
        ax.plot(t, r[f"hand_{key}_cm"], color=HAND, lw=2, label="hand (thumb-index midpoint)", zorder=3)
        bowl_key = "bowl_x_cm" if key == "x" else "bowl_base_z_cm"
        ax.plot(t, r[bowl_key], color=BOWL, lw=2, label="bowl centre" if key == "x" else "bowl base", zorder=3)
        ax.set_ylabel(label, color=INK_2, fontsize=8)
        ax.grid(color=GRID, lw=0.6, zorder=1)
        ax.tick_params(colors=INK_2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    ev = res["events_s"]
    for name in ("lift", "placed", "release"):
        if ev.get(name) is not None:
            for ax in axes:
                ax.axvline(ev[name], color=INK_2, lw=0.8, ls=(0, (3, 3)), zorder=2)
            axes[0].text(
                ev[name],
                1.01,
                name,
                transform=axes[0].get_xaxis_transform(),
                fontsize=7,
                color=INK_2,
                ha="center",
                va="bottom",
            )
    axes[1].set_xlabel("time since hand enters (s), resampled to 20 Hz", color=INK_2, fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND))
    labels.append("gripper closed")
    gap = res["start_gap_to_bowl_rim_cm"]
    fig.suptitle(
        f"Take {k}: start gap {gap:.1f} cm, peak base lift {res['peak_base_lift_cm']:.1f} cm",
        x=0.02,
        ha="left",
        fontsize=10,
        color=INK,
        y=0.99,
    )
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.015, 0.955),
        frameon=False,
        fontsize=8,
        ncol=3,
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]


def _bowl_wire(cam: C.Camera, c: np.ndarray, radius: float) -> list[np.ndarray]:
    th = np.linspace(0, 2 * np.pi, 72)
    rim = np.stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th), np.full_like(th, c[2])], 1)
    base_r = 0.5 * radius
    base = np.stack([c[0] + base_r * np.cos(th), c[1] + base_r * np.sin(th), np.full_like(th, c[2] - C.BOWL_H)], 1)
    return [cam.project(rim), cam.project(base)]


def overlay_video(
    video: str,
    out: Path,
    cam: C.Camera,
    hands: list[dict],
    bowl_all: np.ndarray,
    radius: float,
    labels: list[dict],
    scale: float = 0.5,
    trail_s: float = 1.5,
) -> None:
    """Landmarks, bowl silhouette mask, fitted bowl model and trails, with take / phase / gripper text.

    `labels[i]` per frame: {"text": str, "gripper": bool | None, "pinch_px": (u, v) | None}.
    """
    W, H = int(1920 * scale), int(1080 * scale)
    container = av.open(str(out), "w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": "23", "preset": "medium"}
    trail_b, trail_h = [], []
    for i, t, img in iter_frames(video):
        vis = img.copy()
        c = bowl_all[i]
        rim_px, base_px = _bowl_wire(cam, c, radius)
        # bowl mask: white pixels inside the projected bowl model's bounding region
        x0, y0 = np.floor(np.minimum(rim_px.min(0), base_px.min(0))).astype(int) - 10
        x1, y1 = np.ceil(np.maximum(rim_px.max(0), base_px.max(0))).astype(int) + 10
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1920, x1), min(1080, y1)
        if x1 > x0 and y1 > y0:
            hull = cv2.convexHull(np.concatenate([rim_px, base_px]).astype(np.int32))
            region = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(region, hull, 1)
            m = white_mask(img, v_min=140, s_max=70).astype(bool) & region.astype(bool)
            vis[m] = (0.55 * vis[m] + 0.45 * np.array([235, 104, 52])).astype(np.uint8)
        cv2.polylines(vis, [rim_px.astype(np.int32).reshape(-1, 1, 2)], True, (235, 104, 52), 3)
        cv2.polylines(vis, [base_px.astype(np.int32).reshape(-1, 1, 2)], True, (235, 104, 52), 2)
        cpx = cam.project(np.array([[c[0], c[1], c[2] - C.BOWL_H / 2]]))[0]
        trail_b.append((t, cpx))
        h = hands[i]["hand"]
        lab = labels[i]
        if h is not None:
            lm = np.array(h["landmarks_px"]).astype(int)
            for a, b in BONES:
                cv2.line(vis, tuple(lm[a]), tuple(lm[b]), (255, 255, 255), 2)
            for j, col in ((4, (42, 120, 214)), (8, (27, 175, 122)), (0, (200, 200, 200))):
                cv2.circle(vis, tuple(lm[j]), 9, col, -1)
            pinch = 0.5 * (lm[4] + lm[8])
            trail_h.append((t, pinch))
            if lab.get("gripper"):
                cv2.circle(vis, tuple(pinch.astype(int)), 22, (255, 255, 255), 3)
        trail_b = [p for p in trail_b if t - p[0] < trail_s]
        trail_h = [p for p in trail_h if t - p[0] < trail_s]
        for tr, col in ((trail_b, (235, 104, 52)), (trail_h, (42, 120, 214))):
            if len(tr) > 1:
                cv2.polylines(vis, [np.array([p for _, p in tr]).astype(np.int32).reshape(-1, 1, 2)], False, col, 4)
        small = cv2.resize(vis, (W, H), interpolation=cv2.INTER_AREA)
        cv2.rectangle(small, (0, 0), (W, 34), (20, 20, 20), -1)
        cv2.putText(
            small,
            f"t={t:5.2f}s  {lab['text']}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        frame = av.VideoFrame.from_ndarray(small, format="rgb24")
        for p in stream.encode(frame):
            container.mux(p)
    for p in stream.encode():
        container.mux(p)
    container.close()
