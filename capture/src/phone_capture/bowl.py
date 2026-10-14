"""Bowl tracking by fitting its rim, a horizontal circle of known radius, to silhouette edges.

Why not plain brightness segmentation (or SAM 2): when the bowl is on the plate, bowl and plate are both
white and merge. But the camera looks down, so the bowl's far rim always projects ABOVE the plate's far
edge, against the black card, even with the bowl resting on the plate. So the tracker:
  1. takes the white mask (skin removed),
  2. keeps only outline points whose outside is dark card (real silhouette edges, not hand/plate
     boundaries) and that are not on the plate's static outline,
  3. keeps those above the rim's predicted centre row (the far rim; the lower outline is the bowl body),
  4. fits a circle of the calibrated radius in a horizontal plane at unknown height: centre (x, y) and
     rim height z, through the calibrated camera (robust least squares).
This needs no learned model and is the same procedure in every phase, so placement is not a special case.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy.optimize import least_squares

from .calib import Camera
from .objects import white_mask


def skin_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] < 25) | (hsv[..., 0] > 165)) & (hsv[..., 1] > 45) & (hsv[..., 2] > 50)
    return cv2.dilate(m.astype(np.uint8), np.ones((9, 9), np.uint8))


def plate_outline_mask(plate_mask: np.ndarray, width: int = 9) -> np.ndarray:
    edge = cv2.morphologyEx(plate_mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    return cv2.dilate(edge, np.ones((width, width), np.uint8))


def edge_points(img: np.ndarray, plate_outline: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """Silhouette points of white objects whose outside (4-6 px away) is dark card, inside a ROI."""
    x0, y0, x1, y1 = roi
    sub = img[y0:y1, x0:x1]
    w = white_mask(sub, v_min=140, s_max=70) & (1 - skin_mask(sub))
    gray = cv2.GaussianBlur(cv2.cvtColor(sub, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    cnts = cv2.findContours(w, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]
    pts = []
    blur = cv2.GaussianBlur(w.astype(float), (11, 11), 0)
    gy, gx = np.gradient(blur)
    for c in cnts:
        if len(c) < 40:
            continue
        for x, y in c.reshape(-1, 2):
            n = -np.array([gx[y, x], gy[y, x]])  # outward: down the mask gradient
            nn = np.linalg.norm(n)
            if nn < 1e-6:
                continue
            n /= nn
            ox, oy = int(round(x + 6 * n[0])), int(round(y + 6 * n[1]))
            if 0 <= ox < sub.shape[1] and 0 <= oy < sub.shape[0] and gray[oy, ox] < 70:
                if not plate_outline[y + y0, x + x0]:
                    pts.append((x + x0, y + y0))
    return np.array(pts, float).reshape(-1, 2)


def fit_rim(cam: Camera, pts: np.ndarray, radius: float, init: np.ndarray, y: float = 0.0) -> tuple[np.ndarray, dict]:
    """Rim circle of fixed radius, centred ON the object line (y = 0): fit centre x and rim height z.

    With a partial arc, depth and height trade off along the viewing ray, and a soft depth prior gets
    outvoted by hundreds of edge points under a robust loss. The motion is recorded along the object line
    (the bowl's measured depth at rest is within 0.4 cm of it), so y is fixed: 0 by default, or a given
    depth along the carry path (see pipeline); the fit residual shows when the assumption is violated.
    """
    C = np.asarray(cam.centre)

    def res(p, d):
        xc, zr = p
        lam = (zr - C[2]) / d[:, 2]
        w = C + lam[:, None] * d
        return np.hypot(w[:, 0] - xc, w[:, 1] - y) - radius

    d = cam.ray(pts)
    sol = least_squares(res, [init[0], init[2]], args=(d,), loss="soft_l1", f_scale=0.5)
    r = res(sol.x, d)
    keep = np.abs(r) < max(1.0, 3 * np.median(np.abs(r)))  # trim hand / reflection outliers, refit once
    if 15 <= keep.sum() < len(pts):
        d = d[keep]
        sol = least_squares(res, sol.x, args=(d,), loss="soft_l1", f_scale=0.5)
        r = res(sol.x, d)
    return np.array([sol.x[0], y, sol.x[1]]), {"rms_cm": float(np.sqrt(np.mean(r**2))), "n_points": int(len(r))}


def upper_arc(cam: Camera, pts: np.ndarray, centre: np.ndarray, radius: float, band_px: float = 60) -> np.ndarray:
    """Points near the predicted far half of the rim ellipse (above its centre row, near its outline)."""
    th = np.linspace(0, np.pi, 60)  # far half: y_world >= centre
    ring = np.stack([centre[0] + radius * np.cos(th), centre[1] + radius * np.sin(th), np.full_like(th, centre[2])], 1)
    ring_px = cam.project(ring)
    ccen = cam.project(centre[None])[0]
    if len(pts) == 0:
        return pts
    dist = np.min(np.linalg.norm(pts[:, None, :] - ring_px[None], axis=2), axis=1)
    return pts[(dist < band_px) & (pts[:, 1] < ccen[1] + 5)]


def track_frame(
    cam: Camera, img: np.ndarray, plate_outline: np.ndarray, radius: float, prev: np.ndarray, band_px: float = 60
) -> tuple[np.ndarray | None, dict]:
    """One frame: predicted rim centre `prev` (x, y, z cm) -> refined centre, or None if too few edges."""
    ring = np.stack(
        [
            prev[0] + radius * np.cos(np.linspace(0, 2 * np.pi, 24)),
            prev[1] + radius * np.sin(np.linspace(0, 2 * np.pi, 24)),
            np.full(24, prev[2]),
        ],
        1,
    )
    px = cam.project(ring)
    pad = 120
    roi = (
        max(0, int(px[:, 0].min()) - pad),
        max(0, int(px[:, 1].min()) - pad),
        min(img.shape[1], int(px[:, 0].max()) + pad),
        min(img.shape[0], int(px[:, 1].max()) + pad),
    )
    pts = edge_points(img, plate_outline, roi)
    arc = upper_arc(cam, pts, prev, radius, band_px)
    if len(arc) < 25:
        return None, {"n_points": int(len(arc))}
    x, info = fit_rim(cam, arc, radius, prev)
    # arc coverage: angular span of the supporting points around the fitted centre (degrees)
    C = np.asarray(cam.centre)
    d = cam.ray(arc)
    w = C + ((x[2] - C[2]) / d[:, 2])[:, None] * d
    ang = np.degrees(np.arctan2(w[:, 1] - x[1], w[:, 0] - x[0]))
    info["arc_span_deg"] = float(ang.max() - ang.min())
    info["arc_px"] = arc
    return x, info


def calibrate_radius(
    cam: Camera, img: np.ndarray, plate_outline: np.ndarray, init_xy: np.ndarray, rim_z: float
) -> float:
    """Effective rim radius from a frame with the bowl resting on the card (rim height known), using exactly
    the tracker's own edge points, so resting heights come out right by construction."""
    x, _ = track_frame(cam, img, plate_outline, 8.0, np.array([*init_xy, rim_z]))
    pts = edge_points(img, plate_outline, (0, 0, img.shape[1], img.shape[0]))
    arc = upper_arc(cam, pts, x, 8.0)
    C = np.asarray(cam.centre)
    d = cam.ray(arc)
    w = C + ((rim_z - C[2]) / d[:, 2])[:, None] * d
    from .objects import fit_circle

    _, r, _ = fit_circle(w[:, :2])
    return r


def coarse_detect(cam: Camera, img: np.ndarray, plate_mask: np.ndarray, radius: float) -> np.ndarray | None:
    """Re-acquisition: the largest white blob that is mostly off the plate; the midpoint of its leftmost and
    rightmost outline points approximates the rim centre, back-projected onto the object plane."""
    from .objects import components

    w = white_mask(img, v_min=140, s_max=70) & (1 - skin_mask(img))
    w = w & (1 - cv2.dilate(plate_mask, np.ones((7, 7), np.uint8)))
    comps = [c for c in components(w, min_area=4000, roi_y=(0, img.shape[0]))]
    if not comps:
        return None
    c = comps[0]["contour"]
    mid = 0.5 * (c[c[:, 0].argmin()] + c[c[:, 0].argmax()])
    p = cam.to_plane(mid[None], y=0.0)[0]
    return np.array([p[0], 0.0, p[2]])


def track_video(
    cam: Camera, frames, plate_mask: np.ndarray, radius: float, start: np.ndarray, max_jump_cm: float = 4.0
) -> list[dict]:
    """Sequential tracking with a constant-velocity prediction.

    Per frame, `source` says where the value came from:
      rim     rim-circle fit (accurate; the normal case)
      coarse  rim fit failed or jumped implausibly; blob-midpoint re-detection on the object plane
              (biased ~1 cm low in height, used to keep the track and re-seed the next fit)
      hold    neither worked; previous value repeated
    """
    plate_outline = plate_outline_mask(plate_mask)
    out, prev, vel = [], np.asarray(start, float), np.zeros(3)
    for i, t, img in frames:
        pred = prev + vel
        x, info = track_frame(cam, img, plate_outline, radius, pred)
        rec = {"frame": i, "t": t, "n_points": info.get("n_points", 0)}
        if x is not None and np.linalg.norm(x - pred) < max_jump_cm and info["rms_cm"] < 1.0:
            rec.update(
                source="rim",
                centre=x.tolist(),
                rms_cm=info["rms_cm"],
                arc_span_deg=info["arc_span_deg"],
                arc_px=info["arc_px"][:: max(1, len(info["arc_px"]) // 40)].tolist(),
            )
        else:
            c = coarse_detect(cam, img, plate_mask, radius)
            if c is not None and np.linalg.norm(c - pred) < 3 * max_jump_cm:
                x2, info2 = track_frame(cam, img, plate_outline, radius, c)  # re-seed the rim fit
                if x2 is not None and np.linalg.norm(x2 - c) < max_jump_cm and info2["rms_cm"] < 1.0:
                    x, src = x2, "rim"
                    rec.update(
                        rms_cm=info2["rms_cm"],
                        arc_span_deg=info2["arc_span_deg"],
                        arc_px=info2["arc_px"][:: max(1, len(info2["arc_px"]) // 40)].tolist(),
                    )
                else:
                    x, src = c, "coarse"
            else:
                x, src = pred, "hold"
            rec.update(source=src, centre=np.asarray(x).tolist())
        x = np.asarray(x, float)
        vel = 0.5 * vel + 0.5 * (x - prev) if rec["source"] != "hold" else 0.5 * vel
        prev = x
        out.append(rec)
    return out


def fit_rim_free(cam: Camera, pts: np.ndarray, radius: float, init: np.ndarray) -> tuple[np.ndarray, float]:
    """Rim circle with depth (y) free as well. Only well-determined when most of the rim is visible, i.e.
    with the bowl at rest; used to measure where a take starts. Returns centre (x, y, z) and rms (cm)."""
    C = np.asarray(cam.centre)
    d = cam.ray(pts)

    def res(p):
        lam = (p[2] - C[2]) / d[:, 2]
        w = C + lam[:, None] * d
        return np.hypot(w[:, 0] - p[0], w[:, 1] - p[1]) - radius

    sol = least_squares(res, init, loss="soft_l1", f_scale=0.3)
    return sol.x, float(np.sqrt(np.mean(res(sol.x) ** 2)))


def rest_pose(cam: Camera, recs: list[dict], frames: range, radius: float) -> dict | None:
    """Median free-depth rim fit over rim-tracked frames in `frames` with a long visible arc."""
    xs = []
    for i in frames:
        r = recs[i]
        if r.get("source") == "rim" and r.get("arc_span_deg", 0) > 150 and len(r.get("arc_px", [])) >= 20:
            x, rms = fit_rim_free(cam, np.array(r["arc_px"]), radius, np.array(r["centre"]))
            if rms < 0.5:
                xs.append(x)
    if len(xs) < 3:
        return None
    xs = np.array(xs)
    return {"centre": np.median(xs, 0).tolist(), "n": len(xs), "spread_cm": np.std(xs, 0).tolist()}
