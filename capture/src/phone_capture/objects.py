"""White objects on the black card: segmentation and metric fits through the calibrated camera."""

from __future__ import annotations

import cv2
import numpy as np

from .calib import Camera


def white_mask(img: np.ndarray, v_min: int = 150, s_max: int = 60) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def components(mask: np.ndarray, min_area: int = 3000, roi_y: tuple[int, int] = (380, 1080)) -> list[dict]:
    """Connected white blobs on the card (below the backdrop junction), largest first."""
    m = mask.copy()
    m[: roi_y[0]] = 0
    n, lab, st, cen = cv2.connectedComponentsWithStats(m)
    out = []
    for i in range(1, n):
        if st[i, 4] >= min_area:
            blob = (lab == i).astype(np.uint8)
            cnt = max(cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
            out.append(
                {
                    "area": int(st[i, 4]),
                    "bbox": st[i, :4].tolist(),
                    "centroid": cen[i].tolist(),
                    "contour": cnt.reshape(-1, 2).astype(float),
                    "mask": blob,
                }
            )
    return sorted(out, key=lambda c: -c["area"])


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Algebraic circle fit; returns centre, radius, rms radial residual."""
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy**2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    r = np.sqrt(c + cx**2 + cy**2)
    res = np.linalg.norm(xy - [cx, cy], axis=1) - r
    return np.array([cx, cy]), float(r), float(np.sqrt(np.mean(res**2)))


def rim_circle(cam: Camera, contour: np.ndarray, rim_z: float, keep: str) -> dict:
    """Back-project the part of a contour that traces the object's rim onto the plane z = rim_z, fit a circle.

    For a camera looking down, the far (upper) half of the outline is the rim for both the plate and the
    bowl. keep="upper" uses contour points above the outline's horizontal extremes' midline, "all" uses all.
    """
    pts = contour
    if keep == "upper":
        xl, xr = pts[:, 0].argmin(), pts[:, 0].argmax()
        ymid = 0.5 * (pts[xl, 1] + pts[xr, 1])
        pts = pts[pts[:, 1] <= ymid + 2]
    w = cam.to_floor(pts, z=rim_z)
    c, r, rms = fit_circle(w[:, :2])
    return {"centre_xy_cm": c.tolist(), "diameter_cm": 2 * r, "fit_rms_cm": rms, "n_points": len(pts)}
