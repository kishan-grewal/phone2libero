"""Camera calibration for the side-on bowl -> plate rig, from one empty frame of the recording.

World frame (cm): origin at the plate centre on the floor card, x along the object line toward the plate's
far (right) side, y away from the camera (toward the backdrop), z up. The object plane is y = 0.

A pinhole camera (principal point at the image centre, no distortion, focal length fixed from an earlier
tape-measure calibration of the same lens) is fitted to the rig's known geometry: the plate outline, the floor
card's two ends, the backdrop's vertical edges and the lift-guide marks at known heights (fit_camera_plate).
The bowl is not used in the fit; its fitted size is the independent cross-check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

# Rig layout (cm), from data/raw/measurements.txt.
CARD_LONG, CARD_DEEP = 59.4, 42.0
PLATE_D, PLATE_H = 28.0, 3.0
BOWL_D, BOWL_H = 16.5, 5.5
X_RIGHT_END = PLATE_D / 2 + 1.0  # plate's right edge is 1 cm from the card's right end
X_LEFT_END = X_RIGHT_END - CARD_LONG
Y_BACKDROP = 18.0  # object line is 18 cm in front of the backdrop
IMG_W, IMG_H = 1920, 1080


@dataclass
class Camera:
    f: float
    rvec: list[float]  # world -> camera rotation (Rodrigues)
    centre: list[float]  # camera centre in world (cm)
    cx: float = IMG_W / 2
    cy: float = IMG_H / 2

    @property
    def R(self) -> np.ndarray:
        return cv2.Rodrigues(np.asarray(self.rvec, float))[0]

    def project(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        Xc = (X - np.asarray(self.centre)) @ self.R.T
        return np.stack([self.f * Xc[:, 0] / Xc[:, 2] + self.cx, self.f * Xc[:, 1] / Xc[:, 2] + self.cy], 1)

    def ray(self, uv: np.ndarray) -> np.ndarray:
        uv = np.atleast_2d(uv)
        d = np.stack([(uv[:, 0] - self.cx) / self.f, (uv[:, 1] - self.cy) / self.f, np.ones(len(uv))], 1)
        return d @ self.R  # camera -> world direction

    def to_plane(self, uv: np.ndarray, y: float = 0.0) -> np.ndarray:
        """Back-project pixels onto the vertical plane at depth y (default: the object plane)."""
        d = self.ray(uv)
        C = np.asarray(self.centre)
        lam = (y - C[1]) / d[:, 1]
        return C + lam[:, None] * d

    def to_floor(self, uv: np.ndarray, z: float = 0.0) -> np.ndarray:
        d = self.ray(uv)
        C = np.asarray(self.centre)
        lam = (z - C[2]) / d[:, 2]
        return C + lam[:, None] * d

    def pitch_deg(self) -> float:
        """Downward tilt of the optical axis below horizontal."""
        axis = self.R[2]  # optical axis in world
        return float(np.degrees(np.arcsin(-axis[2])))


# --- feature extraction -------------------------------------------------------------------------


def dark_mask(img: np.ndarray, thr: int = 40) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), (9, 9), 0)
    return (g < thr).astype(np.uint8)


def ransac_line(pts: np.ndarray, tol: float = 2.0, iters: int = 500, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Fit x = a*y + b robustly. Returns (a, b) and the inlier mask."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        if pts[i, 1] == pts[j, 1]:
            continue
        a = (pts[j, 0] - pts[i, 0]) / (pts[j, 1] - pts[i, 1])
        b = pts[i, 0] - a * pts[i, 1]
        inl = np.abs(pts[:, 0] - (a * pts[:, 1] + b)) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    a, b = np.polyfit(pts[best, 1], pts[best, 0], 1)
    inl = np.abs(pts[:, 0] - (a * pts[:, 1] + b)) < tol
    return np.array([a, b]), inl


def card_end_points(img: np.ndarray) -> dict[str, np.ndarray]:
    """Edge pixels of the floor card's left and right ends (dark card vs lighter wooden floor)."""
    dark = cv2.morphologyEx(dark_mask(img), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    left, right = [], []
    for y in range(640, IMG_H - 10, 4):
        row = dark[y]
        # left end: first long dark run from the left, in the left part of the image
        xs = np.where(row[:900])[0]
        if len(xs) and row[xs[0] : xs[0] + 60].all() and xs[0] > 5:
            left.append((xs[0], y))
        # right end: last dark pixel of the run that reaches past the plate, right part of the image
        xs = np.where(row[1450:])[0]
        if len(xs) and xs[-1] + 1450 < IMG_W - 5 and row[xs[-1] + 1450 - 60 : xs[-1] + 1450].all():
            right.append((xs[-1] + 1450, y))
    out = {}
    for k, p in (("left", left), ("right", right)):
        p = np.array(p, float)
        _, inl = ransac_line(p)
        out[k] = p[inl]
    return out


def backdrop_edge_points(img: np.ndarray, y_range=(20, 520)) -> dict[str, np.ndarray]:
    """Edge pixels of the upright backdrop card's left and right sides (dark card vs light wall).

    Scans outward from the image centre column (on the dark backdrop) along each row until the card ends,
    so it works for any framing; rows crossing a blue mark are skipped (the marks sit on the edges)."""
    dark = cv2.morphologyEx(dark_mask(img, thr=60), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    blue = (hsv[..., 0] > 100) & (hsv[..., 0] < 125) & (hsv[..., 1] > 120)
    xc = img.shape[1] // 2
    left, right = [], []
    for y in range(*y_range, 4):
        row = dark[y]
        if not row[xc]:
            continue
        lo = np.where(row[:xc] == 0)[0]
        hi = np.where(row[xc:] == 0)[0]
        if len(lo):
            x = lo[-1] + 1
            if not blue[y, max(0, x - 40) : x + 40].any():
                left.append((x, y))
        if len(hi):
            x = xc + hi[0] - 1
            if not blue[y, max(0, x - 40) : x + 40].any():
                right.append((x, y))
    out = {}
    for k, p in (("left", left), ("right", right)):
        p = np.array(p, float)
        _, inl = ransac_line(p)
        out[k] = p[inl]
    return out


def backdrop_marks(img: np.ndarray, max_y: int = 420) -> np.ndarray:
    """Centres of the blue lift-guide marks on the backdrop (the upper blue blobs), sorted by x then y."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m = ((hsv[..., 0] > 100) & (hsv[..., 0] < 125) & (hsv[..., 1] > 120) & (hsv[..., 2] > 60)).astype(np.uint8)
    n, _, st, cen = cv2.connectedComponentsWithStats(m)
    blobs = [cen[i] for i in range(1, n) if st[i, 4] > 800 and cen[i][1] < max_y]
    if len(blobs) < 2:
        raise ValueError(f"expected at least 2 upper blue marks, found {len(blobs)}")
    return np.array(sorted(blobs, key=lambda c: (c[0] > np.median([b[0] for b in blobs]), c[1])))


# Heights of the lift-guide marks (cm). The original pair, as measured in the tape-calibrated test session
# (left 21.5, right 20.0), and the pair added at 16 cm before G1. Per side, the higher blob is the original.
ORIGINAL_MARK_Z = {"left": 21.5, "right": 20.0}
ADDED_MARK_Z = 16.0


def mark_heights(marks: np.ndarray) -> list[float]:
    xmid = np.median(marks[:, 0])
    out = []
    for side in ("left", "right"):
        sel = [m for m in marks if (m[0] > xmid) == (side == "right")]
        for j, _ in enumerate(sorted(sel, key=lambda m: m[1])):
            out.append(ORIGINAL_MARK_Z[side] if j == 0 else ADDED_MARK_Z)
    return out


def background_homography(
    img_a: np.ndarray, img_b: np.ndarray, exclude: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, dict]:
    """Homography mapping pixels of view A to view B, from SIFT matches on static background only."""
    mask = np.full(img_a.shape[:2], 255, np.uint8)
    for x0, y0, x1, y1 in exclude:
        mask[y0:y1, x0:x1] = 0
    sift = cv2.SIFT_create(4000)
    ga, gb = (cv2.cvtColor(i, cv2.COLOR_RGB2GRAY) for i in (img_a, img_b))
    ka, da = sift.detectAndCompute(ga, mask)
    kb, db = sift.detectAndCompute(gb, mask)
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 2.0)
    inl = inl.ravel().astype(bool)
    err = np.linalg.norm(cv2.perspectiveTransform(pa[inl].reshape(-1, 1, 2), H).reshape(-1, 2) - pb[inl], axis=1)
    shift = pb[inl] - pa[inl]
    return H, {
        "n_matches": len(good),
        "n_inliers": int(inl.sum()),
        "reproj_err_px_median": float(np.median(err)),
        "reproj_err_px_p95": float(np.percentile(err, 95)),
        "shift_px_median": np.median(shift, 0).tolist(),
        "shift_px_max": float(np.linalg.norm(shift, axis=1).max()),
    }


# --- camera fit ---------------------------------------------------------------------------------


def _line_residuals(cam: Camera, pts: np.ndarray, p0: np.ndarray, p1: np.ndarray) -> np.ndarray:
    a, b = cam.project(np.stack([p0, p1]))
    n = np.array([-(b - a)[1], (b - a)[0]])
    n /= np.linalg.norm(n)
    return (pts - a) @ n


def camera_summary(cam: Camera) -> dict:
    return {
        **asdict(cam),
        "pitch_down_deg": cam.pitch_deg(),
        "hfov_deg": float(np.degrees(2 * np.arctan(IMG_W / 2 / cam.f))),
    }


def look_at(centre: np.ndarray, target) -> list[float]:
    z = np.asarray(target, float) - centre
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return cv2.Rodrigues(np.stack([x, np.cross(z, x), z]))[0].ravel().tolist()


# Plate outline diameter, measured in the earlier tape-calibrated test session (outline taken at z = 2.5 cm).
PLATE_OUTLINE_D_MEASURED = 29.0
PLATE_OUTLINE_Z = 2.5


def fit_camera_plate(
    card: dict[str, np.ndarray],
    backdrop: dict[str, np.ndarray],
    marks: np.ndarray,
    plate_contour: np.ndarray,
    init: Camera,
    f: float,
    f_free: bool = False,
) -> tuple[Camera, dict]:
    """Calibrate from the rig itself (no tape in view). World origin = plate centre.

    Fits the camera pose (focal length fixed: same lens and settings as the tape-calibrated test session) to
      - the plate's outline: a circle of the measured diameter at z = PLATE_OUTLINE_Z, centred at x = y = 0,
      - the floor card's ends: lines CARD_LONG apart along x, position along x free (card may have moved),
      - the backdrop's vertical edges and lift-guide marks (known heights), at a free backdrop depth.
    """
    nm = len(marks)
    marks_z = np.asarray(mark_heights(marks), float)
    pc = plate_contour[:: max(1, len(plate_contour) // 300)]
    r_plate = PLATE_OUTLINE_D_MEASURED / 2

    def unpack(p):
        cam = Camera(f=p[-1] if f_free else f, rvec=p[0:3].tolist(), centre=p[3:6].tolist())
        return cam, p[6], p[7], p[8:10], p[10 : 10 + nm]  # card x of right end, backdrop y, edge x, mark x

    def parts(p):
        cam, xr, yb, edge_x, mark_x = unpack(p)
        xl = xr - CARD_LONG
        w = cam.to_floor(pc, z=PLATE_OUTLINE_Z)
        return {
            # plate residual in cm, scaled to ~px (about 20 px per cm at the plate)
            "plate": 20.0 * (np.hypot(w[:, 0], w[:, 1]) - r_plate),
            "card_left": _line_residuals(cam, card["left"], np.array([xl, -30, 0]), np.array([xl, yb, 0])),
            "card_right": _line_residuals(cam, card["right"], np.array([xr, -30, 0]), np.array([xr, yb, 0])),
            "backdrop_left": _line_residuals(
                cam, backdrop["left"], np.array([edge_x[0], yb, 0]), np.array([edge_x[0], yb, 40])
            ),
            "backdrop_right": _line_residuals(
                cam, backdrop["right"], np.array([edge_x[1], yb, 0]), np.array([edge_x[1], yb, 40])
            ),
            "marks": (cam.project(np.stack([mark_x, np.full(nm, yb), marks_z], 1)) - marks).ravel(),
        }

    def residuals(p):
        return np.concatenate(list(parts(p).values()))

    mark_x0 = np.where(marks[:, 0] < np.median(marks[:, 0]), -40.0, 15.0)
    p0 = np.concatenate([init.rvec, init.centre, [15.0, 20.0, -45.0, 15.0], mark_x0, [f] if f_free else []])
    sol = least_squares(residuals, p0, loss="soft_l1", f_scale=2.0, max_nfev=20000)
    cam, xr, yb, edge_x, mark_x = unpack(sol.x)
    pr = parts(sol.x)
    info = {
        **{f"rms_px_{k}": float(np.sqrt(np.mean(v**2))) for k, v in pr.items()},
        "card_right_end_x_cm": float(xr),
        "card_left_end_x_cm": float(xr - CARD_LONG),
        "backdrop_y_cm": float(yb),
        "backdrop_edge_x_cm": edge_x.tolist(),
        "mark_x_cm": mark_x.tolist(),
        "mark_z_cm": marks_z.tolist(),
    }
    return cam, info
