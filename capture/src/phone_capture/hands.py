"""Per-frame hand landmarks with MediaPipe Hands (Tasks API, video mode)."""

from __future__ import annotations

import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .video import iter_frames

WRIST, THUMB_TIP, INDEX_TIP = 0, 4, 8
MODEL = "data/models/hand_landmarker.task"


def track_hands(video: str, out: Path, model: str = MODEL) -> list[dict]:
    """One record per frame: time, and (if a hand is found) all 21 landmarks in pixels + handedness score.

    Low detection thresholds: the scene has one hand at most, and false positives are rare on black card,
    while dropouts during the pinch are what we need to avoid.
    """
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    recs = []
    with vision.HandLandmarker.create_from_options(opts) as lm:
        last_ms = -1
        for i, t, img in iter_frames(video):
            ms = max(int(round(t * 1000)), last_ms + 1)  # strictly increasing timestamps
            last_ms = ms
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(img)), ms)
            rec = {"frame": i, "t": t, "hand": None}
            if res.hand_landmarks:
                h, w = img.shape[:2]
                pts = [[p.x * w, p.y * h] for p in res.hand_landmarks[0]]
                rec["hand"] = {
                    "landmarks_px": pts,
                    "handedness": res.handedness[0][0].category_name,
                    "score": float(res.handedness[0][0].score),
                }
            recs.append(rec)
    out.write_text(json.dumps(recs))
    return recs
