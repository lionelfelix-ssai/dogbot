"""Annotated frame for the web view: boxes, reticles, couch zone, status banner.

All geometry comes in normalized (0-1) frame coordinates and is scaled to
the frame being drawn on. Colors are BGR (OpenCV).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from dogbot.detection import Detection

PERSON_CLS = 0
DOG_CLS = 16

GREEN = (0, 200, 0)
AMBER = (0, 170, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
PERSON_COLOR = (255, 160, 60)
DOG_BOX_COLOR = (200, 200, 200)
BANNER_BG = {"ARMED": (0, 0, 150), "DISARMED": (90, 90, 90),
             "DISABLED": (60, 60, 60), "FAULT": (0, 0, 200)}


@dataclass(frozen=True)
class TrackedDet:
    det: Detection
    track_id: int | None
    in_zone: bool


def _px_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (int(round(x1 * w)), int(round(y1 * h)), int(round(x2 * w)), int(round(y2 * h)))


def _reticle(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color) -> None:
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    r = max(2, int(0.35 * min(x2 - x1, y2 - y1)))
    th = max(1, r // 12)
    cv2.circle(img, (cx, cy), r, color, th, cv2.LINE_AA)
    inner, outer = r // 2, int(r * 1.3)
    for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
        cv2.line(img, (cx + dx * inner, cy + dy * inner),
                 (cx + dx * outer, cy + dy * outer), color, th, cv2.LINE_AA)


def banner_label(status: dict) -> str:
    if status.get("fault"):
        return "FAULT"
    if not status.get("enabled", True):
        return "DISABLED"
    if status.get("armed"):
        return "ARMED"
    return f"DISARMED (arming in {int(round(status.get('seconds_until_armed', 0)))}s)"


def draw_overlay(frame_bgr: np.ndarray, tracks: list[TrackedDet], zone_norm,
                 status: dict) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    lw = max(1, w // 640)

    if zone_norm and len(zone_norm) >= 3:
        if status.get("state") == "blasting":
            zc = RED
        elif status.get("occupied"):
            zc = AMBER
        else:
            zc = GREEN
        pts = np.array([[int(round(x * w)), int(round(y * h))] for x, y in zone_norm],
                       dtype=np.int32)
        cv2.polylines(img, [pts], True, zc, 2 * lw, cv2.LINE_AA)

    for t in tracks:
        x1, y1, x2, y2 = _px_box(t.det.box, w, h)
        if t.det.cls_id == PERSON_CLS:
            cv2.rectangle(img, (x1, y1), (x2, y2), PERSON_COLOR, 1)
        elif t.det.cls_id == DOG_CLS:
            color = RED if t.in_zone else AMBER
            cv2.rectangle(img, (x1, y1), (x2, y2), DOG_BOX_COLOR, lw)
            if t.track_id is not None:
                _reticle(img, x1, y1, x2, y2, color)
                cv2.putText(img, f"#{t.track_id} {t.det.conf:.2f}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45 * lw, color, lw, cv2.LINE_AA)

    label = banner_label(status)
    text = f"{label}  state={status.get('state', '?')}  round={status.get('round', 0)}"
    if status.get("sequence"):
        text += f"  seq={status['sequence']}"
    bh = 24 * lw
    bg = BANNER_BG[label.split(" ")[0]]
    cv2.rectangle(img, (0, 0), (w, bh), bg, -1)
    cv2.putText(img, text, (6, bh - 7 * lw), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * lw,
                WHITE, lw, cv2.LINE_AA)
    return img
