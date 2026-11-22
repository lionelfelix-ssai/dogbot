"""Decode raw YOLO26 Hailo outputs into Detections.

Probe of /home/admin/models/yolo26s.hef (HailoRT 4.20, outputs read as FLOAT32),
tools/capture_fixture.py, 2026-11-20:

  stride  grid   box layer (4 ch)     class layer (80 ch)
     8    80x80  yolo26s/conv61       yolo26s/conv64
    16    40x40  yolo26s/conv77       yolo26s/conv80
    32    20x20  yolo26s/conv91       yolo26s/conv94

CLASS VALUES ARE LOGITS -> apply sigmoid. Observed class ranges were about
[-75, +2.9] (e.g. person max 2.85 -> 0.945, dog max 1.73 -> 0.849), i.e.
outside [0, 1], so they are not probabilities.

Box channels are (l, t, r, b) distances in grid units from the anchor
(gx + 0.5, gy + 0.5); no DFL (reg_max = 1). Observed range about [-0.75, 13.8]
(small negatives only on background cells). Verified on the fixture photo:
decoded boxes land on the person and the dog.

The model is NMS-free; a per-class NMS at IoU 0.7 is kept only as a guard.

Errors (ruling R4): outputs with no box/class head pair raise ValueError
(the pipeline treats that as a vision fault). Boxes with zero or negative
width/height after clipping to [0, 1] are dropped.
"""
from __future__ import annotations

import cv2
import numpy as np

from dogbot.detection import Detection

NMS_IOU = 0.7
_DEFAULT_KEEP = frozenset({0, 16})


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -50.0, 50.0)))


def _heads(outputs: dict[str, np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pair box (4 ch) and class (80 ch) layers by grid size."""
    boxes: dict[int, np.ndarray] = {}
    classes: dict[int, np.ndarray] = {}
    for name, arr in outputs.items():
        a = np.asarray(arr)
        while a.ndim > 3 and a.shape[0] == 1:
            a = a[0]
        if a.ndim != 3 or a.shape[0] != a.shape[1]:
            continue
        if a.shape[2] == 4:
            boxes[a.shape[0]] = a
        elif a.shape[2] == 80:
            classes[a.shape[0]] = a
    return [(boxes[g], classes[g]) for g in sorted(boxes) if g in classes]


def decode_yolo26(outputs: dict[str, np.ndarray], conf_thresh: float,
                  keep: set[int] = _DEFAULT_KEEP, input_size: int = 640) -> list[Detection]:
    keep_ids = np.array(sorted(keep), dtype=np.int64)
    heads = _heads(outputs)
    if not heads:
        raise ValueError(
            "no YOLO26 box/class head pairs in outputs: "
            + ", ".join(f"{k}{np.shape(v)}" for k, v in outputs.items()))
    all_boxes, all_scores, all_cls = [], [], []
    for box, cls in heads:
        g = box.shape[0]
        stride = input_size / g
        logits = cls.reshape(-1, cls.shape[2])
        best = logits.argmax(axis=1)
        score = _sigmoid(logits[np.arange(len(best)), best])
        mask = np.isin(best, keep_ids) & (score >= conf_thresh)
        if not mask.any():
            continue
        idx = np.nonzero(mask)[0]
        gy, gx = np.divmod(idx, g)
        ax, ay = gx + 0.5, gy + 0.5
        ltrb = box.reshape(-1, 4)[idx].astype(np.float64)
        xyxy = np.stack([ax - ltrb[:, 0], ay - ltrb[:, 1],
                         ax + ltrb[:, 2], ay + ltrb[:, 3]], axis=1) * stride / input_size
        xyxy = np.clip(xyxy, 0.0, 1.0)
        valid = (xyxy[:, 2] > xyxy[:, 0]) & (xyxy[:, 3] > xyxy[:, 1])
        if not valid.any():
            continue
        all_boxes.append(xyxy[valid])
        all_scores.append(score[idx][valid])
        all_cls.append(best[idx][valid])
    if not all_boxes:
        return []
    boxes = np.concatenate(all_boxes)
    scores = np.concatenate(all_scores)
    cls_ids = np.concatenate(all_cls)

    dets: list[Detection] = []
    for c in np.unique(cls_ids):
        sel = np.nonzero(cls_ids == c)[0]
        px = boxes[sel] * input_size
        rects = [[float(x1), float(y1), float(x2 - x1), float(y2 - y1)] for x1, y1, x2, y2 in px]
        kept = cv2.dnn.NMSBoxes(rects, scores[sel].astype(float).tolist(), conf_thresh, NMS_IOU)
        for k in np.asarray(kept, dtype=np.int64).reshape(-1):
            i = sel[k]
            dets.append(Detection(int(c), float(scores[i]), tuple(float(v) for v in boxes[i])))
    dets.sort(key=lambda d: d.conf, reverse=True)
    return dets
