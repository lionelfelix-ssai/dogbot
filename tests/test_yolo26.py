from pathlib import Path

import numpy as np
import pytest

from dogbot.detection import Detection
from dogbot.yolo26 import decode_yolo26

FIXTURE = Path(__file__).parent / "fixtures" / "yolo26_outputs.npz"
LOW = -20.0   # logit -> ~0 after sigmoid
HIGH = 10.0   # logit -> ~1 after sigmoid


def _empty_outputs() -> dict[str, np.ndarray]:
    # Layer names/shapes as in the real yolo26s.hef (box 4ch, class 80ch per grid).
    out = {}
    for box_name, cls_name, g in (("yolo26s/conv61", "yolo26s/conv64", 80),
                                  ("yolo26s/conv77", "yolo26s/conv80", 40),
                                  ("yolo26s/conv91", "yolo26s/conv94", 20)):
        out[box_name] = np.zeros((g, g, 4), dtype=np.float32)
        out[cls_name] = np.full((g, g, 80), LOW, dtype=np.float32)
    return out


def _single_cell(cls_id: int) -> dict[str, np.ndarray]:
    out = _empty_outputs()
    gx, gy = 10, 5
    out["yolo26s/conv91"][gy, gx] = [1.0, 1.0, 1.0, 1.0]
    out["yolo26s/conv94"][gy, gx, cls_id] = HIGH
    return out


def test_synthetic_single_box():
    dets = decode_yolo26(_single_cell(16), conf_thresh=0.5)
    assert len(dets) == 1
    d = dets[0]
    assert isinstance(d, Detection)
    assert d.cls_id == 16
    expected = ((10.5 - 1) * 32 / 640, (5.5 - 1) * 32 / 640,
                (10.5 + 1) * 32 / 640, (5.5 + 1) * 32 / 640)
    assert d.box == pytest.approx(expected, abs=1e-3)
    assert 0.99 < d.conf <= 1.0


def test_threshold_filters():
    assert decode_yolo26(_single_cell(16), conf_thresh=0.99999) == []


def test_ignores_other_classes():
    assert decode_yolo26(_single_cell(2), conf_thresh=0.5) == []


def test_fixture_finds_dog_and_person():
    data = np.load(FIXTURE)
    outputs = {k: data[k] for k in data.files if not k.startswith("_")}
    dets = decode_yolo26(outputs, conf_thresh=0.4)
    assert any(d.cls_id == 16 for d in dets)
    assert any(d.cls_id == 0 for d in dets)
    for d in dets:
        x1, y1, x2, y2 = d.box
        assert 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0
