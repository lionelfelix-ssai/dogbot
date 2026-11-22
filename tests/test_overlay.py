import numpy as np

from dogbot.detection import Detection
from dogbot.overlay import TrackedDet, draw_overlay

ZONE = [(0.1, 0.4), (0.9, 0.4), (0.9, 0.95), (0.1, 0.95)]
STATUS = {"armed": True, "enabled": True, "fault": False, "state": "idle",
          "round": 0, "sequence": 0, "seconds_until_armed": 0.0,
          "occupied": False}


def test_overlay_draws_reticle():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dog = TrackedDet(Detection(16, 0.9, (0.4, 0.5, 0.6, 0.9)), track_id=7, in_zone=True)
    out = draw_overlay(frame, [dog], ZONE, STATUS)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    cx, cy = int(0.5 * 1280), int(0.7 * 720)
    r = int(0.35 * min(0.2 * 1280, 0.4 * 720))
    column = out[cy - r - 2:cy + r + 3, cx]
    assert column.any(), "reticle should mark the box-center column"
    # crosshair tick just inside the circle, above center
    assert out[cy - r + 3:cy - r // 2, cx].any()
    # the input frame is not modified
    assert not frame.any()


def test_reticle_red_in_zone_amber_outside():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    box = (0.4, 0.5, 0.6, 0.9)
    cx, cy = int(0.5 * 1280), int(0.7 * 720)
    r = int(0.35 * min(0.2 * 1280, 0.4 * 720))
    red = draw_overlay(frame, [TrackedDet(Detection(16, 0.9, box), 1, True)], [], STATUS)
    amber = draw_overlay(frame, [TrackedDet(Detection(16, 0.9, box), 1, False)], [], STATUS)
    b, g, rr = red[cy - r, cx].astype(int)
    assert rr > 150 and g < 80 and b < 80
    b, g, rr = amber[cy - r, cx].astype(int)
    assert rr > 150 and g > 100 and b < 80


def test_untracked_dog_has_no_reticle():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    dog = TrackedDet(Detection(16, 0.9, (0.4, 0.5, 0.6, 0.9)), track_id=None, in_zone=True)
    out = draw_overlay(frame, [dog], [], STATUS)
    cx, cy = int(0.5 * 1280), int(0.7 * 720)
    assert not out[cy - 20:cy + 20, cx - 20:cx + 20].any()


def test_banner_text_varies_with_status():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    outs = [draw_overlay(frame, [], ZONE, dict(STATUS, **kw)) for kw in (
        {}, {"armed": False, "seconds_until_armed": 42.0},
        {"enabled": False}, {"fault": True})]
    tops = [o[:30].tobytes() for o in outs]
    assert len(set(tops)) == 4


def test_zone_color_follows_state():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    # left edge of the zone, away from the banner
    px = (int(0.1 * 640), int(0.7 * 360))

    def color(**kw):
        out = draw_overlay(frame, [], ZONE, dict(STATUS, **kw))
        return tuple(int(v) for v in out[px[1], px[0]])

    idle, occ, blast = color(), color(occupied=True), color(state="blasting", occupied=True)
    assert idle[1] > 150 and idle[2] < 80           # green
    assert occ[2] > 150 and occ[1] > 100            # amber
    assert blast[2] > 150 and blast[1] < 80         # red
