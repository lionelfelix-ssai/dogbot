import numpy as np
import pytest

from dogbot.pipeline import INPUT_SIZE, PAD_VALUE, UvcSource, letterbox_params


class FakeCap:
    """Stands in for cv2.VideoCapture: yields scripted frames (None = read failure)."""

    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.props = {}
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def read(self):
        if not self.frames:
            return False, None
        f = self.frames.pop(0)
        return (f is not None), f

    def release(self):
        self.released = True


def bgr(value, w=1280, h=720):
    img = np.zeros((h, w, 3), np.uint8)
    img[..., 0] = value          # blue channel only, so channel order is checkable
    return img


def make(frames, **kw):
    cap = FakeCap(frames)
    src = UvcSource("/dev/fake", open_capture=lambda dev: cap, warmup_s=0.05, **kw)
    return src, cap


def test_capture_returns_bgr_main_and_rgb_letterboxed_lores():
    src, _ = make([bgr(200)])
    main, lores = src.capture()
    assert main.shape == (720, 1280, 3)
    assert main[0, 0, 0] == 200 and main[0, 0, 2] == 0           # BGR for OpenCV/overlay
    assert lores.shape == (INPUT_SIZE, INPUT_SIZE, 3)
    _, (px, py), (nw, nh) = letterbox_params(1280, 720)
    assert lores[py + nh // 2, px + nw // 2, 2] == 200            # RGB for the Hailo
    assert lores[py + nh // 2, px + nw // 2, 0] == 0
    assert lores[0, 0, 0] == PAD_VALUE                             # gray letterbox bar


def test_frames_of_another_size_are_resized_to_main_size():
    src, _ = make([bgr(10, 1920, 1080)])
    main, _ = src.capture()
    assert main.shape == (720, 1280, 3)


def test_warmup_skips_black_frames():
    src, cap = make([bgr(0), bgr(0), bgr(90)])
    main, _ = src.capture()
    assert main[0, 0, 0] == 90


def test_read_failure_raises_so_the_pipeline_faults():
    src, _ = make([bgr(50), None])
    src.capture()
    with pytest.raises(RuntimeError, match="no frame"):
        src.capture()


def test_unopenable_device_raises_and_releases():
    cap = FakeCap([], opened=False)
    with pytest.raises(RuntimeError, match="cannot open"):
        UvcSource("/dev/nope", open_capture=lambda dev: cap)
    assert cap.released


def test_requests_mjpg_size_fps_and_small_buffer():
    import cv2
    _, cap = make([bgr(50)])
    assert cap.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
    assert cap.props[cv2.CAP_PROP_FRAME_WIDTH] == 1280
    assert cap.props[cv2.CAP_PROP_FRAME_HEIGHT] == 720
    assert cap.props[cv2.CAP_PROP_BUFFERSIZE] == 1


def test_close_releases():
    src, cap = make([bgr(50)])
    src.close()
    assert cap.released
