"""The live loop: camera -> Hailo -> decode -> tracker -> couch presence ->
arming -> rules engine -> sound + event log, plus the annotated JPEG.

Frame geometry (letterbox): the camera gives a 1280x720 main stream (for the
web view) and a 640x360 lores stream of the *same* field of view (equal
ScalerCrop). The lores image is padded to 640x640 with gray (114) bars top
and bottom, exactly like tools/capture_fixture.py, so the model sees
undistorted objects. decode_yolo26 returns boxes normalized to that padded
640x640 input; `unletterbox` maps them back to the main frame (0-1), the
coordinate system of the couch zone and the overlay. Both directions use
`letterbox_params`, so source and decode cannot drift apart.

Fault (Review Focus 2, ruling R9): any exception from capture, inference,
decode or tracking puts the pipeline in fault: StopSound goes to the sound
action, one `error` event is logged, and CouchGuard is fed active=False with
the last known `occupied`, so missing data never counts as "couch clear".
After FAULT_RECOVER_TICKS good ticks the fault clears and a fresh Arming
grace starts.
"""
from __future__ import annotations

import argparse
import logging
import time
from typing import Callable

import cv2
import numpy as np
import supervision as sv

from dogbot.arming import Arming
from dogbot.config import Settings
from dogbot.detection import Detection
from dogbot.overlay import DOG_CLS, PERSON_CLS, TrackedDet, draw_overlay
from dogbot.rules import CouchGuard, PlayBlasts, StopSound
from dogbot.yolo26 import decode_yolo26
from dogbot.zones import CouchPresence, anchor, point_in_polygon

log = logging.getLogger(__name__)

INPUT_SIZE = 640
PAD_VALUE = 114
DROPOUT_HOLD_S = 0.5       # ruling R7
FAULT_RECOVER_TICKS = 3
JPEG_QUALITY = 75
JPEG_EVERY = 2
FPS_ALPHA = 0.1
SNAPSHOT_KINDS = ("intrusion", "unresolved")
FAULT_BACKOFF_S = 0.2        # between steps while in fault: no CPU spin
STEP_ERROR_BACKOFF_S = 0.1   # after a non-vision exception in step()


def letterbox_params(frame_w: int, frame_h: int, size: int = INPUT_SIZE):
    """-> (scale, (pad_x, pad_y), (content_w, content_h)) for fitting a
    frame_w x frame_h image into a size x size square, centered."""
    scale = min(size / frame_w, size / frame_h)
    nw, nh = int(round(frame_w * scale)), int(round(frame_h * scale))
    return scale, ((size - nw) // 2, (size - nh) // 2), (nw, nh)


def pad_to_square(img: np.ndarray, frame_w: int, frame_h: int,
                  size: int = INPUT_SIZE) -> np.ndarray:
    """Letterbox `img` (already content_w x content_h, or resized to it)."""
    _, (px, py), (nw, nh) = letterbox_params(frame_w, frame_h, size)
    if img.shape[:2] != (nh, nw):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    out[py:py + nh, px:px + nw] = img
    return out


def unletterbox(box, frame_w: int, frame_h: int, size: int = INPUT_SIZE):
    """Model-input-normalized box -> frame-normalized box, clipped to [0, 1]."""
    _, (px, py), (nw, nh) = letterbox_params(frame_w, frame_h, size)
    x1, y1, x2, y2 = box
    fx = lambda v: min(1.0, max(0.0, (v * size - px) / nw))  # noqa: E731
    fy = lambda v: min(1.0, max(0.0, (v * size - py) / nh))  # noqa: E731
    return (fx(x1), fy(y1), fx(x2), fy(y2))


class PicameraSource:
    """IMX708 via Picamera2: main 1280x720 BGR array, lores 640x360 RGB array
    padded to 640x640.

    Picamera2 names formats after libcamera, so the array channel order is
    the reverse of the name: "RGB888" -> B,G,R bytes (OpenCV order) and
    "BGR888" -> R,G,B bytes (verified on the Pi against an rpicam-still JPEG
    and picamera2/request.py FORMAT_TABLE).
    """

    def __init__(self, size=(1280, 720), lens_position: float | None = None):
        from libcamera import controls
        from picamera2 import Picamera2

        self.size = size
        _, _, lores = letterbox_params(*size)
        self._cam = Picamera2()
        cfg = self._cam.create_video_configuration(
            main={"size": size, "format": "RGB888"},
            lores={"size": lores, "format": "BGR888"},
            buffer_count=4)
        self._cam.configure(cfg)
        self._cam.start()
        if lens_position is None:
            self._cam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        else:
            self._cam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                    "LensPosition": float(lens_position)})

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        req = self._cam.capture_request()
        try:
            main = req.make_array("main")
            lores = req.make_array("lores")
        finally:
            req.release()
        return main, pad_to_square(lores, *self.size)

    def close(self) -> None:
        self._cam.stop()
        self._cam.close()


class Pipeline:
    def __init__(self, settings_getter: Callable[[], Settings], events, sound,
                 frame_source, detector, clock: Callable[[], float] = time.monotonic,
                 presence_sources: list[Callable[[], bool]] | None = None,
                 decoder=decode_yolo26):
        self._settings = settings_getter
        self._events = events
        self._sound = sound
        self._source = frame_source
        self._detector = detector
        self._clock = clock
        self._presence_sources = presence_sources if presence_sources is not None else []
        self._decode = decoder

        s = settings_getter()
        now = clock()
        self._presence = CouchPresence(s.dwell_s)
        self._arming = Arming(s.rearm_grace_s, start=now)
        self._guard = CouchGuard(s)
        self._tracker = sv.ByteTrack()

        self._armed = False
        self._occupied = False
        self._last_dog_in_zone: float | None = None
        self._fault = False
        self._step_faulted = False
        self._pipeline_error = False
        self._good_ticks = 0
        self._last_tick: float | None = None
        self._fps = 0.0
        self._frame_no = 0
        self._last_frame: np.ndarray | None = None
        self.last_tracks: list[TrackedDet] = []
        self._jpeg: bytes | None = None
        self._status = self._build_status(s, now, [], [])
        self._status["updated_at"] = None   # no tick yet: the watchdog arms on the first

    # -- public -----------------------------------------------------------

    def latest_jpeg(self) -> bytes | None:
        return self._jpeg

    def raw_jpeg(self) -> bytes | None:
        """The last camera frame without overlay, JPEG-encoded on demand (for
        the zone editor). Safe from another thread: each capture is a new
        array and the reference is swapped whole."""
        frame = self._last_frame
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes() if ok else None

    def status(self) -> dict:
        return dict(self._status)

    def run_forever(self, stop_event) -> None:
        try:
            while not stop_event.is_set():
                self._step_faulted = False
                try:
                    self.step()
                except Exception as e:
                    # Not a vision fault (those are handled in step); e.g. the
                    # event DB failed. Silence, flag it, log once per streak.
                    log.exception("pipeline step failed")
                    self._safe_stop()
                    self._on_step_error(e)
                    stop_event.wait(STEP_ERROR_BACKOFF_S)
                    continue
                if self._pipeline_error:
                    self._pipeline_error = False
                    self._status = {**self._status, "pipeline_error": False}
                if self._step_faulted:
                    stop_event.wait(FAULT_BACKOFF_S)
        finally:
            self._safe_stop()

    def _on_step_error(self, exc: Exception) -> None:
        first = not self._pipeline_error
        self._pipeline_error = True
        # The loop is alive (just failing), so updated_at advances: the
        # watchdog is for a *hung* loop. The UI shows pipeline_error.
        self._status = {**self._status, "pipeline_error": True,
                        "updated_at": self._clock()}
        if first:
            try:
                self._events.log("error", {"source": "pipeline", "error": repr(exc)})
            except Exception:
                log.exception("could not log pipeline error event")

    def step(self) -> None:
        now = self._clock()
        s = self._settings()
        self._sync_settings(s, now)
        self._tick_fps(now)

        try:
            main, lores = self._source.capture()
            outputs = self._detector.infer(lores)
            dets = self._decode(outputs, min(s.dog_conf, s.person_conf))
            h, w = main.shape[:2]
            dogs = [self._to_frame(d, w, h) for d in dets
                    if d.cls_id == DOG_CLS and d.conf >= s.dog_conf]
            people = [self._to_frame(d, w, h) for d in dets
                      if d.cls_id == PERSON_CLS and d.conf >= s.person_conf]
            dog_tracks = self._track(dogs, w, h, s.zone)
        except Exception as e:  # camera, Hailo, decode, tracker
            self._on_fault(s, now, e)
            return

        self._last_frame = main
        if self._fault:
            self._good_ticks += 1
            if self._good_ticks >= FAULT_RECOVER_TICKS:
                self._fault = False
                self._arming = Arming(s.rearm_grace_s, start=now)
                self._events.log("error", {"source": "vision", "recovered": True})

        # couch presence, with a short hold over single missed frames (R7)
        if any(t.in_zone for t in dog_tracks):
            self._last_dog_in_zone = now
        dog_in_zone = (self._last_dog_in_zone is not None
                       and now - self._last_dog_in_zone <= DROPOUT_HOLD_S + 1e-9)
        self._occupied = self._presence.update(now, dog_in_zone)

        # arming: any single frame with a person, or any extra source
        person = bool(people) or self._extra_presence()
        # A person this tick always means disarmed, even with rearm_grace_s = 0.
        armed = self._arming.update(now, person) and not self._fault and not person
        if not self._fault and armed != self._armed:
            self._events.log("armed" if armed else "disarmed", {})
        if not self._fault:
            self._armed = armed

        active = s.enabled and self._armed and not self._fault and not person
        rule_events = self._guard_update(now, active, self._occupied)

        tracks = dog_tracks + [TrackedDet(p, None, False) for p in people]
        self.last_tracks = tracks
        self._status = self._build_status(s, now, dogs, people)
        self._render(s, rule_events)
        self._log_rule_events(rule_events)

    # -- internals ----------------------------------------------------------

    def _sync_settings(self, s: Settings, now: float) -> None:
        self._guard.settings = s
        if s.dwell_s != self._presence.dwell_s:
            self._presence = CouchPresence(s.dwell_s)
        if s.rearm_grace_s != self._arming.rearm_grace_s:
            # Rebuilt from now: a grace change never arms early.
            self._arming = Arming(s.rearm_grace_s, start=now)

    def _tick_fps(self, now: float) -> None:
        if self._last_tick is not None and now > self._last_tick:
            inst = 1.0 / (now - self._last_tick)
            self._fps = inst if self._fps == 0 else (1 - FPS_ALPHA) * self._fps + FPS_ALPHA * inst
        self._last_tick = now

    @staticmethod
    def _to_frame(d: Detection, w: int, h: int) -> Detection:
        return Detection(d.cls_id, d.conf, unletterbox(d.box, w, h))

    def _track(self, dogs: list[Detection], w: int, h: int, zone) -> list[TrackedDet]:
        n = len(dogs)
        xyxy = np.array([[d.box[0] * w, d.box[1] * h, d.box[2] * w, d.box[3] * h]
                         for d in dogs], dtype=np.float32).reshape(n, 4)
        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=np.array([d.conf for d in dogs], dtype=np.float32),
            class_id=np.full(n, DOG_CLS, dtype=int),
            data={"idx": np.arange(n)})
        tracked = self._tracker.update_with_detections(sv_dets)
        ids: dict[int, int] = {}
        if len(tracked) and tracked.tracker_id is not None:
            for i, tid in zip(tracked.data["idx"], tracked.tracker_id):
                ids[int(i)] = int(tid)
        # Presence never depends on the tracker: every dog counts, tracked or not.
        return [TrackedDet(d, ids.get(i), point_in_polygon(anchor(d.box), zone))
                for i, d in enumerate(dogs)]

    def _extra_presence(self) -> bool:
        for src in self._presence_sources:
            try:
                if src():
                    return True
            except Exception:
                log.exception("presence source failed; treating as person present")
                return True
        return False

    def _guard_update(self, now: float, active: bool, occupied: bool) -> list:
        prev = self._guard.state
        cmds, rule_events = self._guard.update(now, active, occupied)
        for cmd in cmds:
            self._dispatch(cmd)
        if self._guard.state == "held" and prev != "held":
            rule_events.append(_Held(self._guard))
        return rule_events

    def _dispatch(self, cmd) -> None:
        try:
            self._sound.handle(cmd)
        except Exception as e:  # R11: sound failure is not a vision fault
            log.exception("sound action failed")
            self._events.log("error", {"source": "sound", "command": type(cmd).__name__,
                                       "error": repr(e)})

    def _safe_stop(self) -> None:
        try:
            self._sound.stop()
        except Exception:
            log.exception("sound stop failed")

    def _on_fault(self, s: Settings, now: float, exc: Exception) -> None:
        self._step_faulted = True
        self._good_ticks = 0
        if not self._fault:
            self._fault = True
            log.error("vision fault: %r", exc)
            self._dispatch(StopSound())
            self._events.log("error", {"source": "vision", "error": repr(exc)})
            if self._armed:
                self._armed = False
                self._events.log("disarmed", {"reason": "fault"})
        rule_events = self._guard_update(now, False, self._occupied)
        self.last_tracks = []
        self._status = self._build_status(s, now, [], [])
        self._render(s, rule_events)
        self._log_rule_events(rule_events)

    def _build_status(self, s: Settings, now: float, dogs, people) -> dict:
        return {
            "armed": self._armed,
            "enabled": s.enabled,
            "fault": self._fault,
            "state": self._guard.state,
            "round": self._guard.round,
            "sequence": self._guard.sequence,
            "seconds_until_armed": self._arming.seconds_until_armed(now),
            "fps": round(self._fps, 1),
            "dogs": len(dogs),
            "people": len(people),
            "occupied": self._occupied,
            "pipeline_error": self._pipeline_error,
            "updated_at": now,
        }

    def _render(self, s: Settings, rule_events) -> None:
        self._frame_no += 1
        need_snapshot = any(e.kind in SNAPSHOT_KINDS for e in rule_events)
        if self._last_frame is None:
            return
        if not need_snapshot and self._frame_no % JPEG_EVERY:
            return
        img = draw_overlay(self._last_frame, self.last_tracks, s.zone, self._status)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if ok:
            self._jpeg = buf.tobytes()

    def _log_rule_events(self, rule_events) -> None:
        for e in rule_events:
            snap = self._jpeg if e.kind in SNAPSHOT_KINDS else None
            self._events.log(e.kind, e.detail, snapshot_jpeg=snap)


class _Held:
    """`held` event (CouchGuard entered held), logged like a RuleEvent."""

    kind = "held"

    def __init__(self, guard: CouchGuard):
        self.detail = {"sequence": guard.sequence, "round": guard.round}


# -- Pi smoke run: python -m dogbot.pipeline --once 50 ------------------------

class _NullSound:
    def handle(self, cmd) -> None:
        if isinstance(cmd, PlayBlasts):
            print(f"  (would play {cmd.count} blasts at level {cmd.level})")

    def stop(self) -> None:
        pass


class _NullEvents:
    def log(self, kind, detail, snapshot_jpeg=None) -> int:
        print(f"  event {kind} {detail}")
        return 0


def main(argv=None) -> None:
    from dogbot.config import load_settings, settings_path
    from dogbot.hailo_runner import HailoDetector

    ap = argparse.ArgumentParser(description="DogBot pipeline smoke run (no sound, no DB)")
    ap.add_argument("--once", type=int, default=50, metavar="N", help="ticks to run")
    ap.add_argument("--hef", default="/home/admin/models/yolo26s.hef")
    ap.add_argument("--lens-position", type=float, default=None)
    ap.add_argument("--save", default=None, help="write the last overlay JPEG here")
    args = ap.parse_args(argv)

    settings = load_settings(settings_path())
    source = PicameraSource(lens_position=args.lens_position)
    detector = HailoDetector(args.hef)
    try:
        pipe = Pipeline(lambda: settings, _NullEvents(), _NullSound(), source, detector)
        for _ in range(5):  # warm-up (AE/AWB, first inference)
            pipe.step()
        t0 = time.monotonic()
        seen = {"dogs": 0, "people": 0}
        for i in range(args.once):
            pipe.step()
            st = pipe.status()
            seen["dogs"] = max(seen["dogs"], st["dogs"])
            seen["people"] = max(seen["people"], st["people"])
            if st["dogs"] or st["people"]:
                confs = ", ".join(f"{'dog' if t.det.cls_id == DOG_CLS else 'person'}"
                                  f"{'#' + str(t.track_id) if t.track_id else ''}"
                                  f" {t.det.conf:.2f}" for t in pipe.last_tracks)
                print(f"tick {i:3d}: {confs}")
        dt = time.monotonic() - t0
        st = pipe.status()
        print(f"{args.once} ticks in {dt:.2f}s -> {args.once / dt:.1f} FPS "
              f"(ema {st['fps']}); max dogs {seen['dogs']}, max people {seen['people']}; "
              f"fault={st['fault']}")
        if args.save and pipe.latest_jpeg():
            with open(args.save, "wb") as f:
                f.write(pipe.latest_jpeg())
            print(f"wrote {args.save}")
    finally:
        detector.close()
        source.close()


if __name__ == "__main__":
    main()
