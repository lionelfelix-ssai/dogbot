"""Run the DogBot web app on a Mac with a simulated pipeline (no camera, no
Hailo, no sound).

    .venv/bin/python tools/dev_server.py [--port 8080] [--dir /tmp/dogbot-dev]

A synthetic living-room frame is annotated with the real overlay code. A
scripted 40 s loop walks through the states the UI must show: a person in
view (disarmed, counting down), armed and idle, a dog on the couch (intrusion
and rounds 1-3, with snapshots in the event log), then clear. Use --fault to
start in fault. Test blasts print instead of playing.
"""
from __future__ import annotations

import argparse
import math
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from dogbot.config import Settings, load_settings, save_settings
from dogbot.detection import Detection
from dogbot.events import EventStore
from dogbot.overlay import TrackedDet, draw_overlay
from dogbot.rules import PlayBlasts
from dogbot.web.app import AppState, make_server
from dogbot.zones import anchor, point_in_polygon

W, H = 1280, 720
DEFAULT_ZONE = [(0.18, 0.55), (0.82, 0.55), (0.86, 0.92), (0.14, 0.92)]


def living_room() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    for y in range(H):                      # wall-to-floor gradient
        t = y / H
        img[y] = (60 + 70 * t, 70 + 60 * t, 85 + 40 * t)
    cv2.rectangle(img, (0, 560), (W, H), (52, 64, 78), -1)          # floor
    cv2.rectangle(img, (230, 380), (1050, 640), (70, 60, 120), -1)   # couch back
    cv2.rectangle(img, (190, 470), (1090, 660), (90, 78, 150), -1)   # seat
    cv2.rectangle(img, (190, 420), (260, 660), (80, 70, 140), -1)    # arms
    cv2.rectangle(img, (1020, 420), (1090, 660), (80, 70, 140), -1)
    cv2.rectangle(img, (930, 90), (1150, 300), (40, 46, 56), -1)     # window
    cv2.rectangle(img, (940, 100), (1140, 290), (150, 120, 90), -1)
    cv2.circle(img, (140, 150), 40, (120, 200, 230), -1)             # lamp
    return img


class SimPipeline:
    """Scripted stand-in for dogbot.pipeline.Pipeline (status, latest_jpeg, raw_jpeg)."""

    CYCLE = 40.0

    def __init__(self, state: AppState, fault: bool = False):
        self._state = state
        self._bg = living_room()
        self._jpeg: bytes | None = None
        self._raw: bytes | None = None
        self._status: dict = {}
        self._fault = fault
        self._t0 = time.monotonic()
        self._phase = None
        self._round_logged = 0

    def latest_jpeg(self):
        return self._jpeg

    def raw_jpeg(self):
        return self._raw

    def status(self):
        return dict(self._status)

    def run(self, stop: threading.Event):
        while not stop.is_set():
            self.step()
            time.sleep(1 / 15)

    def step(self):
        s: Settings = self._state.get_settings()
        t = (time.monotonic() - self._t0) % self.CYCLE
        frame = self._bg.copy()
        tracks: list[TrackedDet] = []
        people = dogs = 0
        armed, guard, rnd, seq, occupied, until = False, "idle", 0, 0, False, 0.0

        if t < 8:
            phase = "person"
            people = 1
            x = 0.05 + 0.02 * math.sin(t)
            p = Detection(0, 0.91, (x, 0.12, x + 0.16, 0.95))
            self._person(frame, p)
            tracks.append(TrackedDet(p, None, False))
            until = s.rearm_grace_s
        elif t < 14:
            phase = "counting"
            until = max(0.0, 14 - t)
        elif t < 18:
            phase = "armed"
            armed = True
        elif t < 32:
            phase = "dog"
            armed, occupied = True, True
            dogs = 1
            dx = 0.40 + 0.04 * math.sin(t * 1.3)
            d = Detection(16, 0.88, (dx, 0.52, dx + 0.17, 0.86))
            self._dog(frame, d)
            in_zone = point_in_polygon(anchor(d.box), s.zone or DEFAULT_ZONE)
            tracks.append(TrackedDet(d, 7, in_zone))
            k = t - 18                       # rounds at 0, 4.5, 9 s
            rnd = 1 if k < 4.5 else 2 if k < 9 else 3
            seq = 1
            guard = "blasting" if (k % 4.5) < rnd * 0.9 else "gap"
        else:
            phase = "clear"
            armed = True
            guard = "paused" if t < 36 else "idle"

        fault = self._fault
        if fault or not s.enabled:
            guard = "held" if guard in ("blasting", "gap") else guard
        if fault:
            armed = False

        self._status = {
            "armed": armed, "enabled": s.enabled, "fault": fault, "state": guard,
            "round": rnd, "sequence": seq, "seconds_until_armed": round(until, 1),
            "fps": round(15 + math.sin(t), 1), "dogs": dogs, "people": people,
            "occupied": occupied,
        }
        ok, raw = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        self._raw = raw.tobytes()
        img = draw_overlay(frame, tracks, s.zone, self._status)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        self._jpeg = buf.tobytes()
        self._log_transitions(phase, rnd, s)

    def _log_transitions(self, phase, rnd, s):
        ev = self._state.events
        active = s.enabled and not self._fault
        if phase != self._phase:
            if phase == "person":
                ev.log("disarmed", {})
            elif phase == "armed":
                ev.log("armed", {})
            elif phase == "dog" and active:
                ev.log("intrusion", {}, snapshot_jpeg=self._jpeg)
            elif phase == "clear" and self._phase == "dog":
                ev.log("cleared", {})
            self._phase = phase
            self._round_logged = 0
        if phase == "dog" and active and rnd > self._round_logged:
            ev.log("round", {"sequence": 1, "round": rnd},
                   snapshot_jpeg=self._jpeg if rnd == 3 else None)
            self._round_logged = rnd

    @staticmethod
    def _person(img, d):
        x1, y1, x2, y2 = (int(d.box[0] * W), int(d.box[1] * H), int(d.box[2] * W), int(d.box[3] * H))
        cx = (x1 + x2) // 2
        cv2.circle(img, (cx, y1 + 45), 38, (150, 170, 200), -1)
        cv2.rectangle(img, (x1 + 25, y1 + 90), (x2 - 25, y2), (110, 80, 60), -1)

    @staticmethod
    def _dog(img, d):
        x1, y1, x2, y2 = (int(d.box[0] * W), int(d.box[1] * H), int(d.box[2] * W), int(d.box[3] * H))
        cv2.ellipse(img, ((x1 + x2) // 2, (y1 + y2) // 2 + 20), ((x2 - x1) // 2, (y2 - y1) // 3), 0, 0, 360, (60, 110, 170), -1)
        cv2.circle(img, (x2 - 30, y1 + 40), 34, (70, 120, 180), -1)
        cv2.circle(img, (x2 - 20, y1 + 32), 5, (20, 20, 20), -1)


class PrintSound:
    def handle(self, cmd):
        if isinstance(cmd, PlayBlasts):
            print(f"[sound] would play {cmd.count} blast(s) at level {cmd.level}", flush=True)

    def stop(self):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--dir", default=None, help="settings + events dir (default: a temp dir)")
    ap.add_argument("--fault", action="store_true", help="simulate a vision fault")
    args = ap.parse_args(argv)

    base = Path(args.dir or tempfile.mkdtemp(prefix="dogbot-dev-"))
    path = base / "settings.json"
    if not path.exists():
        save_settings(Settings(zone=list(DEFAULT_ZONE)), path)
    events = EventStore(base / "events.db", base / "snapshots")
    events.log("startup", {"dev": True})
    state = AppState(settings=load_settings(path), settings_path=path,
                     pipeline=None, events=events, sound=PrintSound())
    state.pipeline = SimPipeline(state, fault=args.fault)
    sim = threading.Thread(target=state.pipeline.run, args=(state.stop,), daemon=True)
    sim.start()
    print(f"DogBot dev server: http://{args.host}:{args.port}  (data in {base})", flush=True)
    try:
        make_server(state, args.host, args.port).run()
    except KeyboardInterrupt:
        pass  # uvicorn re-raises the captured Ctrl-C after a clean shutdown
    finally:
        state.shutdown()
        sim.join(timeout=1)


if __name__ == "__main__":
    main()
