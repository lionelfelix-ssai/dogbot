import threading

import numpy as np
import pytest

from dogbot.config import Settings
from dogbot.detection import Detection
from dogbot.events import EventStore
from dogbot.pipeline import Pipeline, letterbox_params, unletterbox
from dogbot.rules import PlayBlasts, StopSound

ZONE = [(0.1, 0.4), (0.9, 0.4), (0.9, 0.95), (0.1, 0.95)]
DOG_ON_COUCH = Detection(16, 0.9, (0.4, 0.5, 0.6, 0.9))      # anchor (0.5, 0.9)
DOG_ON_COUCH_2 = Detection(16, 0.85, (0.15, 0.45, 0.3, 0.8))  # anchor (0.225, 0.8)
DOG_ON_FLOOR = Detection(16, 0.9, (0.4, 0.0, 0.6, 0.3))      # anchor (0.5, 0.3)
PERSON = Detection(0, 0.8, (0.0, 0.0, 0.2, 0.8))
DT = 0.125  # exact in binary, so dwell/grace deadlines land on a tick


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class FakeSource:
    def __init__(self, size=(640, 640)):
        h, w = size
        self.main = np.zeros((h, w, 3), dtype=np.uint8)
        self.lores = np.zeros((640, 640, 3), dtype=np.uint8)
        self.fail = False

    def capture(self):
        if self.fail:
            raise RuntimeError("camera gone")
        return self.main, self.lores


class FakeDetector:
    """Returns the scripted detection list; `dets` may be set per tick."""

    def __init__(self):
        self.dets: list[Detection] = []
        self.fail = False

    def infer(self, rgb640):
        assert rgb640.shape == (640, 640, 3)
        if self.fail:
            raise RuntimeError("hailo timeout")
        return {"dets": list(self.dets)}


def passthrough_decoder(outputs, conf_thresh):
    return [d for d in outputs["dets"] if d.conf >= conf_thresh]


class FakeSound:
    def __init__(self):
        self.cmds = []
        self.fail = False

    def handle(self, cmd):
        self.cmds.append(cmd)
        if self.fail:
            raise OSError("aplay missing")

    def stop(self):
        self.cmds.append(StopSound())


@pytest.fixture
def rig(tmp_path):
    class Rig:
        pass

    r = Rig()
    r.settings = Settings(zone=list(ZONE))
    r.clock = FakeClock()
    r.source = FakeSource()
    r.detector = FakeDetector()
    r.sound = FakeSound()
    r.events = EventStore(tmp_path / "events.db", tmp_path / "snaps")
    r.extra_presence = []
    r.pipe = Pipeline(lambda: r.settings, r.events, r.sound, r.source, r.detector,
                      clock=r.clock, presence_sources=r.extra_presence,
                      decoder=passthrough_decoder)

    def run(seconds, dets=None):
        if dets is not None:
            r.detector.dets = list(dets)
        for _ in range(int(round(seconds / DT))):
            r.clock.t += DT
            r.pipe.step()

    def kinds():
        return [e["kind"] for e in reversed(r.events.list(limit=10000))]

    def plays():
        return [c for c in r.sound.cmds if isinstance(c, PlayBlasts)]

    r.run, r.kinds, r.plays = run, kinds, plays
    return r


def arm(r):
    r.run(61, [])
    assert r.pipe.status()["armed"]


def test_dog_on_couch_when_armed_blasts(rig):
    arm(rig)
    rig.run(1.1, [DOG_ON_COUCH])
    assert PlayBlasts(1, 1) in rig.sound.cmds
    kinds = rig.kinds()
    assert "armed" in kinds and "intrusion" in kinds
    intrusion = [e for e in rig.events.list(limit=1000) if e["kind"] == "intrusion"][0]
    assert intrusion["has_snapshot"]
    path = rig.events.snapshot_path(intrusion["id"])
    assert path.exists() and path.read_bytes()[:2] == b"\xff\xd8"


def test_person_present_never_blasts(rig):
    rig.run(120, [PERSON, DOG_ON_COUCH])
    assert rig.plays() == []
    assert not rig.pipe.status()["armed"]
    assert "armed" not in rig.kinds()


def test_detector_exception_enters_fault(rig):
    arm(rig)
    rig.run(1.2, [DOG_ON_COUCH])
    assert rig.pipe.status()["state"] == "blasting"
    n_plays = len(rig.plays())
    n_cmds = len(rig.sound.cmds)

    rig.detector.fail = True
    rig.run(DT)
    assert StopSound() in rig.sound.cmds[n_cmds:]
    st = rig.pipe.status()
    assert st["fault"] is True
    assert not st["armed"]
    errors = [e for e in rig.events.list(limit=1000) if e["kind"] == "error"]
    assert len(errors) == 1 and "hailo timeout" in errors[0]["detail"]["error"]

    rig.run(30)  # still failing, dog still scripted
    assert len(rig.plays()) == n_plays
    assert len([k for k in rig.kinds() if k == "error"]) == 1  # logged once

    rig.detector.fail = False
    rig.run(2 * DT)
    assert rig.pipe.status()["fault"] is True  # 2 good ticks are not enough
    rig.run(DT)
    st = rig.pipe.status()
    assert st["fault"] is False
    assert st["armed"] is False
    assert st["seconds_until_armed"] == pytest.approx(60, abs=0.01)
    rig.run(30, [DOG_ON_COUCH])  # fresh grace: still disarmed
    assert len(rig.plays()) == n_plays


def test_camera_exception_enters_fault(rig):
    arm(rig)
    rig.source.fail = True
    rig.run(DT)
    assert rig.pipe.status()["fault"] is True
    assert "error" in rig.kinds()


def test_decode_error_is_a_fault(tmp_path):
    clock = FakeClock()
    det = FakeDetector()
    events = EventStore(tmp_path / "e.db", tmp_path / "s")
    pipe = Pipeline(lambda: Settings(zone=list(ZONE)), events, FakeSound(), FakeSource(),
                    det, clock=clock)  # real decode_yolo26: {"dets": ...} has no heads
    clock.t += DT
    pipe.step()
    assert pipe.status()["fault"] is True
    assert [e["kind"] for e in events.list()] == ["error"]


def test_disabled_never_blasts(rig):
    rig.settings.enabled = False
    rig.run(61, [])
    rig.run(20, [DOG_ON_COUCH])
    assert rig.sound.cmds == []
    assert rig.pipe.status()["enabled"] is False


def test_disable_mid_blast_stops_within_one_tick_and_logs_held(rig):
    arm(rig)
    rig.run(1.2, [DOG_ON_COUCH])
    assert rig.pipe.status()["state"] == "blasting"
    n = len(rig.sound.cmds)
    rig.settings.enabled = False
    rig.run(DT)
    assert StopSound() in rig.sound.cmds[n:]
    assert rig.pipe.status()["state"] == "held"
    assert "held" in rig.kinds()


def test_two_dogs_swapping_ids_on_couch_no_restart(rig):
    arm(rig)
    # Two dogs on the couch; each tick they trade places or one vanishes,
    # so the tracker keeps re-assigning ids. Presence is "any dog in zone".
    patterns = [[DOG_ON_COUCH], [DOG_ON_COUCH_2], [DOG_ON_COUCH, DOG_ON_COUCH_2],
                [DOG_ON_COUCH_2], [DOG_ON_COUCH]]
    seen_ids = set()
    for i in range(160):  # 20 s
        rig.run(DT, patterns[i % len(patterns)])
        seen_ids.update(t.track_id for t in rig.pipe.last_tracks if t.track_id is not None)
    kinds = rig.kinds()
    assert kinds.count("intrusion") == 1
    assert rig.plays().count(PlayBlasts(1, 1)) == 1
    assert StopSound() not in rig.sound.cmds
    rounds = [e["detail"] for e in reversed(rig.events.list(limit=1000)) if e["kind"] == "round"]
    assert [(d["sequence"], d["round"]) for d in rounds][:3] == [(1, 1), (1, 2), (1, 3)]
    assert len(seen_ids) >= 2


def test_single_frame_dropout_does_not_cut_blast(rig):
    rig.settings.blast_s = 5.0  # long round, so a cut would be visible
    arm(rig)
    rig.run(1.2, [DOG_ON_COUCH])
    assert rig.pipe.status()["state"] == "blasting"
    n = len(rig.sound.cmds)
    rig.run(DT, [])               # one missed detection
    rig.run(0.5, [DOG_ON_COUCH])
    assert StopSound() not in rig.sound.cmds[n:]
    assert rig.pipe.status()["state"] == "blasting"


def test_dropout_hold_expires(rig):
    rig.settings.blast_s = 5.0
    arm(rig)
    rig.run(1.2, [DOG_ON_COUCH])
    n = len(rig.sound.cmds)
    rig.run(0.7, [])  # longer than the 0.5 s hold
    assert StopSound() in rig.sound.cmds[n:]
    assert rig.pipe.status()["state"] == "paused"


def test_fault_holds_escalation_without_clearing(rig):
    arm(rig)
    rig.run(1.2, [DOG_ON_COUCH])
    rig.detector.fail = True
    rig.run(20)  # longer than clear_s; no data must not count as "clear"
    assert "cleared" not in rig.kinds()
    assert rig.pipe.status()["state"] == "held"
    rig.detector.fail = False
    rig.run(61 + 0.3)  # recovery, then a fresh grace; the dog never left
    assert rig.pipe.status()["armed"]
    rig.run(6)
    assert rig.kinds().count("intrusion") == 1  # resumed, not restarted
    rounds = [(e["detail"]["sequence"], e["detail"]["round"])
              for e in reversed(rig.events.list(limit=1000)) if e["kind"] == "round"]
    assert rounds[:2] == [(1, 1), (1, 2)]


def test_sound_failure_is_logged_not_fault(rig):
    arm(rig)
    rig.sound.fail = True
    rig.run(1.2, [DOG_ON_COUCH])
    st = rig.pipe.status()
    assert st["fault"] is False
    errors = [e for e in rig.events.list(limit=1000) if e["kind"] == "error"]
    assert errors and "sound" in str(errors[0]["detail"])
    rig.run(1.0)  # keeps running
    assert rig.pipe.status()["state"] in ("blasting", "gap")


def test_single_person_frame_disarms(rig):
    arm(rig)
    rig.run(DT, [PERSON])
    st = rig.pipe.status()
    assert st["armed"] is False
    assert st["seconds_until_armed"] == pytest.approx(60, abs=0.01)
    assert rig.kinds()[-1] == "disarmed"


def test_low_conf_person_does_not_disarm(rig):
    arm(rig)
    rig.run(DT, [Detection(0, 0.3, PERSON.box)])
    assert rig.pipe.status()["armed"] is True


def test_extra_presence_source_keeps_disarmed(rig):
    rig.extra_presence.append(lambda: True)
    rig.run(120, [DOG_ON_COUCH])
    assert rig.plays() == []
    assert not rig.pipe.status()["armed"]


def test_dog_off_couch_never_blasts(rig):
    arm(rig)
    rig.run(10, [DOG_ON_FLOOR])
    assert rig.plays() == []


def test_unresolved_event_has_snapshot(rig):
    arm(rig)
    rig.run(60, [DOG_ON_COUCH])
    unresolved = [e for e in rig.events.list(limit=1000) if e["kind"] == "unresolved"]
    assert len(unresolved) == 1 and unresolved[0]["has_snapshot"]
    assert len(rig.plays()) == 6


def test_settings_change_rebuilds_dwell(rig):
    arm(rig)
    rig.settings.dwell_s = 3.0
    rig.run(2.0, [DOG_ON_COUCH])
    assert rig.plays() == []
    rig.run(1.1)
    assert rig.plays() == [PlayBlasts(1, 1)]
    assert rig.pipe.status()["armed"]  # dwell change left arming alone


def test_status_and_jpeg(rig):
    assert rig.pipe.latest_jpeg() is None
    rig.run(2 * DT, [PERSON, DOG_ON_FLOOR])
    st = rig.pipe.status()
    for key in ("armed", "enabled", "fault", "state", "round", "sequence",
                "seconds_until_armed", "fps", "dogs", "people"):
        assert key in st
    assert st["dogs"] == 1 and st["people"] == 1
    assert st["fps"] == pytest.approx(1 / DT, rel=0.05)
    jpg = rig.pipe.latest_jpeg()
    assert jpg[:2] == b"\xff\xd8"


def test_run_forever_stops_and_silences(rig):
    stop = threading.Event()
    calls = {"n": 0}
    orig = rig.pipe.step

    def step():
        calls["n"] += 1
        rig.clock.t += DT
        orig()
        if calls["n"] >= 5:
            stop.set()

    rig.pipe.step = step
    rig.pipe.run_forever(stop)
    assert calls["n"] == 5
    assert isinstance(rig.sound.cmds[-1], StopSound)


def test_letterbox_roundtrip_16x9():
    scale, (px, py), (nw, nh) = letterbox_params(1280, 720, 640)
    assert (px, py, nw, nh) == (0, 140, 640, 360)
    # a box spanning the full content area maps to the full frame
    box = (0.0, 140 / 640, 1.0, 500 / 640)
    assert unletterbox(box, 1280, 720, 640) == pytest.approx((0.0, 0.0, 1.0, 1.0))
    # boxes in the padding clip to the frame
    assert unletterbox((0.2, 0.0, 0.4, 0.1), 1280, 720, 640)[1] == 0.0


def test_letterbox_identity_for_square_frame():
    assert unletterbox((0.1, 0.2, 0.3, 0.4), 640, 640, 640) == pytest.approx((0.1, 0.2, 0.3, 0.4))


def test_pipeline_maps_letterboxed_boxes_to_frame(tmp_path):
    clock = FakeClock()
    det = FakeDetector()
    # dog anchor at model y = (140 + 0.9*360)/640 -> frame y = 0.9
    det.dets = [Detection(16, 0.9, (0.4, (140 + 0.5 * 360) / 640, 0.6, (140 + 0.9 * 360) / 640))]
    pipe = Pipeline(lambda: Settings(zone=list(ZONE)), EventStore(tmp_path / "e.db", tmp_path / "s"),
                    FakeSound(), FakeSource(size=(720, 1280)), det, clock=clock,
                    decoder=passthrough_decoder)
    clock.t += DT
    pipe.step()
    (t,) = pipe.last_tracks
    assert t.det.box == pytest.approx((0.4, 0.5, 0.6, 0.9))
    assert t.in_zone


def test_person_present_never_blasts_with_zero_grace(rig):
    rig.settings.rearm_grace_s = 0
    rig.run(30, [PERSON, DOG_ON_COUCH])
    assert rig.plays() == []
    st = rig.pipe.status()
    assert st["armed"] is False and st["people"] == 1


def test_extra_presence_never_blasts_with_zero_grace(rig):
    rig.settings.rearm_grace_s = 0
    rig.extra_presence.append(lambda: True)
    rig.run(30, [DOG_ON_COUCH])
    assert rig.plays() == []
    st = rig.pipe.status()
    assert st["armed"] is False and st["people"] == 0
