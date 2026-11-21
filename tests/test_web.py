import asyncio
import json
import threading
import time

import warnings

import pytest
from starlette.exceptions import StarletteDeprecationWarning

# Starlette 1.10 prefers `httpx2` for its TestClient and warns once at import
# when only `httpx` (our dev extra) is installed. Silence just that warning so
# the suite still runs with -W error.
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`",
                            category=StarletteDeprecationWarning)
    from fastapi.testclient import TestClient

from dogbot.config import Settings, load_settings, save_settings
from dogbot.events import EventStore
from dogbot.rules import PlayBlasts, StopSound
from dogbot.web.app import AppState, create_app, mjpeg_frames

ZONE = [(0.1, 0.4), (0.9, 0.4), (0.9, 0.95), (0.1, 0.95)]
JPEG_A = b"\xff\xd8 frame A \xff\xd9"
JPEG_B = b"\xff\xd8 frame B \xff\xd9"
RAW = b"\xff\xd8 raw frame \xff\xd9"


class FakePipeline:
    """Status mirrors the live settings, like the real pipeline's next tick."""

    def __init__(self, settings_getter):
        self._settings = settings_getter
        self.jpeg = JPEG_A
        self.raw = RAW
        self.armed = False
        self.jpeg_calls = 0
        self.updated_at = "now"

    def status(self):
        s = self._settings()
        return {"armed": self.armed, "enabled": s.enabled, "fault": False,
                "state": "idle", "round": 0, "sequence": 0,
                "seconds_until_armed": 42.0, "fps": 30.0, "dogs": 0,
                "people": 1, "occupied": False, "pipeline_error": False,
                "updated_at": time.monotonic() if self.updated_at == "now" else self.updated_at}

    def latest_jpeg(self):
        self.jpeg_calls += 1
        return self.jpeg

    def raw_jpeg(self):
        return self.raw


class FakeSound:
    def __init__(self):
        self.cmds = []
        self.fail = False

    def handle(self, cmd):
        if self.fail:
            raise OSError("aplay missing")
        self.cmds.append(cmd)

    def stop(self):
        self.cmds.append(StopSound())


@pytest.fixture
def rig(tmp_path):
    class Rig:
        pass

    r = Rig()
    r.path = tmp_path / "config" / "settings.json"
    save_settings(Settings(zone=list(ZONE)), r.path)
    r.events = EventStore(tmp_path / "events.db", tmp_path / "snaps")
    r.sound = FakeSound()
    r.state = AppState(settings=load_settings(r.path), settings_path=r.path,
                       pipeline=None, events=r.events, sound=r.sound)
    r.pipe = FakePipeline(r.state.get_settings)
    r.state.pipeline = r.pipe
    r.client = TestClient(create_app(r.state))

    def kinds():
        return [e["kind"] for e in r.events.list(limit=500)]

    r.kinds = kinds
    return r


# -- zone (Review Focus 5) ----------------------------------------------------

@pytest.mark.parametrize("points", [
    [[0.1, 0.1], [0.9, 0.9]],                        # 2 points
    [],                                              # empty
    [[0.1, 0.1], [0.9, 0.1], [1.2, 0.9]],            # x out of range
    [[0.1, 0.1], [0.9, 0.1], [0.5, -0.01]],          # y out of range
    [[0.1, 0.1], [0.9, 0.1], [0.5]],                 # not a pair
    [[0.1, 0.1], [0.9, 0.1], ["0.5", 0.9]],          # string coordinate
    [[0.1, 0.1], [0.9, 0.1], [True, 0.9]],           # bool coordinate
    "nope",
])
def test_bad_zone_rejected_and_old_zone_kept(rig, points):
    r = rig.client.put("/api/zone", json={"points": points})
    assert r.status_code == 422
    assert rig.state.get_settings().zone == ZONE
    assert load_settings(rig.path).zone == ZONE
    assert "zone" not in rig.kinds()


def test_zone_without_points_key_rejected(rig):
    assert rig.client.put("/api/zone", json={"pts": ZONE}).status_code == 422
    assert rig.state.get_settings().zone == ZONE


def test_valid_zone_saved_and_logged(rig):
    new = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8], [0.5, 0.95]]
    r = rig.client.put("/api/zone", json={"points": new})
    assert r.status_code == 200
    assert r.json()["zone"] == new
    expect = [tuple(p) for p in new]
    assert rig.state.get_settings().zone == expect
    assert load_settings(rig.path).zone == expect
    ev = rig.events.list(kinds={"zone"})
    assert len(ev) == 1 and ev[0]["detail"]["points"] == 5


# -- settings -------------------------------------------------------------------

def test_get_settings(rig):
    body = rig.client.get("/api/settings").json()
    assert body["dog_conf"] == 0.5
    assert body["levels"] == {"1": 0.4, "2": 0.7, "3": 1.0}
    assert body["zone"] == [list(p) for p in ZONE]


@pytest.mark.parametrize("payload", [
    {"dog_conf": 2},
    {"dog_conf": True},
    {"rearm_grace_s": 0},
    {"tone_hz": 50},
    {"sequences": 0},
    {"levels": {"1": 1.5}},
    {"levels": {"4": 0.5}},
    {"levels": {"one": 0.5}},
    {"levels": [0.1, 0.2, 0.3]},
    {"level_db": {"2": "loud"}},
    {"enabled": "yes"},
    {"audio_device": ""},
    {"audio_device": "file:'|touch /tmp/pwned'"},   # C-1: alsa file plugin -> shell
    {"audio_device": "hw:1,0 |sh"},
    {"bogus": 1},                       # unknown key
    {"zone": [[0, 0], [1, 0], [1, 1]]},  # zone only via /api/zone
    {"dog_conf": 0.6, "bogus": 1},      # one bad key spoils the update
])
def test_bad_settings_rejected_and_live_settings_untouched(rig, payload):
    before = rig.state.get_settings()
    file_before = rig.path.read_text()
    r = rig.client.put("/api/settings", json=payload)
    assert r.status_code == 422, r.text
    assert rig.state.get_settings() is before
    assert before == load_settings(rig.path)
    assert rig.path.read_text() == file_before
    assert "settings" not in rig.kinds()


def test_settings_body_must_be_object(rig):
    assert rig.client.put("/api/settings", json=[1, 2]).status_code == 422


def test_nan_and_infinity_rejected(rig):
    for raw in ('{"dog_conf": NaN}', '{"clear_s": Infinity}'):
        r = rig.client.put("/api/settings", content=raw,
                           headers={"content-type": "application/json"})
        assert r.status_code == 422
    assert rig.state.get_settings().clear_s == 10


def test_partial_update_merges_saves_and_logs_changed_keys(rig):
    before = rig.state.get_settings()
    r = rig.client.put("/api/settings", json={
        "dog_conf": 0.6, "levels": {"2": 0.55}, "level_db": {"3": 92.5},
        "clear_s": 10})  # unchanged value: not reported as changed
    assert r.status_code == 200, r.text
    s = rig.state.get_settings()
    assert s is not before                      # replaced as a whole
    assert before.dog_conf == 0.5               # old object never mutated
    assert before.levels == {1: 0.4, 2: 0.7, 3: 1.0}
    assert s.dog_conf == 0.6
    assert s.levels == {1: 0.4, 2: 0.55, 3: 1.0}
    assert s.level_db == {1: None, 2: None, 3: 92.5}
    assert s.zone == ZONE
    assert load_settings(rig.path) == s
    assert r.json()["levels"] == {"1": 0.4, "2": 0.55, "3": 1.0}
    ev = rig.events.list(kinds={"settings"})
    assert len(ev) == 1
    assert sorted(ev[0]["detail"]["changed"]) == ["dog_conf", "level_db", "levels"]


def test_noop_update_logs_nothing(rig):
    r = rig.client.put("/api/settings", json={"dog_conf": 0.5})
    assert r.status_code == 200
    assert "settings" not in rig.kinds()


def test_save_failure_leaves_live_settings(rig, monkeypatch):
    before = rig.state.get_settings()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("dogbot.web.app.save_settings", boom)
    r = rig.client.put("/api/settings", json={"dog_conf": 0.7})
    assert r.status_code == 500
    assert rig.state.get_settings() is before


def test_concurrent_updates_do_not_lose_writes(rig):
    def put(key, val):
        rig.client.put("/api/settings", json={key: val})

    threads = [threading.Thread(target=put, args=("levels", {str(n): 0.1 * n}))
               for n in (1, 2, 3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert rig.state.get_settings().levels == pytest.approx({1: 0.1, 2: 0.2, 3: 0.3})


# -- enabled ----------------------------------------------------------------------

def test_disable_shows_in_status_and_logs(rig):
    r = rig.client.post("/api/enabled", json={"enabled": False})
    assert r.status_code == 200
    assert rig.client.get("/api/status").json()["enabled"] is False
    assert load_settings(rig.path).enabled is False
    assert "disabled" in rig.kinds()
    rig.client.post("/api/enabled", json={"enabled": True})
    assert rig.client.get("/api/status").json()["enabled"] is True
    assert rig.kinds()[0] == "enabled"


def test_disable_cuts_running_sound(rig):
    rig.client.post("/api/test-blast", json={"level": 3})
    rig.sound.cmds.clear()
    assert rig.client.post("/api/enabled", json={"enabled": False}).status_code == 200
    assert rig.sound.cmds == [StopSound()]


def test_disable_via_settings_also_cuts_sound(rig):
    rig.sound.cmds.clear()
    assert rig.client.put("/api/settings", json={"enabled": False}).status_code == 200
    assert rig.sound.cmds == [StopSound()]


def test_enable_does_not_touch_sound(rig):
    rig.client.post("/api/enabled", json={"enabled": False})
    rig.sound.cmds.clear()
    rig.client.post("/api/enabled", json={"enabled": True})
    assert rig.sound.cmds == []


def test_disable_without_sound_is_fine(rig):
    rig.state.sound = None
    assert rig.client.post("/api/enabled", json={"enabled": False}).status_code == 200


@pytest.mark.parametrize("payload", [{"enabled": "false"}, {"enabled": 0}, {}])
def test_enabled_must_be_bool(rig, payload):
    assert rig.client.post("/api/enabled", json=payload).status_code == 422
    assert rig.state.get_settings().enabled is True


# -- test blast ----------------------------------------------------------------------

def test_test_blast_plays_even_when_disarmed(rig):
    assert rig.client.get("/api/status").json()["armed"] is False
    r = rig.client.post("/api/test-blast", json={"level": 3})
    assert r.status_code == 200
    assert rig.sound.cmds == [PlayBlasts(1, 3)]
    ev = rig.events.list(kinds={"test_blast"})
    assert ev[0]["detail"]["level"] == 3


@pytest.mark.parametrize("payload", [{"level": 0}, {"level": 4}, {"level": True},
                                     {"level": "2"}, {"level": 2.0}, {}])
def test_test_blast_bad_level(rig, payload):
    assert rig.client.post("/api/test-blast", json=payload).status_code == 422
    assert rig.sound.cmds == []


class FailingAplaySound(FakeSound):
    """Like SoundAction: handle() returns at once; aplay fails a moment later
    and the writer thread sets last_error."""

    def __init__(self, delay=0.05, err=None):
        super().__init__()
        self.last_error = None
        self.delay = delay
        self.err = err

    def handle(self, cmd):
        super().handle(cmd)
        if self.err is not None:
            threading.Timer(self.delay, lambda: setattr(self, "last_error", self.err)).start()


APLAY_ERR = {"error": "aplay exit 1: aplay: main:850: audio open error: No such device",
             "device": "sysdefault", "returncode": 1}


def test_test_blast_early_aplay_failure_is_503(rig):
    rig.state.sound = FailingAplaySound(err=APLAY_ERR)
    r = rig.client.post("/api/test-blast", json={"level": 2})
    assert r.status_code == 503
    assert "audio open error" in r.json()["detail"]
    assert "test_blast" not in rig.kinds()


def test_test_blast_ok_when_aplay_keeps_running(rig):
    rig.state.sound = FailingAplaySound(err=None)
    t0 = time.monotonic()
    r = rig.client.post("/api/test-blast", json={"level": 2})
    assert r.status_code == 200
    assert time.monotonic() - t0 < 1.0
    assert "test_blast" in rig.kinds()


def test_test_blast_ignores_stale_earlier_error(rig):
    snd = FailingAplaySound(err=None)
    snd.last_error = dict(APLAY_ERR)       # from an earlier blast
    rig.state.sound = snd
    assert rig.client.post("/api/test-blast", json={"level": 1}).status_code == 200


def test_status_includes_sound_error(rig):
    assert rig.client.get("/api/status").json()["sound_error"] is None
    snd = FailingAplaySound()
    snd.last_error = APLAY_ERR
    rig.state.sound = snd
    assert rig.client.get("/api/status").json()["sound_error"] == APLAY_ERR


def test_status_not_stale_when_fresh(rig):
    assert rig.client.get("/api/status").json()["stale"] is False


def test_status_stale_when_pipeline_stops_updating(rig):
    rig.pipe.updated_at = time.monotonic() - 2.5
    assert rig.client.get("/api/status").json()["stale"] is True


def test_status_stale_before_first_tick(rig):
    rig.pipe.updated_at = None
    assert rig.client.get("/api/status").json()["stale"] is True


def test_test_blast_sound_failure(rig):
    rig.sound.fail = True
    r = rig.client.post("/api/test-blast", json={"level": 1})
    assert r.status_code == 503
    assert rig.events.list(kinds={"error"})[0]["detail"]["source"] == "sound"


# -- events ----------------------------------------------------------------------

def test_events_filter_by_kinds(rig):
    rig.events.log("intrusion", {})
    rig.events.log("armed", {})
    rig.events.log("round", {"sequence": 1, "round": 1})
    body = rig.client.get("/api/events?kinds=intrusion").json()
    assert [e["kind"] for e in body] == ["intrusion"]
    body = rig.client.get("/api/events?kinds=intrusion, round").json()
    assert [e["kind"] for e in body] == ["round", "intrusion"]
    body = rig.client.get("/api/events").json()
    assert [e["kind"] for e in body] == ["round", "armed", "intrusion"]


def test_events_paging_and_limit_clamp(rig, monkeypatch):
    ids = [rig.events.log("armed", {}) for _ in range(5)]
    body = rig.client.get(f"/api/events?limit=2&before_id={ids[3]}").json()
    assert [e["id"] for e in body] == [ids[2], ids[1]]
    assert len(rig.client.get("/api/events?limit=0").json()) == 1
    seen = []
    real = rig.events.list

    def spy(limit=100, before_id=None, kinds=None):
        seen.append(limit)
        return real(limit=limit, before_id=before_id, kinds=kinds)

    monkeypatch.setattr(rig.events, "list", spy)
    rig.client.get("/api/events?limit=100000")
    rig.client.get("/api/events?limit=-5")
    assert seen == [500, 1]
    assert rig.client.get("/api/events?limit=abc").status_code == 422


def test_snapshot(rig):
    with_snap = rig.events.log("intrusion", {}, snapshot_jpeg=JPEG_B)
    without = rig.events.log("armed", {})
    r = rig.client.get(f"/api/events/{with_snap}/snapshot.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == JPEG_B
    assert rig.client.get(f"/api/events/{without}/snapshot.jpg").status_code == 404
    assert rig.client.get("/api/events/9999/snapshot.jpg").status_code == 404


# -- status, frames, UI -------------------------------------------------------------

def test_status(rig):
    body = rig.client.get("/api/status").json()
    assert body["people"] == 1 and body["seconds_until_armed"] == 42.0
    assert body["zone_set"] is True


def test_frame_is_unannotated(rig):
    r = rig.client.get("/frame.jpg")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content == RAW
    assert "no-store" in r.headers["cache-control"]


def test_frame_unavailable(rig):
    rig.pipe.raw = None
    assert rig.client.get("/frame.jpg").status_code == 503


def test_index_served(rig):
    r = rig.client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "/stream.mjpg" in r.text
    for asset in ("/static/app.js", "/static/style.css"):
        assert rig.client.get(asset).status_code == 200


def test_stream_ends_when_app_stops(rig):
    rig.state.stop.set()
    with rig.client.stream("GET", "/stream.mjpg") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("multipart/x-mixed-replace")
        assert b"".join(r.iter_bytes()) == b""


# -- MJPEG generator ------------------------------------------------------------------

def _collect(state, disconnected=lambda: False, until=1.0, max_parts=None):
    async def is_disconnected():
        return disconnected()

    async def run():
        parts = []
        t0 = time.monotonic()
        async for part in mjpeg_frames(state, is_disconnected, interval=0.01):
            parts.append(part)
            if max_parts and len(parts) >= max_parts:
                break
            if time.monotonic() - t0 > until:
                break
        return parts

    return asyncio.run(asyncio.wait_for(run(), timeout=5))


def test_mjpeg_sends_only_new_frames(rig):
    stop_at = time.monotonic() + 0.2

    def disconnected():
        return time.monotonic() > stop_at

    parts = _collect(rig.state, disconnected)
    assert len(parts) == 1                      # frame A once, then nothing new
    assert parts[0].startswith(b"--frame\r\nContent-Type: image/jpeg\r\n")
    assert parts[0].endswith(JPEG_A + b"\r\n")
    # it polled at the interval instead of spinning
    assert 5 <= rig.pipe.jpeg_calls <= 40


def test_mjpeg_new_frame_is_sent(rig):
    calls = {"n": 0}

    def disconnected():
        calls["n"] += 1
        if calls["n"] == 3:
            rig.pipe.jpeg = JPEG_B
        return calls["n"] > 6

    parts = _collect(rig.state, disconnected)
    assert [p.endswith(JPEG_A + b"\r\n") for p in parts] == [True, False]
    assert parts[1].endswith(JPEG_B + b"\r\n")


def test_mjpeg_stops_on_shutdown(rig):
    def disconnected():
        rig.state.stop.set()
        return False

    assert len(_collect(rig.state, disconnected)) <= 1


def test_mjpeg_waits_for_first_frame(rig):
    rig.pipe.jpeg = None
    stop_at = time.monotonic() + 0.1
    parts = _collect(rig.state, lambda: time.monotonic() > stop_at)
    assert parts == []


def test_settings_getter_tracks_replacements(rig):
    getter = rig.state.get_settings
    rig.client.put("/api/settings", json={"tone_hz": 3000})
    assert getter().tone_hz == 3000


def test_settings_event_detail_is_json(rig):
    rig.client.put("/api/settings", json={"levels": {"1": 0.3}})
    ev = rig.events.list(kinds={"settings"})[0]
    json.dumps(ev["detail"])
    assert ev["detail"]["changed"]["levels"] == {"1": 0.3, "2": 0.7, "3": 1.0}


def test_make_server_exit_ends_streams(rig):
    import signal

    from dogbot.web.app import make_server

    server = make_server(rig.state, "127.0.0.1", 0)
    assert not rig.state.stop.is_set()
    server.handle_exit(signal.SIGINT, None)
    assert rig.state.stop.is_set() and server.should_exit


def test_bad_settings_file_at_startup_falls_back_and_guard_can_be_disabled(tmp_path):
    path = tmp_path / "config" / "settings.json"
    path.parent.mkdir()
    path.write_text('{"rearm_grace_s": 0}')
    events = EventStore(tmp_path / "events.db", tmp_path / "snaps")
    state = AppState.from_path(path, events=events, sound=FakeSound())
    state.pipeline = FakePipeline(state.get_settings)
    assert state.get_settings() == Settings()
    assert state.load_error and "rearm_grace_s" in state.load_error
    err = events.list(kinds={"error"})
    assert len(err) == 1 and err[0]["detail"]["source"] == "settings"
    assert "rearm_grace_s" in err[0]["detail"]["reason"]
    client = TestClient(create_app(state))
    r = client.post("/api/enabled", json={"enabled": False})
    assert r.status_code == 200
    assert load_settings(path).enabled is False
    assert list(path.parent.glob("settings.json.invalid-*"))


def test_good_settings_file_at_startup_logs_nothing(rig):
    state = AppState.from_path(rig.path, events=rig.events, sound=rig.sound)
    assert state.load_error is None
    assert state.get_settings().zone == ZONE
    assert "error" not in rig.kinds()
