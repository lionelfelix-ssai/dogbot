"""Service entry point (`python -m dogbot`): wiring and shutdown, with fakes
for the camera, Hailo, sound and pipeline."""
import http.client
import signal
import socket
import threading
import time
from pathlib import Path

import pytest

import dogbot.__main__ as svc
from dogbot.config import Settings, save_settings
from dogbot.events import EventStore
from dogbot.rules import PlayBlasts
from dogbot.web.app import AppState, make_server

JPEG = b"\xff\xd8 frame \xff\xd9"


class Recorder:
    def __init__(self):
        self.calls = []

    def add(self, name):
        self.calls.append(name)


class FakeSource:
    def __init__(self, rec, lens_position=None):
        self.rec = rec
        self.lens_position = lens_position
        rec.add("source.open")

    def capture(self):
        raise AssertionError("not used")

    def close(self):
        self.rec.add("source.close")


class FakeDetector:
    def __init__(self, rec, hef):
        self.rec = rec
        self.hef = hef
        rec.add("detector.open")

    def close(self):
        self.rec.add("detector.close")


class FakeSound:
    def __init__(self, rec, settings_getter, on_error=None):
        self.rec = rec
        self.settings_getter = settings_getter
        self.on_error = on_error
        self.handled = []
        self.last_error = None

    def handle(self, cmd):
        self.handled.append(cmd)

    def stop(self):
        self.rec.add("sound.stop")


class FakePipeline:
    """Stands in for dogbot.pipeline.Pipeline: records its wiring and runs
    until the stop event is set."""

    def __init__(self, rec, settings_getter, events, sound, frame_source, detector):
        self.rec = rec
        self.settings_getter = settings_getter
        self.events = events
        self.sound = sound
        self.source = frame_source
        self.detector = detector
        self.started = threading.Event()
        self.frame = 0

    def run_forever(self, stop_event):
        self.started.set()
        while not stop_event.is_set():
            self.frame += 1
            time.sleep(0.01)
        self.rec.add("pipeline.stopped")

    def latest_jpeg(self):
        return JPEG + str(self.frame).encode()

    def raw_jpeg(self):
        return JPEG

    def status(self):
        return {"armed": False, "fault": False, "fps": 30.0, "updated_at": time.monotonic()}


def factories(rec):
    return dict(
        make_source=lambda lens_position=None: FakeSource(rec, lens_position),
        make_detector=lambda hef: FakeDetector(rec, hef),
        make_sound=lambda getter, on_error=None: FakeSound(rec, getter, on_error),
        make_pipeline=lambda *a, **k: FakePipeline(rec, *a, **k),
    )


def args_for(tmp_path, *extra):
    return svc.parse_args(["--config", str(tmp_path / "config" / "settings.json"),
                           "--data-dir", str(tmp_path / "data"), *extra])


# -- argument defaults ------------------------------------------------------

def test_defaults(monkeypatch):
    monkeypatch.delenv("DOGBOT_CONFIG", raising=False)
    a = svc.parse_args([])
    assert a.port == 8080
    assert a.host == "0.0.0.0"
    assert a.hef == "/home/admin/models/yolo26s.hef"
    assert Path(a.config) == Path("config/settings.json")
    assert Path(a.data_dir) == Path("data")


def test_config_default_follows_env(monkeypatch, tmp_path):
    monkeypatch.setenv("DOGBOT_CONFIG", str(tmp_path / "s.json"))
    assert Path(svc.parse_args([]).config) == tmp_path / "s.json"


# -- wiring -------------------------------------------------------------------

def test_build_wires_real_parts_with_live_settings(tmp_path):
    rec = Recorder()
    save_settings(Settings(dog_conf=0.6), tmp_path / "config" / "settings.json")
    rt = svc.build(args_for(tmp_path, "--hef", "/x/model.hef", "--lens-position", "2.5"),
                   **factories(rec))

    assert isinstance(rt.state, AppState)
    assert isinstance(rt.events, EventStore)
    assert (tmp_path / "data" / "events.db").exists()
    assert (tmp_path / "data" / "snapshots").is_dir()
    assert rt.state.settings.dog_conf == 0.6
    assert rt.state.events is rt.events
    assert rt.state.sound is rt.sound
    assert rt.state.pipeline is rt.pipeline

    p = rt.pipeline
    assert p.events is rt.events and p.sound is rt.sound
    assert p.source is rt.source and p.detector is rt.detector
    assert rt.detector.hef == "/x/model.hef"
    assert rt.source.lens_position == 2.5

    # pipeline and sound read the *live* settings: a UI change is seen at once
    assert p.settings_getter == rt.state.get_settings
    assert rt.sound.settings_getter == rt.state.get_settings
    rt.state.update_settings(lambda s: Settings(**{**s.__dict__, "tone_hz": 3000.0}))
    assert p.settings_getter().tone_hz == 3000.0
    assert rt.sound.settings_getter().tone_hz == 3000.0


def test_sound_errors_become_error_events(tmp_path, caplog):
    rt = svc.build(args_for(tmp_path), **factories(Recorder()))
    assert rt.sound.on_error is not None
    rt.sound.on_error({"error": "aplay exit 1: no device", "device": "sysdefault",
                       "returncode": 1})
    err = rt.events.list(kinds={"error"})
    assert len(err) == 1
    assert err[0]["detail"] == {"source": "sound", "error": "aplay exit 1: no device",
                                "device": "sysdefault", "returncode": 1}
    assert "aplay exit 1" in caplog.text


def test_real_sound_gets_on_error(tmp_path):
    from dogbot.actions.sound import SoundAction
    cb = lambda d: None  # noqa: E731
    snd = svc._real_sound(lambda: None, on_error=cb)
    assert isinstance(snd, SoundAction) and snd._on_error is cb


def test_build_logs_startup_event(tmp_path):
    rt = svc.build(args_for(tmp_path, "--port", "9000"), **factories(Recorder()))
    kinds = [e["kind"] for e in rt.events.list()]
    assert kinds == ["startup"]
    detail = rt.events.list()[0]["detail"]
    assert detail["port"] == 9000
    assert detail["hef"] == "/home/admin/models/yolo26s.hef"


def test_build_logs_rejected_settings_as_error(tmp_path):
    cfg = tmp_path / "config" / "settings.json"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("{not json")
    rt = svc.build(args_for(tmp_path), **factories(Recorder()))
    kinds = [e["kind"] for e in rt.events.list()]
    assert "error" in kinds and "startup" in kinds
    assert rt.state.load_error is not None


def test_build_closes_camera_if_hailo_fails(tmp_path):
    rec = Recorder()
    f = factories(rec)

    def broken(hef):
        raise RuntimeError("no /dev/hailo0")

    f["make_detector"] = broken
    with pytest.raises(RuntimeError, match="hailo0"):
        svc.build(args_for(tmp_path), **f)
    assert rec.calls == ["source.open", "source.close"]


# -- shutdown -----------------------------------------------------------------

def test_shutdown_order(tmp_path):
    rec = Recorder()
    rt = svc.build(args_for(tmp_path), **factories(rec))
    t = threading.Thread(target=rt.pipeline.run_forever, args=(rt.state.stop,))
    t.start()
    rt.pipeline.started.wait(2)
    rec.calls.clear()
    svc.shutdown(rt, t)
    assert not t.is_alive()
    assert rt.state.stop.is_set()
    assert rec.calls[0] == "sound.stop"
    assert rec.calls.index("pipeline.stopped") < rec.calls.index("detector.close")
    assert rec.calls[-2:] == ["detector.close", "source.close"]


def test_stop_handler_sets_state_stop_and_exits(tmp_path):
    rt = svc.build(args_for(tmp_path), **factories(Recorder()))
    holder = svc._Holder()
    holder.runtime = rt
    with pytest.raises(SystemExit) as exc:
        svc._make_stop_handler(holder)(signal.SIGTERM, None)
    assert exc.value.code == 0
    assert rt.state.stop.is_set()


def test_stop_handler_ignores_repeat_signal(tmp_path):
    holder = svc._Holder()
    handler = svc._make_stop_handler(holder)
    with pytest.raises(SystemExit):
        handler(signal.SIGTERM, None)
    handler(signal.SIGTERM, None)  # second signal during cleanup: no exception


class SignalServer:
    """Fake server: once running, SIGTERM arrives (as from systemctl stop)."""

    def __init__(self, state, host, port):
        self.state = state

    def run(self):
        assert self.state.pipeline.started.wait(2)
        signal.raise_signal(signal.SIGTERM)
        time.sleep(5)  # the handler must not let us get here
        raise AssertionError("SIGTERM did not stop the service")


def test_main_sigterm_cleans_up_and_restores_handler(tmp_path):
    rec = Recorder()
    before = signal.getsignal(signal.SIGTERM)
    argv = ["--config", str(tmp_path / "settings.json"), "--data-dir", str(tmp_path / "data")]
    with pytest.raises(SystemExit) as exc:
        svc.main(argv, build_fn=lambda a: svc.build(a, **factories(rec)),
                 server_factory=SignalServer)
    assert exc.value.code == 0
    assert "pipeline.stopped" in rec.calls
    assert rec.calls[-2:] == ["detector.close", "source.close"]
    assert "sound.stop" in rec.calls
    assert signal.getsignal(signal.SIGTERM) == before


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_sigterm_with_open_mjpeg_stream_exits_fast(tmp_path):
    """R13: real uvicorn server, a client holding /stream.mjpg open, then
    SIGTERM. The service must exit promptly, not wait for the client."""
    rec = Recorder()
    port = _free_port()
    got_frame = threading.Event()
    errors = []
    sent = []

    def client():
        conn = None
        try:
            for _ in range(100):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                try:
                    conn.request("GET", "/stream.mjpg")
                    break
                except OSError:
                    conn.close()
                    time.sleep(0.05)
            resp = conn.getresponse()
            assert resp.status == 200
            assert resp.read(20).startswith(b"--frame")
            got_frame.set()
            while resp.read(1024):   # hold the stream open until the server ends it
                pass
            resp.close()
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)
        finally:
            if conn is not None:
                conn.close()

    def killer():
        if got_frame.wait(10):
            sent.append(time.monotonic())
            signal.raise_signal(signal.SIGTERM)

    c = threading.Thread(target=client, daemon=True)
    c.start()
    k = threading.Thread(target=killer, daemon=True)
    k.start()
    argv = ["--config", str(tmp_path / "settings.json"), "--data-dir", str(tmp_path / "data"),
            "--host", "127.0.0.1", "--port", str(port)]
    with pytest.raises(SystemExit):
        svc.main(argv, build_fn=lambda a: svc.build(a, **factories(rec)),
                 server_factory=make_server)
    elapsed = time.monotonic() - sent[0]
    k.join(1)
    c.join(2)
    assert got_frame.is_set(), errors
    assert not errors, errors
    assert rec.calls[-2:] == ["detector.close", "source.close"]
    assert elapsed < 5, f"shutdown took {elapsed:.1f}s with a stream open"


# -- I-1: watchdog ------------------------------------------------------------

class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class WDRig:
    def __init__(self, tmp_path):
        self.rec = Recorder()
        self.status = {"updated_at": None}
        self.clock = Clock()
        self.exits = []
        self.stopping = False
        self.events = EventStore(tmp_path / "events.db", tmp_path / "snaps")
        self.sound = FakeSound(self.rec, None)
        self.wd = svc.Watchdog(lambda: dict(self.status), self.sound, self.events,
                               clock=self.clock, exit_fn=self.exits.append,
                               is_stopping=lambda: self.stopping)

    def stalled_events(self):
        return [e["detail"] for e in self.events.list(kinds={"error"})]


def test_watchdog_not_armed_before_first_tick(tmp_path):
    r = WDRig(tmp_path)
    r.clock.t += 600                      # camera/Hailo slow to open: fine
    assert r.wd.check() is False
    assert r.exits == [] and r.rec.calls == []


def test_watchdog_quiet_while_ticking(tmp_path):
    r = WDRig(tmp_path)
    for _ in range(20):
        r.clock.t += 0.5
        r.status["updated_at"] = r.clock.t - 0.1
        assert r.wd.check() is False
    assert r.exits == []


def test_watchdog_fires_on_stall(tmp_path):
    r = WDRig(tmp_path)
    r.status["updated_at"] = r.clock.t
    r.clock.t += 2.9
    assert r.wd.check() is False
    r.clock.t += 0.2                      # 3.1 s without an update
    assert r.wd.check() is True
    assert r.rec.calls == ["sound.stop"]
    assert r.stalled_events() == [{"source": "pipeline", "error": "stalled"}]
    assert r.exits == [1]


def test_watchdog_never_fires_while_shutting_down(tmp_path):
    r = WDRig(tmp_path)
    r.status["updated_at"] = r.clock.t
    r.clock.t += 10
    r.stopping = True
    assert r.wd.check() is False
    assert r.exits == []


def test_watchdog_exits_even_if_stop_and_log_fail(tmp_path):
    r = WDRig(tmp_path)

    def boom(*a, **k):
        raise OSError("nope")

    r.sound.stop = boom
    r.events.log = boom
    r.status["updated_at"] = r.clock.t
    r.clock.t += 5
    assert r.wd.check() is True
    assert r.exits == [1]


def test_watchdog_thread_ends_with_stop_event(tmp_path):
    r = WDRig(tmp_path)
    stop = threading.Event()
    t = threading.Thread(target=r.wd.run, args=(stop, 0.01))
    t.start()
    time.sleep(0.05)
    stop.set()
    t.join(1)
    assert not t.is_alive() and r.exits == []


def test_main_starts_watchdog(tmp_path, monkeypatch):
    started = []
    real = svc.Watchdog

    class Spy(real):
        def run(self, stop, interval=svc.WATCHDOG_INTERVAL_S):
            started.append(self)
            super().run(stop, interval)

    monkeypatch.setattr(svc, "Watchdog", Spy)
    rec = Recorder()
    argv = ["--config", str(tmp_path / "settings.json"), "--data-dir", str(tmp_path / "data")]
    with pytest.raises(SystemExit):
        svc.main(argv, build_fn=lambda a: svc.build(a, **factories(rec)),
                 server_factory=SignalServer)
    assert len(started) == 1
