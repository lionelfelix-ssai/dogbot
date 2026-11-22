import io
import subprocess
import threading
import time

import numpy as np
import pytest

from dogbot.actions.sound import SoundAction, synth_blasts
from dogbot.config import Settings
from dogbot.rules import PlayBlasts, StopSound

RATE = 48000


def test_length_peak_and_spacing():
    a = synth_blasts(3, 1.0, 4000, 0.5, 0.4)
    assert a.dtype == np.int16
    assert len(a) == round((3 * 0.5 + 2 * 0.4) * RATE)
    assert abs(int(np.abs(a).max()) - 32767) <= 0.02 * 32767
    # spacing regions are exactly zero
    s0, s1 = int(0.5 * RATE), int(0.9 * RATE)
    assert np.all(a[s0:s1] == 0)
    s2, s3 = int(1.4 * RATE), int(1.8 * RATE)
    assert np.all(a[s2:s3] == 0)


def test_gain_scales_peak():
    a = synth_blasts(1, 0.4, 4000, 0.5, 0.4)
    assert abs(int(np.abs(a).max()) - 0.4 * 32767) <= 0.02 * 32767


def test_fades_in_and_out_of_each_blast():
    a = synth_blasts(2, 1.0, 4000, 0.5, 0.4)
    n = int(0.5 * RATE)
    start2 = int(0.9 * RATE)
    assert a[0] == 0 and a[n - 1] == 0
    assert a[start2] == 0 and a[start2 + n - 1] == 0
    fade = int(0.010 * RATE)
    assert np.abs(a[:fade]).max() < np.abs(a[fade:2 * fade]).max()
    assert np.abs(a[n - fade:n]).max() < np.abs(a[n - 2 * fade:n - fade]).max()


class Sink:
    def __init__(self, block=None):
        self.data = bytearray()
        self.closed = False
        self.block = block

    def write(self, b):
        if self.block is not None:
            self.block.wait(5)
        if self.closed:
            raise BrokenPipeError
        self.data += b
        return len(b)

    def flush(self):
        pass

    def close(self):
        self.closed = True


class FakeProc:
    def __init__(self, args, kwargs, block=None, exit_code=0, stderr=b"",
                 hang=False):
        self.args = args
        self.kwargs = kwargs
        self.stdin = Sink(block)
        self.stderr = io.BytesIO(stderr)
        self.terminated = False
        self.returncode = None
        self.exit_code = exit_code
        self.hang = hang
        self.wait_timeouts = []

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stdin.closed = True

    def kill(self):
        self.terminate()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.hang and self.returncode is None:
            raise subprocess.TimeoutExpired("aplay", timeout)
        if self.returncode is None:
            self.returncode = self.exit_code
        return self.returncode


class Runner:
    def __init__(self, block=None, **proc_kwargs):
        self.procs = []
        self.block = block
        self.proc_kwargs = proc_kwargs

    def __call__(self, args, **kwargs):
        p = FakeProc(args, kwargs, self.block, **self.proc_kwargs)
        self.procs.append(p)
        return p


def make(settings=None, block=None, on_error=None, **proc_kwargs):
    s = settings or Settings(audio_device="plughw:CARD=X")
    r = Runner(block, **proc_kwargs)
    return SoundAction(lambda: s, runner=r, on_error=on_error), r, s


def writers_gone():
    return wait_for(lambda: not any(t.name == "dogbot-sound-writer"
                                    for t in threading.enumerate()))


def wait_for(cond, t=2.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return False


def test_play_launches_aplay_and_streams_gain():
    act, r, s = make()
    act.handle(PlayBlasts(2, 2))
    assert len(r.procs) == 1
    p = r.procs[0]
    assert p.args == ["aplay", "-q", "-D", "plughw:CARD=X", "-f", "S16_LE",
                      "-r", "48000", "-c", "1"]
    assert p.kwargs["stdin"] == subprocess.PIPE
    assert p.kwargs["stdout"] == subprocess.DEVNULL
    assert p.kwargs["stderr"] == subprocess.PIPE
    expected = synth_blasts(2, s.levels[2], s.tone_hz, s.blast_s,
                            s.blast_spacing_s)
    assert wait_for(lambda: bytes(p.stdin.data) == expected.tobytes())
    act.stop()


def test_handle_does_not_block_on_full_pipe():
    gate = threading.Event()
    act, r, _ = make(block=gate)
    t0 = time.monotonic()
    act.handle(PlayBlasts(1, 1))
    assert time.monotonic() - t0 < 0.5
    gate.set()
    act.stop()


def test_stop_terminates_immediately_even_when_writer_blocked():
    gate = threading.Event()
    act, r, _ = make(block=gate)
    act.handle(PlayBlasts(3, 3))
    t0 = time.monotonic()
    act.handle(StopSound())
    assert r.procs[0].terminated
    gate.set()
    assert time.monotonic() - t0 < 0.5
    assert wait_for(lambda: not any(t.name == "dogbot-sound-writer"
                                    for t in threading.enumerate()))


def test_new_play_kills_previous():
    act, r, _ = make()
    act.handle(PlayBlasts(1, 1))
    act.handle(PlayBlasts(1, 2))
    assert r.procs[0].terminated
    assert not r.procs[1].terminated
    act.stop()
    assert r.procs[1].terminated


def test_stop_idempotent_and_safe_on_dead_process():
    act, r, _ = make()
    act.stop()  # nothing playing
    act.handle(PlayBlasts(1, 1))
    p = r.procs[0]

    def boom():
        raise ProcessLookupError

    p.terminate = boom
    act.stop()
    act.stop()


def test_unknown_level_raises_and_plays_nothing():
    act, r, _ = make()
    with pytest.raises(ValueError):
        act.handle(PlayBlasts(1, 9))
    assert r.procs == []


def test_ignores_other_commands():
    act, r, _ = make()
    act.handle("something else")
    assert r.procs == []


def test_settings_read_live():
    s = Settings()
    r = Runner()
    act = SoundAction(lambda: s, runner=r)
    s.audio_device = "hw:1,0"
    act.handle(PlayBlasts(1, 1))
    assert r.procs[0].args[3] == "hw:1,0"
    act.stop()


# -- I-3 / R16: aplay failures are reported, not swallowed --------------------

STDERR = b"aplay: main:850: audio open error: No such file or directory\n"


def test_aplay_nonzero_exit_reports_once_with_stderr():
    errors = []
    act, r, s = make(on_error=errors.append, exit_code=1, stderr=STDERR)
    act.handle(PlayBlasts(1, 1))
    assert writers_gone()
    assert len(errors) == 1
    e = errors[0]
    assert e["returncode"] == 1
    assert e["device"] == "plughw:CARD=X"
    assert e["error"].startswith("aplay exit 1: ")
    assert "audio open error" in e["error"]
    assert act.last_error == e
    # waited with a timeout scaled to the clip length
    data_s = len(synth_blasts(1, 0.4, s.tone_hz, s.blast_s, s.blast_spacing_s)) / RATE
    assert r.procs[0].wait_timeouts[0] == pytest.approx(data_s + 5)


def test_stderr_truncated_to_300_chars():
    errors = []
    act, r, _ = make(on_error=errors.append, exit_code=2, stderr=b"x" * 1000)
    act.handle(PlayBlasts(1, 1))
    assert writers_gone()
    assert errors[0]["error"] == "aplay exit 2: " + "x" * 300


def test_clean_exit_reports_nothing_and_clears_last_error():
    errors = []
    act, r, _ = make(on_error=errors.append, exit_code=1, stderr=STDERR)
    act.handle(PlayBlasts(1, 1))
    assert writers_gone() and act.last_error is not None
    r.proc_kwargs = {}
    act.handle(PlayBlasts(1, 1))
    assert writers_gone()
    assert len(errors) == 1
    assert act.last_error is None


def test_stop_during_playback_reports_nothing():
    errors = []
    gate = threading.Event()
    act, r, _ = make(block=gate, on_error=errors.append)
    act.handle(PlayBlasts(3, 3))
    act.stop()                      # terminate -> returncode -15
    assert r.procs[0].terminated
    gate.set()
    assert writers_gone()
    assert errors == []
    assert act.last_error is None


def test_new_play_replacing_old_reports_nothing_for_the_old():
    errors = []
    gate = threading.Event()
    act, r, _ = make(block=gate, on_error=errors.append)
    act.handle(PlayBlasts(3, 3))
    act.handle(PlayBlasts(1, 1))
    gate.set()
    assert writers_gone()
    assert errors == []


def test_aplay_hang_is_killed_and_reported():
    errors = []
    act, r, _ = make(on_error=errors.append, hang=True)
    act.handle(PlayBlasts(1, 1))
    assert writers_gone()
    assert r.procs[0].terminated          # kill() in the fake marks it
    assert len(errors) == 1
    assert errors[0]["error"] == "aplay timed out"
    assert errors[0]["device"] == "plughw:CARD=X"


def test_on_error_that_raises_does_not_kill_the_writer(caplog):
    def boom(d):
        raise RuntimeError("event DB gone")

    act, r, _ = make(on_error=boom, exit_code=1, stderr=STDERR)
    act.handle(PlayBlasts(1, 1))      # -W error: an unhandled thread exception fails the test
    assert writers_gone()
    assert act.last_error is not None
    assert "on_error" in caplog.text
    act.handle(PlayBlasts(1, 2))      # still usable
    assert len(r.procs) == 2
    assert writers_gone()


def test_stop_fallback_reaps_after_kill():
    act, r, _ = make()
    act.handle(PlayBlasts(1, 1))
    p = r.procs[0]
    calls = []

    def slow_wait(timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("aplay", timeout)
        return -9

    p.wait = slow_wait
    p.terminate = lambda: None
    act.stop()
    assert len(calls) >= 2           # wait after kill() reaps the zombie
