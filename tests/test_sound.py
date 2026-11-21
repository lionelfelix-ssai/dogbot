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
    def __init__(self, args, kwargs, block=None):
        self.args = args
        self.kwargs = kwargs
        self.stdin = Sink(block)
        self.terminated = False
        self.returncode = None

    def terminate(self):
        self.terminated = True
        self.returncode = -15
        self.stdin.closed = True

    def kill(self):
        self.terminate()

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class Runner:
    def __init__(self, block=None):
        self.procs = []
        self.block = block

    def __call__(self, args, **kwargs):
        p = FakeProc(args, kwargs, self.block)
        self.procs.append(p)
        return p


def make(settings=None, block=None):
    s = settings or Settings(audio_device="plughw:CARD=X")
    r = Runner(block)
    return SoundAction(lambda: s, runner=r), r, s


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
    assert p.kwargs["stderr"] == subprocess.DEVNULL
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
