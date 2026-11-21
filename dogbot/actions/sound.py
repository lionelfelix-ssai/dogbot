"""Sound action: escalating beeps piped to ALSA via aplay.

handle() never blocks the pipeline: samples are written to aplay's stdin from
a daemon thread. stop() terminates aplay synchronously, so sound ends within
the call regardless of how backed up the pipe is.
"""
from __future__ import annotations

import subprocess
import threading
from typing import Callable

import numpy as np

from dogbot.config import Settings
from dogbot.rules import PlayBlasts, StopSound

RATE = 48000
FADE_S = 0.010
CHUNK = 4096  # bytes per write; keeps the writer responsive to stop
THREAD_NAME = "dogbot-sound-writer"


def synth_blasts(count: int, gain: float, tone_hz: float, blast_s: float,
                 spacing_s: float, rate: int = RATE) -> np.ndarray:
    """`count` sine blasts separated by silence, as mono int16."""
    n = int(round(blast_s * rate))
    gap = int(round(spacing_s * rate))
    t = np.arange(n) / rate
    blast = np.sin(2 * np.pi * tone_hz * t)
    fade = min(int(FADE_S * rate), n // 2)
    if fade > 1:
        ramp = np.linspace(0.0, 1.0, fade)
        blast[:fade] *= ramp
        blast[n - fade:] *= ramp[::-1]
    blast_i = np.round(blast * gain * 32767).astype(np.int16)
    parts = []
    for i in range(count):
        if i:
            parts.append(np.zeros(gap, dtype=np.int16))
        parts.append(blast_i)
    if not parts:
        return np.zeros(0, dtype=np.int16)
    return np.concatenate(parts)


class _Playback:
    def __init__(self, proc, data: bytes):
        self.proc = proc
        self.data = data
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._run, name=THREAD_NAME,
                                       daemon=True)

    def _run(self) -> None:
        stdin = self.proc.stdin
        try:
            for i in range(0, len(self.data), CHUNK):
                if self.cancel.is_set():
                    return
                stdin.write(self.data[i:i + CHUNK])
            stdin.flush()
        except (BrokenPipeError, OSError, ValueError):
            pass  # aplay died or was terminated
        finally:
            try:
                stdin.close()
            except (BrokenPipeError, OSError, ValueError):
                pass
            if not self.cancel.is_set():
                try:
                    self.proc.wait()  # let aplay drain, then reap it
                except Exception:
                    pass


class SoundAction:
    def __init__(self, settings_getter: Callable[[], Settings],
                 runner=subprocess.Popen):
        self._settings = settings_getter
        self._runner = runner
        self._lock = threading.Lock()
        self._current: _Playback | None = None

    def handle(self, cmd) -> None:
        if isinstance(cmd, StopSound):
            self.stop()
        elif isinstance(cmd, PlayBlasts):
            self._play(cmd)

    def _play(self, cmd: PlayBlasts) -> None:
        s = self._settings()
        if cmd.level not in s.levels:
            raise ValueError(f"unknown sound level {cmd.level}")
        data = synth_blasts(cmd.count, s.levels[cmd.level], s.tone_hz,
                            s.blast_s, s.blast_spacing_s).tobytes()
        args = ["aplay", "-q", "-D", s.audio_device, "-f", "S16_LE",
                "-r", str(RATE), "-c", "1"]
        with self._lock:
            self._stop_locked()
            proc = self._runner(args, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
            pb = _Playback(proc, data)
            self._current = pb
            pb.thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_locked()

    def _stop_locked(self) -> None:
        pb, self._current = self._current, None
        if pb is None:
            return
        pb.cancel.set()
        try:
            pb.proc.terminate()
        except (ProcessLookupError, OSError):
            pass
        try:
            pb.proc.wait(timeout=0.2)
        except subprocess.TimeoutExpired:
            try:
                pb.proc.kill()
            except (ProcessLookupError, OSError):
                pass
        except Exception:
            pass
        # Writer may be blocked in write(); the dead process breaks the pipe
        # and it exits on its own (daemon). Joining would delay stop().
