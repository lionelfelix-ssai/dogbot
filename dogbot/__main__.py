"""The DogBot service: `python -m dogbot [--port 8080] [--hef PATH]`.

Opens the camera and the Hailo device, starts the pipeline thread, and
serves the web app with uvicorn on 0.0.0.0:8080. Settings come from
`config/settings.json` (DOGBOT_CONFIG overrides); the event DB and snapshots
live under `--data-dir` (default `data/`).

Shutdown (SIGTERM from `systemctl stop`, or Ctrl-C): the server's signal
handler sets `state.stop` first, which ends open MJPEG streams (otherwise
uvicorn waits for them forever) and the pipeline loop. Then: stop sound,
join the pipeline thread, close Hailo and the camera.

uvicorn restores the previous signal handlers after a graceful shutdown and
re-raises the captured signal. We install our own SIGTERM handler first, so
that re-raise lands in `_make_stop_handler` (SystemExit(0)) and the cleanup
in `main`'s `finally` runs instead of the process dying on the default
handler. The same handler covers a SIGTERM during startup.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from dogbot.config import settings_path
from dogbot.events import EventStore
from dogbot.web.app import AppState, make_server

log = logging.getLogger("dogbot")

DEFAULT_HEF = "/home/admin/models/yolo26s.hef"
DEFAULT_PORT = 8080
JOIN_TIMEOUT_S = 3.0


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m dogbot",
                                 description="DogBot couch guard service")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--hef", default=DEFAULT_HEF, help="YOLO26 HEF for the Hailo-8L")
    ap.add_argument("--config", default=str(settings_path()),
                    help="settings file (default: $DOGBOT_CONFIG or config/settings.json)")
    ap.add_argument("--data-dir", default=os.environ.get("DOGBOT_DATA", "data"),
                    help="event DB and snapshots (default: $DOGBOT_DATA or data/)")
    ap.add_argument("--lens-position", type=float, default=None,
                    help="fixed focus (dioptres); default continuous autofocus")
    return ap.parse_args(argv)


@dataclass
class Runtime:
    state: AppState
    events: Any
    sound: Any
    source: Any
    detector: Any
    pipeline: Any


def _real_source(lens_position=None):
    from dogbot.pipeline import PicameraSource
    return PicameraSource(lens_position=lens_position)


def _real_detector(hef):
    from dogbot.hailo_runner import HailoDetector
    return HailoDetector(hef)


def _real_sound(settings_getter, on_error=None):
    from dogbot.actions.sound import SoundAction
    return SoundAction(settings_getter, on_error=on_error)


def _sound_error_logger(events) -> Callable[[dict], None]:
    """on_error for SoundAction: aplay failed on its own (R16)."""
    def on_error(d: dict) -> None:
        log.error("sound failed: %s (device %s)", d.get("error"), d.get("device"))
        events.log("error", {"source": "sound", **d})
    return on_error


def _real_pipeline(settings_getter, events, sound, frame_source, detector):
    from dogbot.pipeline import Pipeline
    return Pipeline(settings_getter, events, sound, frame_source, detector)


def build(args: argparse.Namespace, *, make_source=_real_source, make_detector=_real_detector,
          make_sound=_real_sound, make_pipeline=_real_pipeline) -> Runtime:
    """Construct everything; nothing runs yet. Raises if the camera or the
    Hailo device can't be opened (systemd restarts the service)."""
    data = Path(args.data_dir)
    events = EventStore(data / "events.db", data / "snapshots")
    state = AppState.from_path(Path(args.config), events)
    sound = make_sound(state.get_settings, on_error=_sound_error_logger(events))
    state.sound = sound

    source = make_source(lens_position=args.lens_position)
    try:
        detector = make_detector(args.hef)
    except BaseException:
        _quietly(source.close, "camera close")
        raise
    pipeline = make_pipeline(state.get_settings, events, sound, source, detector)
    state.pipeline = pipeline

    events.log("startup", {"port": args.port, "hef": args.hef, "pid": os.getpid(),
                           "config": str(args.config)})
    return Runtime(state, events, sound, source, detector, pipeline)


def _quietly(fn: Callable[[], Any], what: str) -> None:
    try:
        fn()
    except Exception:
        log.exception("%s failed", what)


def shutdown(rt: Runtime | None, thread: threading.Thread | None) -> None:
    """Stop sound, stop the pipeline, close Hailo and the camera."""
    if rt is None:
        return
    rt.state.shutdown()                       # ends MJPEG streams + pipeline loop
    _quietly(rt.sound.stop, "sound stop")
    if thread is not None:
        thread.join(JOIN_TIMEOUT_S)
        if thread.is_alive():
            log.warning("pipeline thread still running after %.0fs; closing devices anyway",
                        JOIN_TIMEOUT_S)
    _quietly(rt.sound.stop, "sound stop")     # in case the last tick started a blast
    _quietly(rt.detector.close, "Hailo close")
    _quietly(rt.source.close, "camera close")
    log.info("DogBot stopped")


class _Holder:
    runtime: Runtime | None = None
    stopping: bool = False


def _make_stop_handler(holder: _Holder):
    def handler(signum, frame):
        if holder.stopping:
            return                            # already shutting down
        holder.stopping = True
        if holder.runtime is not None:
            holder.runtime.state.shutdown()
        log.info("signal %s: shutting down", signum)
        raise SystemExit(0)
    return handler


def main(argv=None, *, build_fn: Callable[[argparse.Namespace], Runtime] = build,
         server_factory=make_server) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    holder = _Holder()
    previous = signal.signal(signal.SIGTERM, _make_stop_handler(holder))
    thread = None
    try:
        holder.runtime = rt = build_fn(args)
        thread = threading.Thread(target=rt.pipeline.run_forever, args=(rt.state.stop,),
                                  name="dogbot-pipeline", daemon=True)
        thread.start()
        log.info("DogBot listening on http://%s:%d", args.host, args.port)
        server_factory(rt.state, args.host, args.port).run()
    except KeyboardInterrupt:
        pass                                  # Ctrl-C: uvicorn re-raises it after shutdown
    finally:
        holder.stopping = True
        try:
            shutdown(holder.runtime, thread)
        finally:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    main()
