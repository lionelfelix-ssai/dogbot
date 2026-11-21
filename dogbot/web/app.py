"""LAN web app: live view, on/off, couch zone, tuning, test blasts, event log.

No auth (LAN-only MVP). The Settings object is shared with the pipeline
thread: writers hold `AppState.lock`, build a new Settings, validate and save
it, and only then swap the reference. A Settings object is never mutated in
place, so `get_settings()` (what the pipeline and the sound action call every
tick) always sees a complete, validated object.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from dogbot.config import Settings, load_settings_checked, save_settings, validate_zone
from dogbot.rules import PlayBlasts

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
STREAM_FPS = 15
BOUNDARY = "frame"
EVENTS_LIMIT_MAX = 500
EVENTS_LIMIT_DEFAULT = 50
NO_STORE = {"Cache-Control": "no-store"}
# After a test blast starts, watch this long for aplay failing at once (no
# device, busy device): the owner then sees the error instead of "ok".
TEST_BLAST_CHECK_S = 0.3
# /api/status says stale when the pipeline hasn't updated for this long.
STALE_S = 2.0
TEST_BLAST_POLL_S = 0.02

# Fields PUT /api/settings may change. `zone` has its own route.
EDITABLE = frozenset(f.name for f in fields(Settings)) - {"zone"}
LEVEL_DICTS = ("levels", "level_db")


@dataclass
class AppState:
    settings: Settings
    settings_path: Path
    pipeline: Any
    events: Any
    sound: Any
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop: threading.Event = field(default_factory=threading.Event)
    load_error: str | None = None

    @classmethod
    def from_path(cls, settings_path: Path, events, sound=None, pipeline=None) -> "AppState":
        """Startup path: load settings with validation. A bad file falls back
        to defaults (see load_settings_checked) and logs an `error` event.
        Wire the pipeline and sound with `state.get_settings` afterwards."""
        settings, reason = load_settings_checked(settings_path)
        if reason is not None:
            events.log("error", {"source": "settings", "reason": reason})
        return cls(settings=settings, settings_path=Path(settings_path),
                   pipeline=pipeline, events=events, sound=sound, load_error=reason)

    def get_settings(self) -> Settings:
        """The current Settings object; pass this as `settings_getter`."""
        return self.settings

    def update_settings(self, build: Callable[[Settings], Settings]) -> tuple[Settings, Settings]:
        """Atomically: new = build(current); validate; save; swap.
        Raises ValueError (bad input) or OSError (save failed); either way
        the live settings are untouched. Returns (old, new)."""
        with self.lock:
            old = self.settings
            new = build(old)
            new.validate()
            save_settings(new, self.settings_path)
            self.settings = new
            return old, new

    def shutdown(self) -> None:
        """Ends open MJPEG streams. Call before stopping the server."""
        self.stop.set()


def settings_json(s: Settings) -> dict:
    d = asdict(s)
    for key in LEVEL_DICTS:
        d[key] = {str(k): v for k, v in d[key].items()}
    d["zone"] = [list(p) for p in s.zone]
    return d


def _merge_settings(current: Settings, payload: dict) -> Settings:
    """Partial update -> new Settings (unvalidated). ValueError on bad keys."""
    if "zone" in payload:
        raise ValueError("zone cannot be changed here; use PUT /api/zone")
    unknown = sorted(set(payload) - EDITABLE)
    if unknown:
        raise ValueError(f"unknown setting(s): {', '.join(unknown)}")
    changes: dict[str, Any] = {}
    for key, value in payload.items():
        if key in LEVEL_DICTS:
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be an object like {{\"1\": ...}}")
            merged = dict(getattr(current, key))
            for k, v in value.items():
                try:
                    level = int(k)
                except (TypeError, ValueError):
                    raise ValueError(f"{key} key {k!r} is not a level number") from None
                merged[level] = v
            changes[key] = merged
        else:
            changes[key] = value
    return replace(current, **changes)


def _changed(old: Settings, new: Settings) -> dict:
    new_json = settings_json(new)
    return {f.name: new_json[f.name] for f in fields(Settings)
            if getattr(old, f.name) != getattr(new, f.name)}


async def mjpeg_frames(state: AppState, is_disconnected: Callable[[], Awaitable[bool]],
                       interval: float = 1 / STREAM_FPS) -> AsyncIterator[bytes]:
    """multipart/x-mixed-replace parts of the pipeline's annotated JPEG.

    Sends a frame only when the pipeline has produced a new one, polls at
    `interval` (<= 15 fps) otherwise, and returns when the client goes away
    or the app shuts down."""
    last = None
    while not state.stop.is_set():
        if await is_disconnected():
            return
        jpeg = state.pipeline.latest_jpeg()
        if jpeg is not None and jpeg is not last:
            last = jpeg
            yield (b"--" + BOUNDARY.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                   + jpeg + b"\r\n")
        await asyncio.sleep(interval)


def _require_obj(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(422, "body must be a JSON object")
    return payload


def create_app(state: AppState) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app):
        yield
        state.shutdown()

    app = FastAPI(title="DogBot", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.dogbot = state
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    # -- pages and frames -----------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/stream.mjpg")
    async def stream(request: Request):
        return StreamingResponse(
            mjpeg_frames(state, request.is_disconnected),
            media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
            headers=NO_STORE)

    @app.get("/frame.jpg")
    def frame():
        jpeg = state.pipeline.raw_jpeg()
        if jpeg is None:
            raise HTTPException(503, "no camera frame yet", headers={"Retry-After": "1"})
        return Response(jpeg, media_type="image/jpeg", headers=NO_STORE)

    # -- status and settings ----------------------------------------------------

    @app.get("/api/status")
    def status():
        st = state.pipeline.status()
        s = state.get_settings()
        st["enabled"] = s.enabled     # authoritative now, not at the next tick
        st["zone_set"] = bool(s.zone)
        st["sound_error"] = getattr(state.sound, "last_error", None)
        updated_at = st.get("updated_at")
        st["stale"] = updated_at is None or time.monotonic() - updated_at > STALE_S
        return st

    @app.get("/api/settings")
    def get_settings():
        return settings_json(state.get_settings())

    def apply(build: Callable[[Settings], Settings]) -> tuple[Settings, Settings]:
        try:
            return state.update_settings(build)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        except OSError as e:
            log.exception("saving settings failed")
            raise HTTPException(500, f"could not save settings: {e}") from None

    def cut_sound() -> None:
        """Off means quiet now: also cut a running test blast, which the
        pipeline (it only stops what it started) would let finish."""
        if state.sound is None:
            return
        try:
            state.sound.stop()
        except Exception:
            log.exception("sound stop on disable failed")

    @app.put("/api/settings")
    def put_settings(payload: Any = Body(...)):
        payload = _require_obj(payload)
        old, new = apply(lambda cur: _merge_settings(cur, payload))
        if payload.get("enabled") is False:
            cut_sound()
        changed = _changed(old, new)
        if changed:
            state.events.log("settings", {"changed": changed})
            if "enabled" in changed:
                state.events.log("enabled" if new.enabled else "disabled", {"via": "settings"})
        return settings_json(new)

    @app.put("/api/zone")
    def put_zone(payload: Any = Body(...)):
        points = _require_obj(payload).get("points")
        try:
            validate_zone(points)
            if len(points) < 3:
                raise ValueError("zone needs at least 3 points")
        except (ValueError, TypeError) as e:
            raise HTTPException(422, str(e)) from None
        zone = [(p[0], p[1]) for p in points]
        _, new = apply(lambda cur: replace(cur, zone=zone))
        state.events.log("zone", {"points": len(zone)})
        return settings_json(new)

    @app.post("/api/enabled")
    def post_enabled(payload: Any = Body(...)):
        enabled = _require_obj(payload).get("enabled")
        if not isinstance(enabled, bool):
            raise HTTPException(422, "enabled must be true or false")
        old, _ = apply(lambda cur: replace(cur, enabled=enabled))
        if not enabled:
            cut_sound()
        if old.enabled != enabled:
            state.events.log("enabled" if enabled else "disabled", {})
        return status()

    @app.post("/api/test-blast")
    def test_blast(payload: Any = Body(...)):
        level = _require_obj(payload).get("level")
        if isinstance(level, bool) or not isinstance(level, int) or level not in (1, 2, 3):
            raise HTTPException(422, "level must be 1, 2 or 3")
        # Calibration: plays even when disarmed or disabled, by design.
        s = state.get_settings()
        sound = state.sound
        before = getattr(sound, "last_error", None)
        try:
            sound.handle(PlayBlasts(1, level))
        except Exception as e:
            log.exception("test blast failed")
            state.events.log("error", {"source": "sound", "command": "test_blast",
                                       "error": repr(e)})
            raise HTTPException(503, f"sound failed: {e}") from None
        if hasattr(sound, "last_error"):
            # aplay runs in the background; catch it failing right away. The
            # error event itself comes from SoundAction's on_error.
            deadline = time.monotonic() + TEST_BLAST_CHECK_S
            while time.monotonic() < deadline:
                err = sound.last_error
                if err is not None and err is not before:
                    raise HTTPException(503, f"sound failed: {err.get('error')}")
                time.sleep(TEST_BLAST_POLL_S)
        state.events.log("test_blast", {"level": level, "gain": s.levels[level]})
        return {"ok": True, "level": level, "gain": s.levels[level]}

    # -- events -----------------------------------------------------------------

    @app.get("/api/events")
    def list_events(limit: int = EVENTS_LIMIT_DEFAULT, before_id: int | None = None,
                    kinds: str | None = Query(None)):
        limit = max(1, min(EVENTS_LIMIT_MAX, limit))
        kind_set = {k.strip() for k in kinds.split(",") if k.strip()} if kinds else None
        return state.events.list(limit=limit, before_id=before_id, kinds=kind_set or None)

    @app.get("/api/events/{event_id}/snapshot.jpg")
    def snapshot(event_id: int):
        path = state.events.snapshot_path(event_id)
        if path is None or not Path(path).is_file():
            raise HTTPException(404, "no snapshot for this event")
        return FileResponse(path, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})

    return app


def make_server(state: AppState, host: str = "0.0.0.0", port: int = 8080):
    """A uvicorn Server whose signal handling also ends open MJPEG streams.

    uvicorn waits for in-flight responses before running lifespan shutdown,
    and an MJPEG response never finishes on its own, so Ctrl-C/SIGTERM must
    set `state.stop` first or shutdown hangs until the browser tab closes."""
    import uvicorn

    class _Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            state.shutdown()
            super().handle_exit(sig, frame)

    config = uvicorn.Config(create_app(state), host=host, port=port, log_level="info")
    return _Server(config)
