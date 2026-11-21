"""Settings model for DogBot. Zones are normalized (0-1) points."""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

log = logging.getLogger(__name__)


def _levels() -> dict[int, float]:
    return {1: 0.4, 2: 0.7, 3: 1.0}


def _level_db() -> dict[int, float | None]:
    return {1: None, 2: None, 3: None}


@dataclass
class Settings:
    dog_conf: float = 0.50
    person_conf: float = 0.40
    dwell_s: float = 1.0
    rearm_grace_s: float = 60
    clear_s: float = 10
    round_gap_s: float = 5
    blast_s: float = 0.5
    blast_spacing_s: float = 0.4
    tone_hz: float = 4000
    levels: dict[int, float] = field(default_factory=_levels)
    level_db: dict[int, float | None] = field(default_factory=_level_db)
    sequences: int = 2
    enabled: bool = True
    audio_device: str = "default"
    zone: list[tuple[float, float]] = field(default_factory=list)

    def validate(self) -> None:
        """Raise ValueError unless every field has a sane type and range.

        bool is rejected where a number is expected, NaN/inf are rejected
        everywhere, and rearm_grace_s >= MIN_REARM_GRACE_S so a zero grace
        can never hide a person who just stepped out of view."""
        for name in ("dog_conf", "person_conf"):
            _check_range(name, getattr(self, name), 0.0, 1.0)
        for name in ("dwell_s", "clear_s", "round_gap_s", "blast_s",
                     "blast_spacing_s"):
            _check_range(name, getattr(self, name), 0.0, None)
        _check_range("rearm_grace_s", self.rearm_grace_s, MIN_REARM_GRACE_S, None)
        _check_range("tone_hz", self.tone_hz, MIN_TONE_HZ, MAX_TONE_HZ)
        if (not isinstance(self.sequences, int) or isinstance(self.sequences, bool)
                or self.sequences < 1):
            raise ValueError("sequences must be an integer >= 1")
        if not isinstance(self.levels, dict) or set(self.levels) != LEVEL_KEYS:
            raise ValueError("levels must have exactly the keys 1, 2, 3")
        for k, g in self.levels.items():
            _check_range(f"level {k} gain", g, 0.0, 1.0)
        if not isinstance(self.level_db, dict) or set(self.level_db) != LEVEL_KEYS:
            raise ValueError("level_db must have exactly the keys 1, 2, 3")
        for k, db in self.level_db.items():
            if db is not None:
                _check_number(f"level {k} dB", db)
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be true or false")
        if not isinstance(self.audio_device, str) or not self.audio_device.strip():
            raise ValueError("audio_device must be a non-empty string")
        validate_zone(self.zone)


LEVEL_KEYS = {1, 2, 3}
MIN_REARM_GRACE_S = 5.0
MIN_TONE_HZ, MAX_TONE_HZ = 200.0, 20000.0


def _check_number(name: str, v) -> None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be a number, got {v!r}")
    if not math.isfinite(v):
        raise ValueError(f"{name} must be finite, got {v}")


def _check_range(name: str, v, lo: float | None, hi: float | None) -> None:
    _check_number(name, v)
    if lo is not None and v < lo:
        raise ValueError(f"{name} must be >= {lo:g}, got {v}")
    if hi is not None and v > hi:
        raise ValueError(f"{name} must be <= {hi:g}, got {v}")


def validate_zone(zone) -> None:
    """A zone is empty (no couch yet) or >= 3 finite (x, y) points in [0, 1]."""
    if not isinstance(zone, (list, tuple)):
        raise ValueError("zone must be a list of points")
    if not zone:
        return
    if len(zone) < 3:
        raise ValueError("zone needs at least 3 points")
    for p in zone:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"zone point {p!r} must be [x, y]")
        for v in p:
            _check_range("zone coordinate", v, 0.0, 1.0)


def settings_path() -> Path:
    return Path(os.environ.get("DOGBOT_CONFIG", "config/settings.json"))


def _parse_settings(text: str) -> Settings:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("settings file must hold a JSON object")
    known = {f.name for f in fields(Settings)}
    data = {k: v for k, v in data.items() if k in known}
    for key in ("levels", "level_db"):
        if key in data:
            if not isinstance(data[key], dict):
                raise ValueError(f"{key} must be an object")
            data[key] = {int(k): v for k, v in data[key].items()}
    if "zone" in data:
        if not isinstance(data["zone"], list):
            raise ValueError("zone must be a list")
        data["zone"] = [tuple(p) for p in data["zone"]]
    s = Settings(**data)
    s.validate()
    return s


def load_settings_checked(path: Path) -> tuple[Settings, str | None]:
    """-> (settings, reason). A missing file gives defaults and no reason.
    A file that is corrupt or fails validate() is renamed to
    `<name>.invalid-<epoch>` (kept for inspection) and defaults are returned
    with the reason, so a bad file can never run the pipeline or lock the
    owner out of saving (every save validates the whole object).

    A file that can't be *read* (permissions, I/O error) is left in place:
    its content may be fine, so defaults are used for this run only."""
    path = Path(path)
    if not path.exists():
        return Settings(), None
    try:
        text = path.read_text()
    except OSError as e:
        reason = f"settings file unreadable: {path.name}: {e} (left in place; using defaults)"
        log.error("%s", reason)
        return Settings(), reason
    try:
        return _parse_settings(text), None
    except Exception as e:  # JSON, types, ranges: all mean "unusable file"
        reason = f"{path.name}: {type(e).__name__}: {e}"
    moved = path.with_name(f"{path.name}.invalid-{int(time.time())}")
    try:
        os.replace(path, moved)
        reason += f" (moved to {moved.name}; using defaults)"
    except OSError as e:
        reason += f" (could not move it aside: {e}; using defaults)"
    log.error("settings file rejected: %s", reason)
    return Settings(), reason


def load_settings(path: Path) -> Settings:
    return load_settings_checked(path)[0]


def save_settings(s: Settings, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(asdict(s), f, indent=2)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
