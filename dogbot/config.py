"""Settings model for DogBot. Zones are normalized (0-1) points."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


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
        for name in ("dog_conf", "person_conf"):
            v = getattr(self, name)
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        for name in ("dwell_s", "rearm_grace_s", "clear_s", "round_gap_s",
                     "blast_s", "blast_spacing_s"):
            v = getattr(self, name)
            if v < 0:
                raise ValueError(f"{name} must be >= 0, got {v}")
        if self.tone_hz <= 0:
            raise ValueError("tone_hz must be > 0")
        if not isinstance(self.sequences, int) or self.sequences < 1:
            raise ValueError("sequences must be an integer >= 1")
        if set(self.levels) != {1, 2, 3}:
            raise ValueError("levels must have keys 1, 2, 3")
        for k, g in self.levels.items():
            if not 0.0 <= g <= 1.0:
                raise ValueError(f"level {k} gain must be in [0, 1], got {g}")
        if set(self.level_db) != {1, 2, 3}:
            raise ValueError("level_db must have keys 1, 2, 3")
        if self.zone:
            if len(self.zone) < 3:
                raise ValueError("zone needs at least 3 points")
            for x, y in self.zone:
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    raise ValueError(f"zone point ({x}, {y}) outside [0, 1]")


def settings_path() -> Path:
    return Path(os.environ.get("DOGBOT_CONFIG", "config/settings.json"))


def load_settings(path: Path) -> Settings:
    path = Path(path)
    if not path.exists():
        return Settings()
    data = json.loads(path.read_text())
    known = {f.name for f in fields(Settings)}
    data = {k: v for k, v in data.items() if k in known}
    for key in ("levels", "level_db"):
        if key in data:
            data[key] = {int(k): v for k, v in data[key].items()}
    if "zone" in data:
        data["zone"] = [tuple(p) for p in data["zone"]]
    return Settings(**data)


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
