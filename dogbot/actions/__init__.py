"""Actions: things DogBot does in the world in response to commands."""
from __future__ import annotations

from typing import Protocol


class Action(Protocol):
    def handle(self, cmd) -> None: ...

    def stop(self) -> None: ...
