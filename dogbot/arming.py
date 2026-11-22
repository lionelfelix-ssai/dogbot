"""Person-based auto arming: armed only after a quiet grace period."""
from __future__ import annotations


class Arming:
    """Starts disarmed; armed once `rearm_grace_s` passes with no person seen."""

    def __init__(self, rearm_grace_s: float, start: float) -> None:
        self.rearm_grace_s = rearm_grace_s
        self._last_person_or_start = start

    def update(self, now: float, person_present: bool) -> bool:
        if person_present:
            self._last_person_or_start = now
        return now - self._last_person_or_start >= self.rearm_grace_s

    def seconds_until_armed(self, now: float) -> float:
        return max(0.0, self.rearm_grace_s - (now - self._last_person_or_start))
