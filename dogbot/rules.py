"""Couch guard: the escalation state machine that decides when to blast.

Pure module: no clock calls, no I/O. The caller passes `now` (seconds) on
every tick and executes the returned commands.

States::

    idle       --occupied & active-->        blasting (round 1, seq 1) + intrusion + round
    blasting   --round duration elapsed-->   gap
    gap        --gap elapsed (occupied)-->   blasting (next round) + round
    gap        --gap elapsed after final round of final sequence--> unresolved + unresolved
    blasting   --!occupied-->                paused + StopSound      (ruling R1)
    gap        --!occupied-->                paused
    paused     --occupied-->                 gap (gap restarts at now)
    paused     --!occupied for clear_s-->    idle + cleared
    unresolved --!occupied for clear_s-->    idle + cleared (timer resets if the dog returns)
    any non-idle --!active-->                idle + StopSound + stopped

Round n of each sequence plays n blasts at level n (n = 1, 2, 3).
Round duration = count*blast_s + (count-1)*blast_spacing_s; the gap
(round_gap_s) is measured from the end of the round. A round interrupted
by the dog leaving counts as played: on return the next round follows.

Event details:
    intrusion  {}
    round      {"sequence": s, "round": r}
    cleared    {}
    unresolved {"sequences": n}
    stopped    {"reason": "inactive"}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

from dogbot.config import Settings

ROUNDS_PER_SEQUENCE = 3
_EPS = 1e-6  # tolerance so tick-stepped float times hit deadlines on time


@dataclass(frozen=True)
class PlayBlasts:
    """Play `count` blasts at escalation `level` (1-3)."""
    count: int
    level: int


@dataclass(frozen=True)
class StopSound:
    """Stop any sound that is playing, immediately."""


Command = Union[PlayBlasts, StopSound]


@dataclass
class RuleEvent:
    kind: str  # intrusion, round, cleared, unresolved, stopped
    detail: dict = field(default_factory=dict)


class CouchGuard:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._reset()

    def _reset(self) -> None:
        self.state = "idle"
        self.round = 0
        self.sequence = 0
        self._round_end = 0.0
        self._gap_end = 0.0
        self._clear_since: float | None = None

    def _round_duration(self, count: int) -> float:
        s = self.settings
        return count * s.blast_s + (count - 1) * s.blast_spacing_s

    def _start_round(self, now: float, sequence: int, rnd: int,
                     cmds: list, events: list) -> None:
        self.state = "blasting"
        self.sequence = sequence
        self.round = rnd
        self._round_end = now + self._round_duration(rnd)
        cmds.append(PlayBlasts(count=rnd, level=rnd))
        events.append(RuleEvent("round", {"sequence": sequence, "round": rnd}))

    def _is_final_round(self) -> bool:
        return (self.round >= ROUNDS_PER_SEQUENCE
                and self.sequence >= self.settings.sequences)

    def update(self, now: float, active: bool,
               occupied: bool) -> tuple[list[Command], list[RuleEvent]]:
        """Advance the machine. `active` = enabled and armed and not fault;
        `occupied` = dwell-confirmed dog presence in the couch zone."""
        cmds: list[Command] = []
        events: list[RuleEvent] = []

        if not active:
            if self.state != "idle":
                cmds.append(StopSound())
                events.append(RuleEvent("stopped", {"reason": "inactive"}))
                self._reset()
            return cmds, events

        if self.state == "idle":
            if occupied:
                events.append(RuleEvent("intrusion", {}))
                self._start_round(now, 1, 1, cmds, events)
            return cmds, events

        if self.state in ("blasting", "gap") and not occupied:
            if self.state == "blasting":
                cmds.append(StopSound())
            self.state = "paused"
            self._clear_since = now
            return cmds, events

        if self.state == "paused":
            if occupied:
                self.state = "gap"
                self._gap_end = now + self.settings.round_gap_s
                self._clear_since = None
            elif now - self._clear_since + _EPS >= self.settings.clear_s:
                events.append(RuleEvent("cleared", {}))
                self._reset()
                return cmds, events

        if self.state == "unresolved":
            if occupied:
                self._clear_since = None
            else:
                if self._clear_since is None:
                    self._clear_since = now
                if now - self._clear_since + _EPS >= self.settings.clear_s:
                    events.append(RuleEvent("cleared", {}))
                    self._reset()
            return cmds, events

        if self.state == "blasting" and now + _EPS >= self._round_end:
            self.state = "gap"
            self._gap_end = self._round_end + self.settings.round_gap_s

        if self.state == "gap" and now + _EPS >= self._gap_end:
            if self._is_final_round():
                self.state = "unresolved"
                self._clear_since = None
                events.append(RuleEvent("unresolved",
                                        {"sequences": self.settings.sequences}))
            elif self.round >= ROUNDS_PER_SEQUENCE:
                self._start_round(now, self.sequence + 1, 1, cmds, events)
            else:
                self._start_round(now, self.sequence, self.round + 1, cmds, events)

        return cmds, events
