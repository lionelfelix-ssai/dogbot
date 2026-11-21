"""Detection result shared by the decoder, tracker and zone logic."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One detected object. `box` is normalized (x1, y1, x2, y2) in [0, 1]."""

    cls_id: int
    conf: float
    box: tuple[float, float, float, float]
