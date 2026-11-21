"""Couch zone geometry and dwell-based presence."""
from __future__ import annotations

Point = tuple[float, float]


def anchor(box: tuple[float, float, float, float]) -> Point:
    """Bottom-center of a normalized (x1, y1, x2, y2) box."""
    x1, _y1, x2, y2 = box
    return ((x1 + x2) / 2, y2)


def point_in_polygon(pt: Point, poly: list[Point]) -> bool:
    """Ray-casting point-in-polygon. Fewer than 3 points -> False."""
    if len(poly) < 3:
        return False
    x, y = pt
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class CouchPresence:
    """True once any dog has been in the zone continuously for >= dwell_s."""

    def __init__(self, dwell_s: float) -> None:
        self.dwell_s = dwell_s
        self._since: float | None = None

    def update(self, now: float, any_dog_in_zone: bool) -> bool:
        if not any_dog_in_zone:
            self._since = None
            return False
        if self._since is None:
            self._since = now
        return now - self._since >= self.dwell_s
