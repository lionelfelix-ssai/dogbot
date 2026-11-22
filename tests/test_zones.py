import pytest

from dogbot.zones import CouchPresence, anchor, point_in_polygon

SQUARE = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]


def test_anchor_bottom_center():
    assert anchor((0.2, 0.2, 0.4, 0.6)) == pytest.approx((0.3, 0.6))


def test_point_in_square():
    assert point_in_polygon((0.5, 0.5), SQUARE) is True


def test_point_outside():
    assert point_in_polygon((0.9, 0.5), SQUARE) is False
    assert point_in_polygon((0.5, 0.1), SQUARE) is False


def test_empty_polygon_false():
    assert point_in_polygon((0.5, 0.5), []) is False
    assert point_in_polygon((0.5, 0.5), [(0, 0), (1, 1)]) is False


def test_dwell():
    p = CouchPresence(dwell_s=1.0)
    assert p.update(0.0, True) is False
    assert p.update(0.9, True) is False
    assert p.update(1.0, True) is True


def test_dog_leaves_resets_dwell():
    p = CouchPresence(dwell_s=1.0)
    assert p.update(0.0, True) is False
    assert p.update(0.5, False) is False
    assert p.update(0.6, True) is False
    assert p.update(1.2, True) is False
    assert p.update(1.6, True) is True


def test_two_dogs_alternating_counts_as_continuous():
    p = CouchPresence(dwell_s=1.0)
    results = [p.update(t / 10, True) for t in range(0, 31)]
    assert results[:10] == [False] * 10
    assert all(results[10:])
