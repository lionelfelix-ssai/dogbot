import pytest

from dogbot.config import Settings, load_settings, save_settings


def test_defaults_when_missing(tmp_path):
    s = load_settings(tmp_path / "x.json")
    assert s.dog_conf == 0.5
    assert s.sequences == 2


def test_roundtrip(tmp_path):
    s = Settings()
    s.rearm_grace_s = 30
    s.zone = [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    p = tmp_path / "s.json"
    save_settings(s, p)
    t = load_settings(p)
    assert t == s
    assert t.rearm_grace_s == 30
    assert t.zone == [(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]
    assert t.levels == {1: 0.4, 2: 0.7, 3: 1.0}


def test_validate_rejects_short_zone():
    s = Settings(zone=[(0.1, 0.1), (0.9, 0.9)])
    with pytest.raises(ValueError):
        s.validate()
    Settings(zone=[]).validate()


def test_validate_rejects_out_of_range():
    with pytest.raises(ValueError):
        Settings(zone=[(1.2, 0.5), (0.1, 0.1), (0.5, 0.9)]).validate()
    with pytest.raises(ValueError):
        Settings(dog_conf=1.5).validate()
    with pytest.raises(ValueError):
        Settings(levels={1: -0.1, 2: 0.7, 3: 1.0}).validate()
