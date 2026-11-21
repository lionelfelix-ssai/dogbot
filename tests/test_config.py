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


# -- stricter validation (Task 9: the web API 422 path relies on it) ----------

NAN, INF = float("nan"), float("inf")


@pytest.mark.parametrize("kwargs", [
    {"dog_conf": True},
    {"person_conf": False},
    {"dwell_s": True},
    {"tone_hz": True},
    {"sequences": True},
    {"dog_conf": "0.5"},
    {"clear_s": None},
    {"levels": {1: True, 2: 0.7, 3: 1.0}},
    {"level_db": {1: True, 2: None, 3: None}},
])
def test_validate_rejects_wrong_types(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


@pytest.mark.parametrize("kwargs", [
    {"dog_conf": NAN},
    {"person_conf": INF},
    {"dwell_s": NAN},
    {"clear_s": INF},
    {"round_gap_s": -INF},
    {"blast_s": NAN},
    {"tone_hz": NAN},
    {"levels": {1: NAN, 2: 0.7, 3: 1.0}},
    {"level_db": {1: NAN, 2: None, 3: None}},
    {"level_db": {1: None, 2: INF, 3: None}},
    {"zone": [(NAN, 0.1), (0.9, 0.1), (0.5, 0.9)]},
])
def test_validate_rejects_nan_and_inf(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


@pytest.mark.parametrize("kwargs", [
    {"sequences": 0},
    {"sequences": 1.5},
    {"sequences": 2.0},
    {"rearm_grace_s": 4.9},
    {"rearm_grace_s": 0},
    {"dwell_s": -0.1},
    {"blast_spacing_s": -1},
    {"tone_hz": 199},
    {"tone_hz": 20001},
    {"levels": {1: 0.4, 2: 0.7}},
    {"levels": {1: 0.4, 2: 0.7, 3: 1.0, 4: 1.0}},
    {"levels": {"1": 0.4, "2": 0.7, "3": 1.0}},
    {"levels": {1: 0.4, 2: 0.7, 3: 1.01}},
    {"level_db": {1: None, 2: None}},
    {"level_db": {1: "85", 2: None, 3: None}},
    {"enabled": 1},
    {"enabled": "true"},
    {"audio_device": ""},
    {"audio_device": None},
    {"audio_device": 3},
    {"zone": [(0.1, 0.1), (0.9, 0.1), (0.5,)]},
    {"zone": [(0.1, 0.1), (0.9, 0.1), (True, 0.9)]},
])
def test_validate_rejects_bad_values(kwargs):
    with pytest.raises(ValueError):
        Settings(**kwargs).validate()


@pytest.mark.parametrize("kwargs", [
    {},
    {"rearm_grace_s": 5},
    {"tone_hz": 200},
    {"tone_hz": 20000},
    {"dwell_s": 0, "clear_s": 0, "round_gap_s": 0, "blast_spacing_s": 0},
    {"dog_conf": 1, "person_conf": 0},          # ints are fine for floats
    {"sequences": 1},
    {"levels": {1: 0, 2: 0.5, 3: 1}},
    {"level_db": {1: 78.5, 2: 85, 3: None}},
    {"enabled": False},
    {"audio_device": "plughw:CARD=Device,DEV=0"},
    {"zone": [(0, 0), (1, 0), (1, 1)]},
    {"zone": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]},
])
def test_validate_accepts_good_values(kwargs):
    Settings(**kwargs).validate()


# -- R14: validate on load, fall back to defaults ------------------------------

def test_missing_file_is_not_an_error(tmp_path):
    from dogbot.config import load_settings_checked
    s, reason = load_settings_checked(tmp_path / "nope.json")
    assert s == Settings() and reason is None


def test_valid_file_loads_without_reason(tmp_path):
    from dogbot.config import load_settings_checked
    p = tmp_path / "s.json"
    save_settings(Settings(rearm_grace_s=30), p)
    s, reason = load_settings_checked(p)
    assert s.rearm_grace_s == 30 and reason is None
    assert p.exists()


def test_invalid_file_falls_back_to_defaults_and_is_renamed(tmp_path):
    from dogbot.config import load_settings_checked
    p = tmp_path / "settings.json"
    p.write_text('{"rearm_grace_s": 0}')
    s, reason = load_settings_checked(p)
    assert s == Settings()
    assert reason and "rearm_grace_s" in reason
    assert not p.exists()
    moved = list(tmp_path.glob("settings.json.invalid-*"))
    assert len(moved) == 1 and moved[0].read_text() == '{"rearm_grace_s": 0}'
    assert load_settings(p) == Settings()      # plain wrapper agrees


@pytest.mark.parametrize("text", [
    "{not json",
    "[1, 2, 3]",
    '{"levels": {"one": 0.4}}',
    '{"levels": 5}',
    '{"zone": [5, 6, 7]}',
    '{"dog_conf": "high"}',
])
def test_corrupt_json_falls_back(tmp_path, text):
    from dogbot.config import load_settings_checked
    p = tmp_path / "settings.json"
    p.write_text(text)
    s, reason = load_settings_checked(p)
    assert s == Settings() and reason
    assert not p.exists()
    assert len(list(tmp_path.glob("settings.json.invalid-*"))) == 1


def test_load_settings_keeps_signature(tmp_path):
    p = tmp_path / "settings.json"
    p.write_text('{"tone_hz": 50000}')
    assert load_settings(p) == Settings()


def test_unreadable_file_not_renamed(tmp_path, monkeypatch):
    from pathlib import Path

    from dogbot.config import load_settings_checked
    p = tmp_path / "settings.json"
    save_settings(Settings(rearm_grace_s=30, zone=[(0.1, 0.1), (0.9, 0.1), (0.5, 0.9)]), p)
    real = Path.read_bytes

    def read_bytes(self):
        if self == p:
            raise PermissionError(13, "Permission denied", str(self))
        return real(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    s, reason = load_settings_checked(p)
    assert s == Settings()
    assert reason and "unreadable" in reason
    assert p.exists()
    assert list(tmp_path.glob("settings.json.invalid-*")) == []
    monkeypatch.undo()
    assert load_settings_checked(p)[0].rearm_grace_s == 30   # intact for next start


def test_non_utf8_file_moved_aside(tmp_path):
    from dogbot.config import load_settings_checked
    p = tmp_path / "settings.json"
    p.write_bytes(b"\xff\xfe{")
    s, reason = load_settings_checked(p)
    assert s == Settings() and reason
    assert not p.exists()
    moved = list(tmp_path.glob("settings.json.invalid-*"))
    assert len(moved) == 1 and moved[0].read_bytes() == b"\xff\xfe{"


# -- C-1: audio_device reaches `aplay -D`; only plain ALSA names allowed -------

@pytest.mark.parametrize("dev", [
    "file:'|touch /tmp/pwned'",
    "file:/tmp/x.raw",
    "|touch /tmp/pwned",
    "plug:'|sh'",
    "hw:'1'",
    'hw:"1"',
    "hw:1,0;sh",
    "/dev/snd/pcmC1D0p",
    "hw:CARD=a/b",
    "default ",
    " default",
    "plughw:CARD=Device, DEV=0",
    "hw:1,0\n",
    "pulse",
    "dmix",
    "sysdefault:CARD=",
    "hw:",
])
def test_audio_device_rejects_unsafe_names(dev):
    with pytest.raises(ValueError, match="audio_device"):
        Settings(audio_device=dev).validate()


@pytest.mark.parametrize("dev", [
    "default", "sysdefault", "sysdefault:CARD=Device", "plughw:CARD=Device,DEV=0",
    "hw:1,0", "hw:1", "plughw:1,0", "hw:CARD=Device", "plughw:CARD=Device,0",
])
def test_audio_device_accepts_alsa_names(dev):
    Settings(audio_device=dev).validate()


def test_planted_audio_device_in_file_rejected_at_load(tmp_path):
    from dogbot.config import load_settings_checked
    p = tmp_path / "s.json"
    p.write_text('{"audio_device": "file:\'|touch /tmp/pwned\'"}')
    s, reason = load_settings_checked(p)
    assert reason and "audio_device" in reason
    assert s.audio_device == Settings().audio_device


def test_default_audio_device_is_sysdefault():
    assert Settings().audio_device == "sysdefault"
