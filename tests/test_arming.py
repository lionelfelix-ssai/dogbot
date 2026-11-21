from dogbot.arming import Arming


def test_starts_disarmed():
    a = Arming(rearm_grace_s=60, start=0.0)
    assert a.update(0.0, False) is False
    assert a.update(59.0, False) is False
    assert a.update(60.0, False) is True


def test_person_disarms_immediately():
    a = Arming(rearm_grace_s=60, start=0.0)
    assert a.update(60.0, False) is True
    assert a.update(61.0, True) is False


def test_single_frame_person_restarts_grace():
    a = Arming(rearm_grace_s=60, start=0.0)
    assert a.update(100.0, True) is False
    assert a.update(130.0, False) is False
    assert a.update(159.9, False) is False
    assert a.update(160.0, False) is True


def test_seconds_until_armed():
    a = Arming(rearm_grace_s=60, start=0.0)
    a.update(100.0, True)
    assert a.seconds_until_armed(130.0) == 30
    assert a.seconds_until_armed(200.0) == 0
