import pytest

from dogbot.config import Settings
from dogbot.rules import CouchGuard, PlayBlasts, RuleEvent, StopSound

TICK = 0.1
TOL = 0.11  # one tick of slack for "≈"


def run(guard, start_i, end_i, occupied, active=lambda t: True):
    """Tick the guard at t = i*0.1 for i in [start_i, end_i).

    Returns a list of (t, commands, events) for ticks that emitted anything.
    """
    out = []
    for i in range(start_i, end_i):
        t = i * TICK
        cmds, events = guard.update(t, active(t), occupied(t))
        if cmds or events:
            out.append((t, cmds, events))
    return out


def plays(log):
    return [(t, c) for t, cmds, _ in log for c in cmds if isinstance(c, PlayBlasts)]


def events_of(log, kind):
    return [(t, e) for t, _, evs in log for e in evs if e.kind == kind]


def test_full_escalation():
    g = CouchGuard(Settings())
    log = run(g, 0, 600, occupied=lambda t: True)  # 60 s on the couch

    t0, cmds0, evs0 = log[0]
    assert t0 == 0
    assert cmds0 == [PlayBlasts(1, 1)]
    assert RuleEvent("intrusion", {}) in evs0
    assert RuleEvent("round", {"sequence": 1, "round": 1}) in evs0

    p = plays(log)
    expected = [
        (0.0, PlayBlasts(1, 1)),
        (5.5, PlayBlasts(2, 2)),
        (11.9, PlayBlasts(3, 3)),
        (19.2, PlayBlasts(1, 1)),   # sequence 2
        (24.7, PlayBlasts(2, 2)),
        (31.1, PlayBlasts(3, 3)),
    ]
    assert len(p) == len(expected)
    for (t, c), (et, ec) in zip(p, expected):
        assert c == ec
        assert t == pytest.approx(et, abs=TOL)

    rounds = [e.detail for _, e in events_of(log, "round")]
    assert rounds == [{"sequence": s, "round": r} for s in (1, 2) for r in (1, 2, 3)]

    unresolved = events_of(log, "unresolved")
    assert len(unresolved) == 1
    # seq 2 round 3 ends at 33.4, then a 5 s gap
    assert unresolved[0][0] == pytest.approx(38.4, abs=TOL)
    assert unresolved[0][1] == RuleEvent("unresolved", {"sequences": 2})
    assert g.state == "unresolved"
    assert len(events_of(log, "intrusion")) == 1


def test_dog_leaves_then_clear():
    g = CouchGuard(Settings())
    log = run(g, 0, 200, occupied=lambda t: t < 3.0 - 1e-9)

    assert plays(log) == [(0.0, PlayBlasts(1, 1))]  # no new rounds while paused
    cleared = events_of(log, "cleared")
    assert len(cleared) == 1
    assert cleared[0][0] == pytest.approx(13.0, abs=TOL)
    assert cleared[0][1] == RuleEvent("cleared", {})
    assert g.state == "idle"


def test_paused_state_while_away():
    g = CouchGuard(Settings())
    run(g, 0, 30, occupied=lambda t: True)
    g.update(3.0, True, False)
    assert g.state == "paused"
    g.update(12.9, True, False)
    assert g.state == "paused"


def test_resume_before_clear():
    g = CouchGuard(Settings())
    occ = lambda t: not (2.0 - 1e-9 <= t < 6.0 - 1e-9)
    log = run(g, 0, 200, occupied=occ)
    p = plays(log)
    assert p[0] == (0.0, PlayBlasts(1, 1))
    t1, c1 = p[1]
    assert c1 == PlayBlasts(2, 2)
    # gap restarts on return at t=6
    assert t1 == pytest.approx(11.0, abs=TOL)
    assert events_of(log, "cleared") == []
    assert len(events_of(log, "intrusion")) == 1


def test_restart_after_full_clear():
    g = CouchGuard(Settings())
    # on 0-3, away 3-20 (cleared at 13), back at 20
    occ = lambda t: t < 3.0 - 1e-9 or t >= 20.0 - 1e-9
    log = run(g, 0, 230, occupied=occ)
    p = plays(log)
    assert p[0] == (0.0, PlayBlasts(1, 1))
    assert p[1][1] == PlayBlasts(1, 1)
    assert p[1][0] == pytest.approx(20.0, abs=TOL)
    assert len(events_of(log, "intrusion")) == 2
    assert g.round == 1 and g.sequence == 1


def test_deactivate_stops_sound():
    g = CouchGuard(Settings())
    g.update(0.0, True, True)
    g.update(0.1, True, True)
    assert g.state == "blasting"
    cmds, events = g.update(0.2, False, True)
    assert cmds == [StopSound()]
    assert events == [RuleEvent("stopped", {"reason": "inactive"})]
    assert g.state == "held"
    assert g.round == 1 and g.sequence == 1


def test_deactivate_from_gap_holds_without_stopped_event():
    g = CouchGuard(Settings())
    run(g, 0, 20, occupied=lambda t: True)
    assert g.state == "gap"
    cmds, events = g.update(2.0, False, True)
    assert cmds == [StopSound()]
    assert events == []
    assert g.state == "held"
    # inactive while held: silent
    log = run(g, 21, 100, occupied=lambda t: True, active=lambda t: False)
    assert log == []
    assert g.state == "held"


def test_held_resumes_with_gap_then_next_round():
    g = CouchGuard(Settings())
    run(g, 0, 20, occupied=lambda t: True)       # round 1 done, in gap
    run(g, 20, 40, occupied=lambda t: True, active=lambda t: False)
    log = run(g, 40, 150, occupied=lambda t: True)  # reactivated at t=4
    p = plays(log)
    assert p[0][1] == PlayBlasts(2, 2)
    assert p[0][0] == pytest.approx(9.0, abs=TOL)
    assert events_of(log, "intrusion") == []


def test_flapping_active_cannot_bypass_cap():
    s = Settings()
    g = CouchGuard(s)
    active = lambda t: int(t + 1e-9) % 2 == 0     # on 1 s, off 1 s
    log = []
    reactivations = []
    prev = None
    for i in range(0, 1200):
        t = i * TICK
        a = active(t)
        if prev is False and a:
            reactivations.append(t)
        prev = a
        cmds, evs = g.update(t, a, True)
        log.append((t, cmds, evs))
    p = plays(log)
    assert len(p) <= 3 * s.sequences
    assert len(events_of(log, "intrusion")) == 1
    for t, _ in p[1:]:
        last = max(r for r in reactivations if r <= t + 1e-9) if any(
            r <= t + 1e-9 for r in reactivations) else None
        if last is not None:
            assert t - last >= s.round_gap_s - 1e-6


def test_slow_flapping_still_reaches_unresolved():
    s = Settings()
    g = CouchGuard(s)
    active = lambda t: (t + 1e-9) % 10 >= 1.0     # off 1 s in every 10 s
    log = run(g, 10, 3000, occupied=lambda t: True, active=active)
    p = plays(log)
    assert len(p) <= 3 * s.sequences
    assert len(events_of(log, "unresolved")) == 1
    assert len(events_of(log, "intrusion")) == 1
    assert g.state == "unresolved"


def test_held_clears_after_no_occupancy():
    g = CouchGuard(Settings())
    g.update(0.0, True, True)
    g.update(0.1, False, True)
    assert g.state == "held"
    # dog leaves at t=1 while still inactive; clear counted regardless
    log = run(g, 10, 120, occupied=lambda t: False, active=lambda t: False)
    cleared = events_of(log, "cleared")
    assert len(cleared) == 1
    assert cleared[0][0] == pytest.approx(11.0, abs=TOL)
    assert [c for _, cmds, _ in log for c in cmds] == []
    assert g.state == "idle"
    cmds, events = g.update(20.0, True, True)
    assert cmds == [PlayBlasts(1, 1)]
    assert RuleEvent("intrusion", {}) in events


def test_held_clear_timer_resets_on_return():
    g = CouchGuard(Settings())
    g.update(0.0, True, True)
    g.update(0.1, False, True)
    run(g, 10, 60, occupied=lambda t: False, active=lambda t: False)
    run(g, 60, 61, occupied=lambda t: True, active=lambda t: False)
    log = run(g, 61, 150, occupied=lambda t: False, active=lambda t: False)
    assert events_of(log, "cleared") == []
    assert g.state == "held"


def test_held_at_final_round_goes_unresolved():
    s = Settings()
    g = CouchGuard(s)
    run(g, 0, 312, occupied=lambda t: True)       # seq 2 round 3 starts at 31.1
    assert (g.state, g.sequence, g.round) == ("blasting", 2, 3)
    g.update(31.2, False, True)
    assert g.state == "held"
    cmds, events = g.update(40.0, True, True)
    assert cmds == []
    assert events == [RuleEvent("unresolved", {"sequences": 2})]
    assert g.state == "unresolved"


def test_unresolved_inactive_stays_silent():
    g = CouchGuard(Settings())
    run(g, 0, 390, occupied=lambda t: True)
    assert g.state == "unresolved"
    log = run(g, 390, 500, occupied=lambda t: True, active=lambda t: False)
    assert log == []
    assert g.state == "unresolved"
    log = run(g, 500, 700, occupied=lambda t: True)
    assert plays(log) == []
    assert g.state == "unresolved"


def test_leave_during_final_round_then_return():
    g = CouchGuard(Settings())
    run(g, 0, 315, occupied=lambda t: True)       # in seq 2 round 3
    assert (g.state, g.sequence, g.round) == ("blasting", 2, 3)
    cmds, _ = g.update(31.5, True, False)
    assert cmds == [StopSound()]
    run(g, 316, 350, occupied=lambda t: False)   # away < clear_s
    log = run(g, 350, 900, occupied=lambda t: True)
    assert plays(log) == []
    assert len(events_of(log, "unresolved")) == 1
    assert events_of(log, "cleared") == []
    assert g.state == "unresolved"


def test_inactive_never_blasts():
    g = CouchGuard(Settings())
    log = run(g, 0, 300, occupied=lambda t: True, active=lambda t: False)
    assert log == []
    assert g.state == "idle"


def test_inactive_while_idle_emits_nothing():
    g = CouchGuard(Settings())
    assert g.update(0.0, False, False) == ([], [])
    assert g.update(0.1, False, True) == ([], [])


def test_dog_leaves_mid_blast_stops_sound():
    g = CouchGuard(Settings())
    run(g, 0, 56, occupied=lambda t: True)  # round 2 starts at 5.5 (1.4 s long)
    assert g.state == "blasting"
    assert g.round == 2
    cmds, events = g.update(5.8, True, False)
    assert cmds == [StopSound()]
    assert events == []
    assert g.state == "paused"
    # interrupted round counts; on return the gap restarts, then round 3
    log = run(g, 60, 200, occupied=lambda t: True)
    p = plays(log)
    assert p[0][1] == PlayBlasts(3, 3)
    assert p[0][0] == pytest.approx(11.0, abs=TOL)


def test_unresolved_silent_until_clear():
    g = CouchGuard(Settings())
    run(g, 0, 390, occupied=lambda t: True)
    assert g.state == "unresolved"
    log = run(g, 390, 990, occupied=lambda t: True)  # 60 s more on the couch
    assert log == []
    assert g.state == "unresolved"
    # leave at t=99; cleared after clear_s
    log = run(g, 990, 1200, occupied=lambda t: False)
    assert [e for _, _, evs in log for e in evs] == [RuleEvent("cleared", {})]
    assert log[0][1] == []
    assert log[0][0] == pytest.approx(109.0, abs=TOL)
    assert g.state == "idle"


def test_unresolved_return_resets_clear_timer():
    g = CouchGuard(Settings())
    run(g, 0, 390, occupied=lambda t: True)
    run(g, 390, 450, occupied=lambda t: False)   # away 6 s
    run(g, 450, 460, occupied=lambda t: True)    # back briefly
    log = run(g, 460, 550, occupied=lambda t: False)  # away 9 s < clear_s
    assert log == []
    assert g.state == "unresolved"
