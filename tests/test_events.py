import threading
from datetime import datetime

from dogbot.events import EventStore


def make(tmp_path):
    return EventStore(tmp_path / "data" / "events.db", tmp_path / "snaps" / "x")


def test_list_newest_first_and_roundtrip(tmp_path):
    s = make(tmp_path)
    ids = [s.log("armed", {}), s.log("intrusion", {"dog": 1, "conf": 0.9}), s.log("round", {"n": 1})]
    rows = s.list()
    assert [r["id"] for r in rows] == ids[::-1]
    assert rows[1]["detail"] == {"dog": 1, "conf": 0.9}
    assert rows[1]["kind"] == "intrusion"
    assert rows[0]["has_snapshot"] is False
    ts = datetime.fromisoformat(rows[0]["ts"])
    assert ts.utcoffset() is not None


def test_kinds_filter(tmp_path):
    s = make(tmp_path)
    s.log("armed", {})
    s.log("held", {})
    s.log("round", {})
    assert [r["kind"] for r in s.list(kinds={"armed", "held"})] == ["held", "armed"]


def test_before_id_and_limit(tmp_path):
    s = make(tmp_path)
    ids = [s.log("round", {"i": i}) for i in range(5)]
    page = s.list(limit=2, before_id=ids[3])
    assert [r["id"] for r in page] == [ids[2], ids[1]]


def test_snapshot_written(tmp_path):
    s = make(tmp_path)
    eid = s.log("intrusion", {}, snapshot_jpeg=b"\xff\xd8jpeg")
    p = s.snapshot_path(eid)
    assert p == tmp_path / "snaps" / "x" / f"{eid}.jpg"
    assert p.read_bytes() == b"\xff\xd8jpeg"
    assert s.list()[0]["has_snapshot"] is True


def test_no_snapshot_path_none(tmp_path):
    s = make(tmp_path)
    eid = s.log("armed", {})
    assert s.snapshot_path(eid) is None
    assert s.snapshot_path(9999) is None


def test_snapshot_failure_keeps_row(tmp_path):
    s = make(tmp_path)
    (tmp_path / "snaps" / "x").rmdir()
    (tmp_path / "snaps" / "x").write_bytes(b"not a dir")  # dir path is now a file
    eid = s.log("intrusion", {"a": 1}, snapshot_jpeg=b"jpg")
    row = s.list()[0]
    assert row["id"] == eid and row["has_snapshot"] is False
    assert s.snapshot_path(eid) is None


def test_concurrent_logging(tmp_path):
    s = make(tmp_path)
    def work():
        for _ in range(25):
            s.log("round", {})
            s.list(limit=5)
    ts = [threading.Thread(target=work) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(s.list(limit=1000)) == 100
