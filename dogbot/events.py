"""SQLite event log with JPEG snapshots (thread-safe)."""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger(__name__)

KINDS = (
    "armed", "disarmed", "intrusion", "round", "cleared", "unresolved",
    "stopped", "enabled", "disabled", "settings", "zone", "test_blast",
    "error", "startup", "held",
)


class EventStore:
    def __init__(self, db_path: Path, snapshot_dir: Path):
        self._db_path = Path(db_path)
        self._snap_dir = Path(snapshot_dir)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._snap_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, "
            "kind TEXT NOT NULL, detail TEXT NOT NULL, "
            "has_snapshot INTEGER NOT NULL DEFAULT 0)"
        )
        self._conn.commit()

    def log(self, kind: str, detail: dict, snapshot_jpeg: bytes | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (ts, kind, detail) VALUES (?, ?, ?)",
                (time.time(), kind, json.dumps(detail)),
            )
            event_id = cur.lastrowid
            self._conn.commit()
            if snapshot_jpeg is not None:
                try:
                    self._snap_dir.mkdir(parents=True, exist_ok=True)
                    (self._snap_dir / f"{event_id}.jpg").write_bytes(snapshot_jpeg)
                    self._conn.execute(
                        "UPDATE events SET has_snapshot = 1 WHERE id = ?", (event_id,)
                    )
                    self._conn.commit()
                except OSError:
                    log.exception("snapshot write failed for event %s", event_id)
        return event_id

    def list(self, limit: int = 100, before_id: int | None = None,
             kinds: set[str] | None = None) -> list[dict]:
        sql = "SELECT id, ts, kind, detail, has_snapshot FROM events"
        where, params = [], []
        if before_id is not None:
            where.append("id < ?")
            params.append(before_id)
        if kinds:
            where.append("kind IN (%s)" % ",".join("?" * len(kinds)))
            params.extend(sorted(kinds))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            {
                "id": i,
                "ts": datetime.fromtimestamp(ts).astimezone().isoformat(timespec="seconds"),
                "kind": k,
                "detail": json.loads(d),
                "has_snapshot": bool(h),
            }
            for i, ts, k, d, h in rows
        ]

    def snapshot_path(self, event_id: int) -> Path | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT has_snapshot FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        if not row or not row[0]:
            return None
        return self._snap_dir / f"{event_id}.jpg"
