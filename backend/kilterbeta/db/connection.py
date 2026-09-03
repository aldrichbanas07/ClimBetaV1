"""SQLite connection helpers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def connect(path: Path, read_only: bool = False) -> sqlite3.Connection:
    """Open a connection with sane defaults for this project.

    ``check_same_thread=False``: sqlite3's default thread-affinity check
    assumes one connection is used by exactly one thread for its whole life.
    That does not hold for a FastAPI sync dependency -- anyio's threadpool is
    free to run the code before and after a generator dependency's ``yield``
    on two different worker threads, so a connection opened in one and closed
    in the other trips the check even though it is never touched
    concurrently. Each connection here is still scoped to a single request
    and never shared across requests, so the actual safety property the check
    exists for is not being violated.
    """
    path = Path(path)
    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"database not found: {path}")
        conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(path))

    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if not read_only:
        # Faster bulk ETL writes; the analysis DB is rebuildable so durability
        # is not worth the sync cost.
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()


@contextmanager
def open_db(path: Path, read_only: bool = False, init: bool = False) -> Iterator[sqlite3.Connection]:
    conn = connect(path, read_only=read_only)
    try:
        if init:
            init_schema(conn)
        yield conn
    finally:
        conn.close()


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None
