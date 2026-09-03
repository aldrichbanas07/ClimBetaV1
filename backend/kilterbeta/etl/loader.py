"""Write side of the ETL.

Both the sample board and the real Kilter ingest funnel through these
functions, so the two sources cannot drift apart in how they populate the
schema. All writes are idempotent upserts: re-running the ETL over an existing
database updates rather than duplicates.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from ..domain.holds import Hold, HoldRole
from ..db.connection import set_meta


def upsert_layout(
    conn: sqlite3.Connection,
    layout_id: int,
    name: str,
    holds: Sequence[Hold],
    product_name: Optional[str] = None,
    source: str = "sample",
) -> None:
    xs = [h.x for h in holds] or [0.0]
    ys = [h.y for h in holds] or [0.0]
    conn.execute(
        """
        INSERT INTO layouts(id, name, product_name, min_x, max_x, min_y, max_y, source)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            name = excluded.name,
            product_name = excluded.product_name,
            min_x = excluded.min_x, max_x = excluded.max_x,
            min_y = excluded.min_y, max_y = excluded.max_y,
            source = excluded.source
        """,
        (layout_id, name, product_name, min(xs), max(xs), min(ys), max(ys), source),
    )


def upsert_holds(
    conn: sqlite3.Connection,
    layout_id: int,
    holds: Iterable[Hold],
    type_sources: Optional[Mapping[int, str]] = None,
    set_ids: Optional[Mapping[int, int]] = None,
) -> int:
    type_sources = type_sources or {}
    set_ids = set_ids or {}
    rows = [
        (
            h.hold_id,
            layout_id,
            h.placement_id,
            h.hole_id,
            set_ids.get(h.hold_id),
            h.name,
            h.x,
            h.y,
            h.hold_type.value,
            type_sources.get(h.hold_id, "default"),
            h.size,
            h.role.value,
        )
        for h in holds
    ]
    conn.executemany(
        """
        INSERT INTO holds(hold_id, layout_id, placement_id, hole_id, set_id, name,
                          x, y, hold_type, hold_type_source, size, default_role)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(hold_id) DO UPDATE SET
            layout_id = excluded.layout_id,
            placement_id = excluded.placement_id,
            hole_id = excluded.hole_id,
            set_id = excluded.set_id,
            name = excluded.name,
            x = excluded.x, y = excluded.y,
            hold_type = excluded.hold_type,
            hold_type_source = excluded.hold_type_source,
            size = excluded.size,
            default_role = excluded.default_role
        """,
        rows,
    )
    return len(rows)


def upsert_climb(
    conn: sqlite3.Connection,
    climb_id: str,
    layout_id: int,
    name: str,
    holds: Sequence[Tuple[int, HoldRole]],
    setter: Optional[str] = None,
    description: Optional[str] = None,
    setter_angle: Optional[int] = None,
    is_listed: bool = True,
    frames: Optional[str] = None,
    source: str = "sample",
) -> None:
    conn.execute(
        """
        INSERT INTO climbs(climb_id, layout_id, name, setter, description,
                           setter_angle, is_listed, hold_count, frames, source)
        VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(climb_id) DO UPDATE SET
            layout_id = excluded.layout_id,
            name = excluded.name,
            setter = excluded.setter,
            description = excluded.description,
            setter_angle = excluded.setter_angle,
            is_listed = excluded.is_listed,
            hold_count = excluded.hold_count,
            frames = excluded.frames,
            source = excluded.source
        """,
        (
            climb_id,
            layout_id,
            name,
            setter,
            description,
            setter_angle,
            1 if is_listed else 0,
            len(holds),
            frames,
            source,
        ),
    )
    # Replace the hold set wholesale: a re-ingested climb may have been edited.
    conn.execute("DELETE FROM climb_holds WHERE climb_id = ?", (climb_id,))
    conn.executemany(
        "INSERT OR REPLACE INTO climb_holds(climb_id, hold_id, role) VALUES(?, ?, ?)",
        [(climb_id, hold_id, role.value) for hold_id, role in holds],
    )


def upsert_stats(
    conn: sqlite3.Connection,
    climb_id: str,
    per_angle: Mapping[int, Mapping[str, Optional[float]]],
) -> int:
    rows = [
        (
            climb_id,
            int(angle),
            data.get("display_difficulty"),
            data.get("benchmark_difficulty"),
            data.get("ascensionist_count"),
            data.get("quality_average"),
        )
        for angle, data in per_angle.items()
    ]
    conn.executemany(
        """
        INSERT INTO climb_stats(climb_id, angle, display_difficulty,
                                benchmark_difficulty, ascensionist_count, quality_average)
        VALUES(?, ?, ?, ?, ?, ?)
        ON CONFLICT(climb_id, angle) DO UPDATE SET
            display_difficulty = excluded.display_difficulty,
            benchmark_difficulty = excluded.benchmark_difficulty,
            ascensionist_count = excluded.ascensionist_count,
            quality_average = excluded.quality_average
        """,
        rows,
    )
    return len(rows)


def upsert_grades(conn: sqlite3.Connection, grades: Mapping[int, Tuple[Optional[str], Optional[str]]]) -> int:
    rows = [(int(d), names[0], names[1]) for d, names in grades.items()]
    conn.executemany(
        """
        INSERT INTO difficulty_grades(difficulty, boulder_name, route_name)
        VALUES(?, ?, ?)
        ON CONFLICT(difficulty) DO UPDATE SET
            boulder_name = excluded.boulder_name,
            route_name = excluded.route_name
        """,
        rows,
    )
    return len(rows)


def record_ingest(conn: sqlite3.Connection, source: str, detail: str) -> None:
    from datetime import datetime, timezone

    set_meta(conn, f"ingest:{source}", f"{datetime.now(timezone.utc).isoformat()} {detail}")
