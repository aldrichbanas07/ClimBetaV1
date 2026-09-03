"""Read-side queries: the only place the API touches SQL.

Returns domain objects (``Hold``), not rows, so the beta generator never sees
a database detail.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..domain.holds import Hold, HoldRole, HoldType


def _hold_type(raw: Optional[str]) -> HoldType:
    try:
        return HoldType(raw) if raw else HoldType.UNKNOWN
    except ValueError:
        return HoldType.UNKNOWN


def _role(raw: Optional[str]) -> HoldRole:
    try:
        return HoldRole(raw) if raw else HoldRole.HAND
    except ValueError:
        return HoldRole.HAND


@dataclass
class Layout:
    id: int
    name: str
    product_name: Optional[str]
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    source: str


@dataclass
class ClimbSummary:
    climb_id: str
    layout_id: int
    name: str
    setter: Optional[str]
    description: Optional[str]
    setter_angle: Optional[int]
    hold_count: int
    source: str
    #: Angles for which community difficulty exists, if any.
    graded_angles: List[int]


class Repository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------- layouts

    def list_layouts(self) -> List[Layout]:
        rows = self.conn.execute(
            "SELECT id, name, product_name, min_x, max_x, min_y, max_y, source "
            "FROM layouts ORDER BY id"
        ).fetchall()
        return [Layout(**dict(r)) for r in rows]

    def get_layout(self, layout_id: int) -> Optional[Layout]:
        row = self.conn.execute(
            "SELECT id, name, product_name, min_x, max_x, min_y, max_y, source "
            "FROM layouts WHERE id = ?",
            (layout_id,),
        ).fetchone()
        return Layout(**dict(row)) if row else None

    # --------------------------------------------------------------- holds

    def layout_holds(self, layout_id: int) -> List[Hold]:
        """Every hold on the board, for the faint 'unused holds' underlay."""
        rows = self.conn.execute(
            "SELECT hold_id, placement_id, hole_id, name, x, y, hold_type, size, default_role "
            "FROM holds WHERE layout_id = ? ORDER BY y, x",
            (layout_id,),
        ).fetchall()
        return [
            Hold(
                hold_id=r["hold_id"],
                x=r["x"],
                y=r["y"],
                hold_type=_hold_type(r["hold_type"]),
                role=_role(r["default_role"]),
                placement_id=r["placement_id"],
                hole_id=r["hole_id"],
                name=r["name"],
                size=r["size"],
            )
            for r in rows
        ]

    def climb_holds(self, climb_id: str) -> List[Hold]:
        """The holds of one climb, with per-climb roles applied."""
        rows = self.conn.execute(
            """
            SELECT h.hold_id, h.placement_id, h.hole_id, h.name, h.x, h.y,
                   h.hold_type, h.size, ch.role
            FROM climb_holds ch
            JOIN holds h ON h.hold_id = ch.hold_id
            WHERE ch.climb_id = ?
            ORDER BY h.y, h.x
            """,
            (climb_id,),
        ).fetchall()
        return [
            Hold(
                hold_id=r["hold_id"],
                x=r["x"],
                y=r["y"],
                hold_type=_hold_type(r["hold_type"]),
                role=_role(r["role"]),
                placement_id=r["placement_id"],
                hole_id=r["hole_id"],
                name=r["name"],
                size=r["size"],
            )
            for r in rows
        ]

    def holds_by_ids(self, hold_ids: Sequence[int]) -> Dict[int, Hold]:
        """Look up arbitrary holds, for ad-hoc ``hold_list`` requests."""
        if not hold_ids:
            return {}
        marks = ",".join("?" for _ in hold_ids)
        rows = self.conn.execute(
            f"SELECT hold_id, placement_id, hole_id, name, x, y, hold_type, size, default_role "
            f"FROM holds WHERE hold_id IN ({marks})",
            tuple(int(h) for h in hold_ids),
        ).fetchall()
        return {
            r["hold_id"]: Hold(
                hold_id=r["hold_id"],
                x=r["x"],
                y=r["y"],
                hold_type=_hold_type(r["hold_type"]),
                role=_role(r["default_role"]),
                placement_id=r["placement_id"],
                hole_id=r["hole_id"],
                name=r["name"],
                size=r["size"],
            )
            for r in rows
        }

    # -------------------------------------------------------------- climbs

    def list_climbs(
        self,
        layout_id: Optional[int] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
        min_holds: int = 2,
    ) -> List[ClimbSummary]:
        where = ["c.is_listed = 1", "c.hold_count >= ?"]
        params: List[object] = [min_holds]
        if layout_id is not None:
            where.append("c.layout_id = ?")
            params.append(layout_id)
        if search:
            where.append("c.name LIKE ?")
            params.append(f"%{search}%")

        rows = self.conn.execute(
            f"""
            SELECT c.climb_id, c.layout_id, c.name, c.setter, c.description,
                   c.setter_angle, c.hold_count, c.source,
                   (SELECT GROUP_CONCAT(s.angle)
                      FROM climb_stats s
                     WHERE s.climb_id = c.climb_id) AS angles
            FROM climbs c
            WHERE {' AND '.join(where)}
            ORDER BY c.hold_count, c.name
            LIMIT ? OFFSET ?
            """,
            (*params, limit, offset),
        ).fetchall()

        out: List[ClimbSummary] = []
        for r in rows:
            angles = sorted(
                {int(a) for a in (r["angles"] or "").split(",") if a.strip().isdigit()}
            )
            out.append(
                ClimbSummary(
                    climb_id=r["climb_id"],
                    layout_id=r["layout_id"],
                    name=r["name"],
                    setter=r["setter"],
                    description=r["description"],
                    setter_angle=r["setter_angle"],
                    hold_count=r["hold_count"],
                    source=r["source"],
                    graded_angles=angles,
                )
            )
        return out

    def get_climb(self, climb_id: str) -> Optional[ClimbSummary]:
        results = self.conn.execute(
            """
            SELECT c.climb_id, c.layout_id, c.name, c.setter, c.description,
                   c.setter_angle, c.hold_count, c.source,
                   (SELECT GROUP_CONCAT(s.angle)
                      FROM climb_stats s WHERE s.climb_id = c.climb_id) AS angles
            FROM climbs c WHERE c.climb_id = ?
            """,
            (climb_id,),
        ).fetchone()
        if not results:
            return None
        angles = sorted(
            {int(a) for a in (results["angles"] or "").split(",") if a.strip().isdigit()}
        )
        return ClimbSummary(
            climb_id=results["climb_id"],
            layout_id=results["layout_id"],
            name=results["name"],
            setter=results["setter"],
            description=results["description"],
            setter_angle=results["setter_angle"],
            hold_count=results["hold_count"],
            source=results["source"],
            graded_angles=angles,
        )

    def climb_stats(self, climb_id: str) -> Dict[int, Dict[str, Optional[float]]]:
        rows = self.conn.execute(
            "SELECT angle, display_difficulty, benchmark_difficulty, "
            "ascensionist_count, quality_average FROM climb_stats WHERE climb_id = ?",
            (climb_id,),
        ).fetchall()
        return {
            r["angle"]: {
                "display_difficulty": r["display_difficulty"],
                "benchmark_difficulty": r["benchmark_difficulty"],
                "ascensionist_count": r["ascensionist_count"],
                "quality_average": r["quality_average"],
            }
            for r in rows
        }

    # ---------------------------------------------------------- reference

    def difficulty_grades(self) -> Dict[int, str]:
        rows = self.conn.execute(
            "SELECT difficulty, boulder_name FROM difficulty_grades "
            "WHERE boulder_name IS NOT NULL"
        ).fetchall()
        return {int(r["difficulty"]): r["boulder_name"] for r in rows}

    def calibration_samples(
        self, min_ascents: int = 10, limit: int = 4000
    ) -> Iterator[Tuple[str, int, float]]:
        """(climb_id, angle, difficulty) rows worth fitting against.

        Filters on ascent count: a climb with two ascents has a community
        grade that is essentially noise.
        """
        rows = self.conn.execute(
            """
            SELECT s.climb_id, s.angle, s.display_difficulty
            FROM climb_stats s
            JOIN climbs c ON c.climb_id = s.climb_id
            WHERE s.display_difficulty IS NOT NULL
              AND COALESCE(s.ascensionist_count, 0) >= ?
              AND c.is_listed = 1
            ORDER BY s.ascensionist_count DESC
            LIMIT ?
            """,
            (min_ascents, limit),
        ).fetchall()
        for r in rows:
            yield (r["climb_id"], int(r["angle"]), float(r["display_difficulty"]))

    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for table in ("layouts", "holds", "climbs", "climb_holds", "climb_stats"):
            out[table] = self.conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
        return out
