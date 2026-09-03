"""Ingest a real Kilter Board app database.

Input is a copy of the app's own ``db.sqlite3`` (the file synced by the Kilter
app, also obtainable with the ``boardlib`` tool). We read it strictly
read-only and never write back.

Relevant source tables
----------------------
``holes``            physical hole positions: ``id, x, y, name``
``placements``       a mountable position on a layout: ``id, layout_id, hole_id,
                     set_id, default_placement_role_id``
``placement_roles``  role vocabulary: ``id, name, full_name, screen_color``
``sets``             physical hold sets, ``id, name`` (used for type hints)
``climbs``           ``uuid, layout_id, name, description, setter_username,
                     angle, frames, frames_count, is_listed, is_draft``
``climb_stats``      per-angle community grades: ``climb_uuid, angle,
                     display_difficulty, ascensionist_count, quality_average``
``difficulty_grades`` numeric difficulty -> ``boulder_name`` / ``route_name``

The ``frames`` string
---------------------
A climb's holds are encoded as a flat run of ``p<placement_id>r<role_id>``
tokens, e.g. ``p1123r15p1145r13``. ``placement_id`` is why we key our own
``holds.hold_id`` on ``placements.id``: it makes the mapping a no-op.

Multi-frame climbs (``frames_count > 1``) encode an animated sequence rather
than a single boulder problem and are skipped.

Schema drift
------------
The app's schema has changed across releases, so every query is built from the
columns that actually exist (``_columns``) rather than from a fixed list. A
missing optional column degrades that field to NULL instead of failing the
whole ingest.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from ..domain.holds import Hold, HoldRole, HoldType
from .hold_types import Classification, classify

FRAME_TOKEN = re.compile(r"p(\d+)r(\d+)")

#: Fallback role-id mapping for the standard Kilter layouts, used only when
#: ``placement_roles.name`` is missing or unrecognisable.
FALLBACK_ROLE_IDS: Dict[int, HoldRole] = {
    12: HoldRole.START,
    13: HoldRole.HAND,
    14: HoldRole.FINISH,
    15: HoldRole.FOOT,
}


def _columns(conn: sqlite3.Connection, table: str) -> Set[str]:
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {r[1] for r in rows}


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?", (table,)
    ).fetchone()
    return row is not None


def _pick(available: Set[str], *names: str) -> Optional[str]:
    """First of ``names`` that exists, else None."""
    for n in names:
        if n in available:
            return n
    return None


def _select(available: Set[str], mapping: Dict[str, Sequence[str]]) -> Tuple[str, Dict[str, bool]]:
    """Build a SELECT list, aliasing whichever column name is present.

    Absent optional columns are selected as NULL so downstream row access by
    alias always works.
    """
    parts: List[str] = []
    present: Dict[str, bool] = {}
    for alias, candidates in mapping.items():
        col = _pick(available, *candidates)
        if col:
            parts.append(f"{col} AS {alias}")
            present[alias] = True
        else:
            parts.append(f"NULL AS {alias}")
            present[alias] = False
    return ", ".join(parts), present


def parse_frames(frames: Optional[str]) -> List[Tuple[int, int]]:
    """``'p1123r15p1145r13'`` -> ``[(1123, 15), (1145, 13)]``."""
    if not frames:
        return []
    return [(int(p), int(r)) for p, r in FRAME_TOKEN.findall(frames)]


def role_from_name(name: Optional[str], role_id: Optional[int] = None) -> HoldRole:
    """Map a Kilter role to ours, preferring the database's own name."""
    if name:
        low = name.strip().lower()
        if "start" in low:
            return HoldRole.START
        if "finish" in low:
            return HoldRole.FINISH
        if "foot" in low:
            return HoldRole.FOOT
        if "hand" in low or "middle" in low:
            return HoldRole.HAND
    if role_id is not None and role_id in FALLBACK_ROLE_IDS:
        return FALLBACK_ROLE_IDS[role_id]
    return HoldRole.HAND


@dataclass
class KilterLayout:
    layout_id: int
    name: str
    product_name: Optional[str]
    n_placements: int


class KilterSource:
    """Read-only accessor over a Kilter app database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Kilter app database not found at {self.path}. "
                "Copy the app's db.sqlite3 there, or run the sample ETL instead."
            )
        self.conn = sqlite3.connect(f"file:{self.path.as_posix()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row

        for required in ("holes", "placements", "climbs"):
            if not _has_table(self.conn, required):
                raise ValueError(
                    f"{self.path} does not look like a Kilter app database "
                    f"(missing table '{required}')"
                )

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "KilterSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------- layouts

    def layouts(self) -> List[KilterLayout]:
        cols = _columns(self.conn, "layouts")
        name_col = _pick(cols, "name") or "id"
        product_join = ""
        product_sel = "NULL AS product_name"
        if _has_table(self.conn, "products") and "product_id" in cols:
            product_sel = "p.name AS product_name"
            product_join = "LEFT JOIN products p ON p.id = l.product_id"

        rows = self.conn.execute(
            f"""
            SELECT l.id AS layout_id, l.{name_col} AS name, {product_sel},
                   (SELECT COUNT(*) FROM placements pl WHERE pl.layout_id = l.id) AS n
            FROM layouts l
            {product_join}
            ORDER BY n DESC
            """
        ).fetchall()
        return [
            KilterLayout(
                layout_id=r["layout_id"],
                name=str(r["name"]),
                product_name=r["product_name"],
                n_placements=r["n"] or 0,
            )
            for r in rows
        ]

    def default_layout_id(self) -> int:
        """The layout with the most placements -- in practice the main board."""
        layouts = self.layouts()
        if not layouts:
            raise ValueError("no layouts found in the Kilter database")
        return layouts[0].layout_id

    def set_names(self) -> Dict[int, str]:
        if not _has_table(self.conn, "sets"):
            return {}
        cols = _columns(self.conn, "sets")
        name_col = _pick(cols, "name") or "id"
        rows = self.conn.execute(f"SELECT id, {name_col} AS name FROM sets").fetchall()
        return {r["id"]: str(r["name"]) for r in rows}

    def role_names(self) -> Dict[int, str]:
        if not _has_table(self.conn, "placement_roles"):
            return {}
        cols = _columns(self.conn, "placement_roles")
        name_col = _pick(cols, "name", "full_name")
        if not name_col:
            return {}
        rows = self.conn.execute(
            f"SELECT id, {name_col} AS name FROM placement_roles"
        ).fetchall()
        return {r["id"]: str(r["name"]) for r in rows}

    # --------------------------------------------------------------- holds

    def holds(
        self,
        layout_id: int,
        overrides: Optional[Dict[int, Classification]] = None,
    ) -> Tuple[List[Hold], Dict[int, str], Dict[int, int]]:
        """Every placement on a layout, as domain ``Hold`` objects.

        Returns ``(holds, type_sources, set_ids)``. ``role`` on each hold is the
        placement's *default* role; per-climb roles come from the frames string.
        """
        overrides = overrides or {}
        set_names = self.set_names()
        roles = self.role_names()

        placement_cols = _columns(self.conn, "placements")
        hole_cols = _columns(self.conn, "holes")

        set_col = "pl.set_id" if "set_id" in placement_cols else "NULL"
        role_col = (
            "pl.default_placement_role_id"
            if "default_placement_role_id" in placement_cols
            else "NULL"
        )
        hole_name = "h.name" if "name" in hole_cols else "NULL"

        rows = self.conn.execute(
            f"""
            SELECT pl.id AS placement_id, pl.hole_id AS hole_id,
                   {set_col} AS set_id, {role_col} AS role_id,
                   h.x AS x, h.y AS y, {hole_name} AS name
            FROM placements pl
            JOIN holes h ON h.id = pl.hole_id
            WHERE pl.layout_id = ?
            """,
            (layout_id,),
        ).fetchall()

        holds: List[Hold] = []
        type_sources: Dict[int, str] = {}
        set_ids: Dict[int, int] = {}

        for r in rows:
            if r["x"] is None or r["y"] is None:
                continue
            pid = int(r["placement_id"])
            role_id = r["role_id"]
            default_role = role_from_name(
                roles.get(role_id) if role_id is not None else None,
                int(role_id) if role_id is not None else None,
            )
            set_name = set_names.get(r["set_id"]) if r["set_id"] is not None else None

            cls = classify(
                pid,
                overrides,
                set_name=set_name,
                default_role=default_role.value,
            )
            holds.append(
                Hold(
                    hold_id=pid,           # keyed on placements.id; see module docstring
                    x=float(r["x"]),
                    y=float(r["y"]),
                    hold_type=cls.hold_type,
                    role=default_role,
                    placement_id=pid,
                    hole_id=int(r["hole_id"]) if r["hole_id"] is not None else None,
                    name=r["name"],
                    size=cls.size,
                )
            )
            type_sources[pid] = cls.source
            if r["set_id"] is not None:
                set_ids[pid] = int(r["set_id"])

        return holds, type_sources, set_ids

    # -------------------------------------------------------------- climbs

    def climbs(
        self,
        layout_id: int,
        limit: Optional[int] = None,
        min_ascents: int = 0,
        listed_only: bool = True,
    ) -> Iterator[Dict[str, object]]:
        """Yield climbs with their parsed hold/role list.

        ``min_ascents`` filters on the best per-angle ascent count, which is the
        cheap way to skip the long tail of unrepeated climbs.
        """
        cols = _columns(self.conn, "climbs")
        select_list, _ = _select(
            cols,
            {
                "climb_id": ("uuid", "id"),
                "name": ("name",),
                "description": ("description",),
                "setter": ("setter_username", "setter_name"),
                "angle": ("angle",),
                "frames": ("frames",),
                "frames_count": ("frames_count",),
                "is_listed": ("is_listed",),
                "is_draft": ("is_draft",),
            },
        )

        where = ["layout_id = ?"]
        params: List[object] = [layout_id]
        if listed_only and "is_listed" in cols:
            where.append("is_listed = 1")
        if "is_draft" in cols:
            where.append("COALESCE(is_draft, 0) = 0")
        if "frames_count" in cols:
            where.append("COALESCE(frames_count, 1) = 1")

        sql = f"SELECT {select_list} FROM climbs WHERE {' AND '.join(where)}"
        if limit:
            sql += f" LIMIT {int(limit)}"

        roles = self.role_names()
        stats_by_climb = self._all_stats(layout_id) if min_ascents > 0 else None

        for r in self.conn.execute(sql, params):
            climb_id = r["climb_id"]
            if climb_id is None:
                continue
            climb_id = str(climb_id)

            if stats_by_climb is not None:
                per_angle = stats_by_climb.get(climb_id) or {}
                best = max(
                    (d.get("ascensionist_count") or 0 for d in per_angle.values()),
                    default=0,
                )
                if best < min_ascents:
                    continue

            frame_holds: List[Tuple[int, HoldRole]] = []
            for placement_id, role_id in parse_frames(r["frames"]):
                frame_holds.append(
                    (placement_id, role_from_name(roles.get(role_id), role_id))
                )
            if not frame_holds:
                continue

            yield {
                "climb_id": climb_id,
                "name": str(r["name"] or "(unnamed)"),
                "description": r["description"],
                "setter": r["setter"],
                "setter_angle": int(r["angle"]) if r["angle"] is not None else None,
                "frames": r["frames"],
                "holds": frame_holds,
            }

    def _all_stats(self, layout_id: int) -> Dict[str, Dict[int, Dict[str, Optional[float]]]]:
        out: Dict[str, Dict[int, Dict[str, Optional[float]]]] = {}
        for climb_id, angle, data in self.stats(layout_id):
            out.setdefault(climb_id, {})[angle] = data
        return out

    def stats(
        self, layout_id: Optional[int] = None
    ) -> Iterator[Tuple[str, int, Dict[str, Optional[float]]]]:
        """Per-angle community difficulty rows."""
        if not _has_table(self.conn, "climb_stats"):
            return
        cols = _columns(self.conn, "climb_stats")
        key_col = _pick(cols, "climb_uuid", "climb_id")
        if not key_col or "angle" not in cols:
            return

        # Build the select list explicitly, qualified with the table alias.
        wanted = {
            "display_difficulty": ("display_difficulty", "difficulty_average"),
            "benchmark_difficulty": ("benchmark_difficulty",),
            "ascensionist_count": ("ascensionist_count",),
            "quality_average": ("quality_average",),
        }
        parts = [f"s.{key_col} AS climb_id", "s.angle AS angle"]
        for alias, candidates in wanted.items():
            col = _pick(cols, *candidates)
            parts.append(f"s.{col} AS {alias}" if col else f"NULL AS {alias}")

        join = ""
        where = ""
        params: List[object] = []
        # Only join to filter by layout when climbs.uuid exists to join on.
        if layout_id is not None and "uuid" in _columns(self.conn, "climbs"):
            join = f"JOIN climbs c ON c.uuid = s.{key_col}"
            where = "WHERE c.layout_id = ?"
            params.append(layout_id)

        sql = f"SELECT {', '.join(parts)} FROM climb_stats s {join} {where}"

        for r in self.conn.execute(sql, params):
            if r["climb_id"] is None or r["angle"] is None:
                continue
            yield (
                str(r["climb_id"]),
                int(r["angle"]),
                {
                    "display_difficulty": r["display_difficulty"],
                    "benchmark_difficulty": r["benchmark_difficulty"],
                    "ascensionist_count": r["ascensionist_count"],
                    "quality_average": r["quality_average"],
                },
            )

    def difficulty_grades(self) -> Dict[int, Tuple[Optional[str], Optional[str]]]:
        if not _has_table(self.conn, "difficulty_grades"):
            return {}
        cols = _columns(self.conn, "difficulty_grades")
        if "difficulty" not in cols:
            return {}
        boulder = _pick(cols, "boulder_name")
        route = _pick(cols, "route_name")
        sel = ", ".join(
            [
                "difficulty",
                f"{boulder} AS boulder_name" if boulder else "NULL AS boulder_name",
                f"{route} AS route_name" if route else "NULL AS route_name",
            ]
        )
        rows = self.conn.execute(f"SELECT {sel} FROM difficulty_grades").fetchall()
        return {int(r["difficulty"]): (r["boulder_name"], r["route_name"]) for r in rows}
