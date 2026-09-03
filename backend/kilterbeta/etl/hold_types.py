"""Hold-type classification.

**The Kilter Board app database does not record hold type.** It stores hole
geometry (``holes``), which physical set a placement belongs to
(``placements.set_id``), and how a hold is used within a climb
(``placement_roles``: start / hand / finish / foot). Whether a given hold is a
crimp, a jug or a sloper is simply not in there.

Since hold type is the main input to our difficulty model, this module is the
seam where that information is supplied. Precedence, highest first:

1. ``data/hold_types.csv`` -- hand-curated overrides, keyed by hold id.
   Authoritative, marked ``hold_type_source='manual'``.
2. Set-name keywords -- Kilter's screw-on sets are the small footholds, so a
   placement from one is a ``foot_chip``. Marked ``'heuristic'``.
3. The placement's default role -- a foot-only placement is a ``foot_chip``.
   Marked ``'heuristic'``.
4. ``unknown``, marked ``'default'``. The difficulty model treats unknown as an
   average hold rather than as an error, so the tool degrades gracefully.

Generate a starter CSV with::

    python -m kilterbeta.etl.cli export-hold-types --out data/hold_types.csv

then fill in the ``hold_type`` column. That is the intended path from "real
Kilter data ingested" to "real Kilter data with real hold types".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from ..domain.holds import HoldType

#: Typical graspable width in inches, used for match feasibility and to soften
#: reach cost on big holds.
DEFAULT_SIZES: Dict[HoldType, float] = {
    HoldType.JUG: 5.0,
    HoldType.SLOPER: 5.5,
    HoldType.PINCH: 3.5,
    HoldType.EDGE: 3.0,
    HoldType.POCKET: 2.5,
    HoldType.CRIMP: 1.5,
    HoldType.FOOT_CHIP: 1.2,
    HoldType.UNKNOWN: 3.0,
}

#: Accepted spellings in the override CSV, so curating it is forgiving.
ALIASES: Dict[str, HoldType] = {
    "jug": HoldType.JUG, "hold": HoldType.JUG, "bucket": HoldType.JUG,
    "edge": HoldType.EDGE, "rail": HoldType.EDGE, "incut": HoldType.EDGE,
    "crimp": HoldType.CRIMP, "crimper": HoldType.CRIMP, "chip": HoldType.CRIMP,
    "sloper": HoldType.SLOPER, "slope": HoldType.SLOPER, "ball": HoldType.SLOPER,
    "pinch": HoldType.PINCH,
    "pocket": HoldType.POCKET, "mono": HoldType.POCKET, "hole": HoldType.POCKET,
    "foot": HoldType.FOOT_CHIP, "foot_chip": HoldType.FOOT_CHIP,
    "footchip": HoldType.FOOT_CHIP, "jib": HoldType.FOOT_CHIP,
    "screw_on": HoldType.FOOT_CHIP, "screwon": HoldType.FOOT_CHIP,
    "unknown": HoldType.UNKNOWN, "": HoldType.UNKNOWN,
}

#: Substrings in a Kilter set name that reliably imply a type.
SET_NAME_HINTS = (
    ("screw", HoldType.FOOT_CHIP),
    ("jib", HoldType.FOOT_CHIP),
    ("foot", HoldType.FOOT_CHIP),
)


def parse_hold_type(raw: Optional[str]) -> HoldType:
    """Lenient string -> HoldType. Unrecognised values become UNKNOWN."""
    if raw is None:
        return HoldType.UNKNOWN
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if key in ALIASES:
        return ALIASES[key]
    try:
        return HoldType(key)
    except ValueError:
        return HoldType.UNKNOWN


@dataclass(frozen=True)
class Classification:
    hold_type: HoldType
    size: float
    source: str  # 'manual' | 'heuristic' | 'default'


def load_overrides(path: Path) -> Dict[int, Classification]:
    """Read ``hold_types.csv``.

    Expected columns: ``hold_id``, ``hold_type``, and optionally ``size``.
    Extra columns (name, set_name, notes) are ignored, so the file exported for
    curation can be edited and fed straight back.
    """
    path = Path(path)
    if not path.exists():
        return {}

    out: Dict[int, Classification] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_id = (row.get("hold_id") or "").strip()
            if not raw_id or raw_id.startswith("#"):
                continue
            try:
                hold_id = int(raw_id)
            except ValueError:
                continue

            raw_type = (row.get("hold_type") or "").strip()
            if not raw_type:
                continue  # left blank during curation; fall through to heuristics
            hold_type = parse_hold_type(raw_type)

            size_raw = (row.get("size") or "").strip()
            try:
                size = float(size_raw) if size_raw else DEFAULT_SIZES[hold_type]
            except ValueError:
                size = DEFAULT_SIZES[hold_type]

            out[hold_id] = Classification(hold_type, size, "manual")
    return out


def classify(
    hold_id: int,
    overrides: Dict[int, Classification],
    set_name: Optional[str] = None,
    default_role: Optional[str] = None,
) -> Classification:
    """Resolve one hold's type by the precedence documented above."""
    override = overrides.get(hold_id)
    if override is not None:
        return override

    if set_name:
        low = set_name.lower()
        for needle, hold_type in SET_NAME_HINTS:
            if needle in low:
                return Classification(hold_type, DEFAULT_SIZES[hold_type], "heuristic")

    if default_role and default_role.lower() == "foot":
        return Classification(HoldType.FOOT_CHIP, DEFAULT_SIZES[HoldType.FOOT_CHIP], "heuristic")

    return Classification(HoldType.UNKNOWN, DEFAULT_SIZES[HoldType.UNKNOWN], "default")


def write_template(path: Path, rows) -> int:
    """Write a curation template: one row per hold, ``hold_type`` left blank.

    ``rows`` yields dicts with at least ``hold_id``; ``name``, ``set_name``,
    ``x``, ``y`` and ``current_type`` are included when available to make
    hand-labelling tractable.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["hold_id", "hold_type", "size", "name", "set_name", "x", "y", "current_type"]
    n = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n
