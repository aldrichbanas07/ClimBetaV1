"""Core hold vocabulary.

Coordinate system (shared by every layer, from ETL through to the SVG frontend):

    x  -> inches, increasing to the climber's right, origin at board left edge
    y  -> inches, increasing UP, origin at board bottom edge

This matches the raw ``holes.x`` / ``holes.y`` columns in the Kilter Board
app database, so real data flows in without a transform.

Wall angle is measured in degrees of *overhang from vertical*: 0 = vertical
slab-ish wall, 70 = steeply overhanging. This is the same convention the
Kilter Board app uses for its angle setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class HoldType(str, Enum):
    """Physical shape/graspability class of a hold.

    IMPORTANT: this is *not* present in the Kilter Board app database. The app
    only stores geometry (``holes``) and role (``placement_roles``). Hold type
    is supplied by our own classification layer -- see
    ``kilterbeta.etl.hold_types``. Anything downstream should treat UNKNOWN as
    "average hold" rather than as an error.
    """

    JUG = "jug"
    EDGE = "edge"
    CRIMP = "crimp"
    SLOPER = "sloper"
    PINCH = "pinch"
    POCKET = "pocket"
    FOOT_CHIP = "foot_chip"
    UNKNOWN = "unknown"


class HoldRole(str, Enum):
    """How a hold is used within one specific climb (Kilter's colour coding)."""

    START = "start"      # green  -- hands start here
    HAND = "hand"        # blue   -- intermediate hand hold
    FINISH = "finish"    # purple -- hands finish here
    FOOT = "foot"        # orange -- feet only
    ANY = "any"          # anything goes


class Limb(str, Enum):
    LH = "LH"
    RH = "RH"
    LF = "LF"
    RF = "RF"

    @property
    def is_hand(self) -> bool:
        return self in (Limb.LH, Limb.RH)

    @property
    def is_foot(self) -> bool:
        return self in (Limb.LF, Limb.RF)

    @property
    def is_left(self) -> bool:
        return self in (Limb.LH, Limb.LF)

    @property
    def opposite(self) -> "Limb":
        return _OPPOSITE[self]


_OPPOSITE = {
    Limb.LH: Limb.RH,
    Limb.RH: Limb.LH,
    Limb.LF: Limb.RF,
    Limb.RF: Limb.LF,
}

HANDS = (Limb.LH, Limb.RH)
FEET = (Limb.LF, Limb.RF)
ALL_LIMBS = (Limb.LH, Limb.RH, Limb.LF, Limb.RF)


# Standard Kilter ruleset: orange holds are feet-only; every other lit hold may
# also be stood on. Kept as data so alternative rulesets are a config change.
_HAND_OK = frozenset({HoldRole.START, HoldRole.HAND, HoldRole.FINISH, HoldRole.ANY})
_FOOT_OK = frozenset(
    {HoldRole.START, HoldRole.HAND, HoldRole.FINISH, HoldRole.FOOT, HoldRole.ANY}
)


def role_allows(role: HoldRole, limb: Limb) -> bool:
    """Whether ``limb`` is permitted on a hold with this role."""
    return role in (_HAND_OK if limb.is_hand else _FOOT_OK)


@dataclass(frozen=True)
class Hold:
    """One hold as used by a climb: geometry + type + role.

    ``hold_id`` is our stable internal id. For real Kilter data it is the
    ``placements.id``, which is what climb frames reference; for sample data it
    is a synthetic integer. ``placement_id`` / ``hole_id`` are kept so that a
    generated beta can be traced back to physical hardware on the board.
    """

    hold_id: int
    x: float
    y: float
    hold_type: HoldType = HoldType.UNKNOWN
    role: HoldRole = HoldRole.HAND
    placement_id: Optional[int] = None
    hole_id: Optional[int] = None
    name: Optional[str] = None
    # Rough graspable width in inches. Drives the "can two limbs match here?"
    # test and softens reach cost for big holds.
    size: float = 3.0

    @property
    def position(self):  # -> tuple[float, float]
        return (self.x, self.y)

    def allows(self, limb: Limb) -> bool:
        return role_allows(self.role, limb)

    @property
    def is_matchable(self) -> bool:
        """Big enough for two limbs at once."""
        return self.size >= 3.0 or self.hold_type in (HoldType.JUG, HoldType.SLOPER)
