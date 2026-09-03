"""A synthetic, Kilter-shaped sample board.

The real Kilter hold layout is proprietary and ships inside the app database,
so this module builds a board with the same *geometry and feel* -- a 12x12ft
wall, an 8 inch staggered bolt-on grid, a kickboard row, interleaved screw-on
footholds -- and populates it with a plausible mix of hold types.

It is deliberately generated rather than checked in as a big CSV so that:
  * it is reproducible (a fixed hash, not an RNG seed, decides every type);
  * it is small to read and easy to reason about;
  * swapping in real data is a source swap, not a schema change.

Everything here writes through the same ``loader`` as the real Kilter ingest,
so the two paths cannot drift apart. Rows are tagged ``source='sample'`` and
the layout is named accordingly -- nothing in the API pretends this is real
Kilter data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..domain.holds import Hold, HoldRole, HoldType
from .hold_types import DEFAULT_SIZES

SAMPLE_LAYOUT_ID = 9001
SAMPLE_LAYOUT_NAME = "Sample 12x12 (synthetic)"

# --- Board geometry, in inches --------------------------------------------
BOARD_WIDTH = 144.0
BOARD_HEIGHT = 156.0

HAND_X0, HAND_X1, HAND_DX = 8.0, 136.0, 8.0
HAND_Y0, HAND_Y1, HAND_DY = 20.0, 148.0, 8.0
KICKBOARD_Y = 8.0

#: Hold-type mix for bolt-on hand holds, roughly matching how a Kilter board
#: feels: lots of edges and crimps, a decent number of jugs, fewer slopers.
TYPE_MIX: Sequence[Tuple[HoldType, float]] = (
    (HoldType.EDGE, 0.24),
    (HoldType.CRIMP, 0.22),
    (HoldType.JUG, 0.17),
    (HoldType.PINCH, 0.13),
    (HoldType.SLOPER, 0.10),
    (HoldType.POCKET, 0.08),
    (HoldType.FOOT_CHIP, 0.06),
)


def _unit_hash(*parts) -> float:
    """Deterministic float in [0, 1) from arbitrary inputs.

    Used instead of ``random`` so the sample board is identical on every
    machine and every Python version.
    """
    raw = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(raw).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _pick_type(u: float) -> HoldType:
    acc = 0.0
    for hold_type, weight in TYPE_MIX:
        acc += weight
        if u < acc:
            return hold_type
    return TYPE_MIX[-1][0]


@dataclass
class SampleClimbSpec:
    """A hand-authored climb, described by intent rather than by hold ids.

    ``waypoints`` are (x, y, preferred_type) hints; the builder snaps each to
    the nearest matching hold on the generated board. That keeps the specs
    readable and stops them breaking if the board layout is tweaked.
    """

    climb_id: str
    name: str
    setter: str
    description: str
    setter_angle: int
    start: Sequence[Tuple[float, float, Optional[HoldType]]]
    hands: Sequence[Tuple[float, float, Optional[HoldType]]]
    finish: Tuple[float, float, Optional[HoldType]]
    feet: Sequence[Tuple[float, float, Optional[HoldType]]]


def build_board() -> List[Hold]:
    """All holds on the synthetic board."""
    holds: List[Hold] = []
    hold_id = SAMPLE_LAYOUT_ID * 100  # keeps sample ids clear of real placement ids

    # Kickboard: a row of positive footholds along the bottom.
    x = 12.0
    col = 0
    while x <= BOARD_WIDTH - 12.0:
        hold_id += 1
        holds.append(
            Hold(
                hold_id=hold_id,
                x=x,
                y=KICKBOARD_Y,
                hold_type=HoldType.JUG,
                role=HoldRole.FOOT,
                placement_id=hold_id,
                hole_id=hold_id,
                name=f"kick-{col}",
                size=DEFAULT_SIZES[HoldType.JUG],
            )
        )
        x += 12.0
        col += 1

    # Main bolt-on grid, staggered every other row like the real board.
    row = 0
    y = HAND_Y0
    while y <= HAND_Y1:
        offset = 0.0 if row % 2 == 0 else HAND_DX / 2.0
        x = HAND_X0 + offset
        c = 0
        while x <= HAND_X1:
            u = _unit_hash("hand", row, c)
            hold_type = _pick_type(u)
            hold_id += 1
            holds.append(
                Hold(
                    hold_id=hold_id,
                    x=x,
                    y=y,
                    hold_type=hold_type,
                    role=HoldRole.HAND,
                    placement_id=hold_id,
                    hole_id=hold_id,
                    name=f"r{row}c{c}",
                    size=DEFAULT_SIZES[hold_type],
                )
            )
            x += HAND_DX
            c += 1
        row += 1
        y += HAND_DY

    # Screw-on footholds, sparsely interleaved between the grid rows.
    row = 0
    y = HAND_Y0 + HAND_DY / 2.0
    while y <= HAND_Y1:
        x = HAND_X0 + HAND_DX / 4.0
        c = 0
        while x <= HAND_X1:
            if _unit_hash("foot", row, c) < 0.34:
                hold_id += 1
                holds.append(
                    Hold(
                        hold_id=hold_id,
                        x=x,
                        y=y,
                        hold_type=HoldType.FOOT_CHIP,
                        role=HoldRole.FOOT,
                        placement_id=hold_id,
                        hole_id=hold_id,
                        name=f"f{row}c{c}",
                        size=DEFAULT_SIZES[HoldType.FOOT_CHIP],
                    )
                )
            x += HAND_DX
            c += 1
        row += 1
        y += HAND_DY

    return holds


def _nearest(
    board: Sequence[Hold],
    x: float,
    y: float,
    preferred: Optional[HoldType],
    used: set,
    hand_usable: bool,
) -> Optional[Hold]:
    """Snap a waypoint to the nearest unused hold, preferring a given type."""

    def score(h: Hold) -> float:
        d = ((h.x - x) ** 2 + (h.y - y) ** 2) ** 0.5
        # A type mismatch costs the equivalent of 40 inches, so type wins
        # unless the nearest match is absurdly far away.
        penalty = 0.0 if (preferred is None or h.hold_type is preferred) else 40.0
        return d + penalty

    candidates = [
        h
        for h in board
        if h.hold_id not in used
        and (not hand_usable or h.hold_type is not HoldType.FOOT_CHIP)
    ]
    if not candidates:
        return None
    return min(candidates, key=score)


def sample_climbs() -> List[SampleClimbSpec]:
    """Six hand-authored demo climbs spanning the difficulty model's range.

    They exist to exercise different parts of the cost model -- big reaches,
    bad hold types, traverses, foot-dependent sequences -- so the demo shows
    the angle slider actually changing something.
    """
    return [
        SampleClimbSpec(
            climb_id="sample-001",
            name="Jug Ladder",
            setter="demo",
            description="Straight up on positive holds. Should stay easy at every angle.",
            setter_angle=20,
            start=[(60.0, 28.0, HoldType.JUG), (84.0, 28.0, HoldType.JUG)],
            hands=[
                (64.0, 52.0, HoldType.JUG), (80.0, 68.0, HoldType.JUG),
                (64.0, 84.0, HoldType.JUG), (80.0, 100.0, HoldType.JUG),
                (68.0, 116.0, HoldType.JUG),
            ],
            finish=(72.0, 140.0, HoldType.JUG),
            feet=[
                (60.0, 8.0, None), (84.0, 8.0, None),
                (56.0, 36.0, HoldType.FOOT_CHIP), (88.0, 44.0, HoldType.FOOT_CHIP),
                (56.0, 68.0, HoldType.FOOT_CHIP), (88.0, 76.0, HoldType.FOOT_CHIP),
                (60.0, 100.0, HoldType.FOOT_CHIP), (84.0, 108.0, HoldType.FOOT_CHIP),
            ],
        ),
        SampleClimbSpec(
            climb_id="sample-002",
            name="Crimp Ladder",
            setter="demo",
            description="Same line, small edges. The angle slider should bite hard here.",
            setter_angle=40,
            start=[(60.0, 28.0, HoldType.CRIMP), (84.0, 28.0, HoldType.CRIMP)],
            hands=[
                (64.0, 52.0, HoldType.CRIMP), (80.0, 68.0, HoldType.CRIMP),
                (64.0, 84.0, HoldType.CRIMP), (80.0, 100.0, HoldType.CRIMP),
                (68.0, 116.0, HoldType.CRIMP),
            ],
            finish=(72.0, 140.0, HoldType.JUG),
            feet=[
                (60.0, 8.0, None), (84.0, 8.0, None),
                (56.0, 36.0, HoldType.FOOT_CHIP), (88.0, 44.0, HoldType.FOOT_CHIP),
                (56.0, 68.0, HoldType.FOOT_CHIP), (88.0, 76.0, HoldType.FOOT_CHIP),
                (60.0, 100.0, HoldType.FOOT_CHIP), (84.0, 108.0, HoldType.FOOT_CHIP),
            ],
        ),
        SampleClimbSpec(
            climb_id="sample-003",
            name="Sloper Traverse",
            setter="demo",
            description="Rounded holds, rising leftward. Slopers are the most angle-sensitive type.",
            setter_angle=25,
            start=[(96.0, 28.0, HoldType.SLOPER)],
            hands=[
                (80.0, 44.0, HoldType.SLOPER), (60.0, 56.0, HoldType.SLOPER),
                (44.0, 72.0, HoldType.SLOPER), (32.0, 92.0, HoldType.SLOPER),
                (44.0, 112.0, HoldType.SLOPER),
            ],
            finish=(56.0, 132.0, HoldType.JUG),
            feet=[
                (96.0, 8.0, None), (72.0, 8.0, None),
                (88.0, 28.0, HoldType.FOOT_CHIP), (68.0, 40.0, HoldType.FOOT_CHIP),
                (48.0, 56.0, HoldType.FOOT_CHIP), (36.0, 76.0, HoldType.FOOT_CHIP),
                (28.0, 96.0, HoldType.FOOT_CHIP),
            ],
        ),
        SampleClimbSpec(
            climb_id="sample-004",
            name="Big Moves",
            setter="demo",
            description="Good holds, far apart. Tests the reach model rather than the hold model.",
            setter_angle=45,
            # Spans here are deliberately near the top of the static range
            # (roughly 26-32 in between consecutive hand holds) rather than
            # beyond it -- a gap no body model can bridge just fails to plan.
            start=[(56.0, 24.0, HoldType.JUG), (88.0, 24.0, HoldType.JUG)],
            hands=[
                (72.0, 52.0, HoldType.JUG),
                (48.0, 80.0, HoldType.JUG),
                (84.0, 104.0, HoldType.JUG),
                (60.0, 128.0, HoldType.JUG),
            ],
            finish=(76.0, 148.0, HoldType.JUG),
            feet=[
                (56.0, 8.0, None), (88.0, 8.0, None),
                (64.0, 36.0, HoldType.FOOT_CHIP), (80.0, 36.0, HoldType.FOOT_CHIP),
                (56.0, 60.0, HoldType.FOOT_CHIP), (80.0, 76.0, HoldType.FOOT_CHIP),
                (52.0, 100.0, HoldType.FOOT_CHIP), (76.0, 112.0, HoldType.FOOT_CHIP),
            ],
        ),
        SampleClimbSpec(
            climb_id="sample-005",
            name="Pinch Power",
            setter="demo",
            description="Pinches on a steep wall, minimal feet. Body tension does the work.",
            setter_angle=55,
            start=[(64.0, 32.0, HoldType.PINCH), (88.0, 36.0, HoldType.PINCH)],
            hands=[
                (72.0, 60.0, HoldType.PINCH), (88.0, 84.0, HoldType.PINCH),
                (68.0, 104.0, HoldType.PINCH), (84.0, 124.0, HoldType.PINCH),
            ],
            finish=(72.0, 144.0, HoldType.JUG),
            feet=[
                (72.0, 8.0, None),
                (80.0, 44.0, HoldType.FOOT_CHIP), (68.0, 80.0, HoldType.FOOT_CHIP),
            ],
        ),
        SampleClimbSpec(
            climb_id="sample-006",
            name="Technical Feet",
            setter="demo",
            description="Poor hands, generous feet. Rewards footwork and drop-knees (phase 2 territory).",
            setter_angle=15,
            start=[(52.0, 28.0, HoldType.EDGE), (76.0, 28.0, HoldType.EDGE)],
            hands=[
                (44.0, 52.0, HoldType.EDGE), (68.0, 64.0, HoldType.POCKET),
                (48.0, 84.0, HoldType.EDGE), (76.0, 96.0, HoldType.POCKET),
                (56.0, 116.0, HoldType.EDGE),
            ],
            finish=(64.0, 136.0, HoldType.JUG),
            feet=[
                (48.0, 8.0, None), (72.0, 8.0, None),
                (44.0, 32.0, HoldType.FOOT_CHIP), (68.0, 36.0, HoldType.FOOT_CHIP),
                (40.0, 60.0, HoldType.FOOT_CHIP), (64.0, 60.0, HoldType.FOOT_CHIP),
                (44.0, 88.0, HoldType.FOOT_CHIP), (72.0, 88.0, HoldType.FOOT_CHIP),
                (52.0, 112.0, HoldType.FOOT_CHIP), (76.0, 112.0, HoldType.FOOT_CHIP),
            ],
        ),
    ]


def resolve_climb(
    spec: SampleClimbSpec, board: Sequence[Hold]
) -> List[Tuple[int, HoldRole]]:
    """Snap a spec's waypoints onto real board holds -> (hold_id, role) pairs."""
    used: set = set()
    out: List[Tuple[int, HoldRole]] = []

    for x, y, t in spec.start:
        h = _nearest(board, x, y, t, used, hand_usable=True)
        if h:
            used.add(h.hold_id)
            out.append((h.hold_id, HoldRole.START))

    for x, y, t in spec.hands:
        h = _nearest(board, x, y, t, used, hand_usable=True)
        if h:
            used.add(h.hold_id)
            out.append((h.hold_id, HoldRole.HAND))

    fx, fy, ft = spec.finish
    h = _nearest(board, fx, fy, ft, used, hand_usable=True)
    if h:
        used.add(h.hold_id)
        out.append((h.hold_id, HoldRole.FINISH))

    for x, y, t in spec.feet:
        h = _nearest(board, x, y, t, used, hand_usable=False)
        if h:
            used.add(h.hold_id)
            out.append((h.hold_id, HoldRole.FOOT))

    return out


def synthetic_stats(spec: SampleClimbSpec, hold_ids: Sequence[int]) -> Dict[int, float]:
    """Plausible per-angle 'community' difficulty for the sample climbs.

    Purely so the calibration code path is exercisable end to end without the
    real database. These numbers are invented, and everything that consumes
    them reports ``calibrated`` alongside so they are never mistaken for real
    community grades.
    """
    base = {
        "sample-001": 12.0, "sample-002": 19.0, "sample-003": 18.0,
        "sample-004": 20.0, "sample-005": 23.0, "sample-006": 16.0,
    }.get(spec.climb_id, 17.0)
    out: Dict[int, float] = {}
    for angle in range(0, 75, 5):
        # Steeper is harder, and harder climbs steepen faster.
        out[angle] = round(base + (angle - 20) * (0.055 + base / 900.0), 2)
    return out
