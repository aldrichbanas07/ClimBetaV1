"""Shared fixtures.

Tests import from the ``backend`` package directly (see ``pyproject.toml``'s
``pythonpath`` setting), so no installed package is required to run them.
"""

from __future__ import annotations

from typing import List

import pytest

from kilterbeta.domain.holds import Hold, HoldRole, HoldType


def make_hold(
    hold_id: int,
    x: float,
    y: float,
    hold_type: HoldType = HoldType.JUG,
    role: HoldRole = HoldRole.HAND,
    size: float = 3.0,
) -> Hold:
    return Hold(hold_id=hold_id, x=x, y=y, hold_type=hold_type, role=role, size=size)


@pytest.fixture
def jug_ladder() -> List[Hold]:
    """A simple, generously-spaced straight-up jug ladder.

    Exists as the "this must always work, fast, with a short obvious path"
    smoke test fixture.
    """
    holds = [
        make_hold(1, 40.0, 20.0, HoldType.JUG, HoldRole.START),
        make_hold(2, 60.0, 20.0, HoldType.JUG, HoldRole.START),
        make_hold(3, 44.0, 40.0, HoldType.JUG, HoldRole.HAND),
        make_hold(4, 58.0, 56.0, HoldType.JUG, HoldRole.HAND),
        make_hold(5, 44.0, 72.0, HoldType.JUG, HoldRole.HAND),
        make_hold(6, 58.0, 88.0, HoldType.JUG, HoldRole.HAND),
        make_hold(7, 50.0, 104.0, HoldType.JUG, HoldRole.FINISH),
        make_hold(10, 40.0, 2.0, HoldType.JUG, HoldRole.FOOT),
        make_hold(11, 60.0, 2.0, HoldType.JUG, HoldRole.FOOT),
        make_hold(12, 36.0, 28.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
        make_hold(13, 64.0, 44.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
        make_hold(14, 36.0, 60.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
        make_hold(15, 64.0, 76.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
    ]
    return holds


@pytest.fixture
def two_hold_climb() -> List[Hold]:
    """Degenerate minimal climb: one match-start hold, one finish hold.

    With only two holds and no feet, this is inherently a single reach, so the
    gap is kept comfortably inside a no-feet dynamic reach (rather than
    exercising the reach *limit*, which belongs to the search/body tests).
    """
    return [
        make_hold(1, 40.0, 10.0, HoldType.JUG, HoldRole.START),
        make_hold(2, 40.0, 34.0, HoldType.JUG, HoldRole.FINISH),
    ]
