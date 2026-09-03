import pytest

from kilterbeta.beta.difficulty import (
    DifficultyModel,
    foot_load_share,
    hand_load_share,
    steepness,
)
from kilterbeta.domain.holds import HoldType, Limb


def test_steepness_monotonic_and_bounded():
    values = [steepness(a) for a in range(0, 71, 5)]
    assert values[0] == pytest.approx(0.0, abs=1e-9)
    assert values[-1] == pytest.approx(1.0, abs=1e-9)
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_hand_load_share_increases_with_angle():
    assert hand_load_share(0) < hand_load_share(40) < hand_load_share(70)


def test_foot_load_share_decreases_with_angle():
    assert foot_load_share(0) > foot_load_share(40) > foot_load_share(70)


@pytest.mark.parametrize("angle", [0, 25, 45, 70])
def test_jug_always_cheaper_than_crimp_and_sloper(angle):
    model = DifficultyModel()
    jug = model.hold_cost(HoldType.JUG, Limb.RH, angle)
    crimp = model.hold_cost(HoldType.CRIMP, Limb.RH, angle)
    sloper = model.hold_cost(HoldType.SLOPER, Limb.RH, angle)
    assert jug < crimp
    assert jug < sloper


def test_sloper_degrades_faster_than_jug_with_angle():
    model = DifficultyModel()
    jug_ratio = model.hold_cost(HoldType.JUG, Limb.RH, 70) / model.hold_cost(
        HoldType.JUG, Limb.RH, 0
    )
    sloper_ratio = model.hold_cost(HoldType.SLOPER, Limb.RH, 70) / model.hold_cost(
        HoldType.SLOPER, Limb.RH, 0
    )
    assert sloper_ratio > jug_ratio


def test_unknown_hold_type_is_priced_as_average():
    model = DifficultyModel()
    costs = [
        model.hold_cost(t, Limb.RH, 40)
        for t in HoldType
        if t not in (HoldType.UNKNOWN, HoldType.FOOT_CHIP)
    ]
    unknown = model.hold_cost(HoldType.UNKNOWN, Limb.RH, 40)
    assert min(costs) <= unknown <= max(costs)


def test_body_tension_cost_penalises_both_over_extension_and_scrunching():
    model = DifficultyModel()
    span = 40.0
    ideal_sep = model.ideal_extension * span
    relaxed = model.body_tension_cost(40, ideal_sep, span, feet_on=2)
    stretched = model.body_tension_cost(40, span * 1.3, span, feet_on=2)
    scrunched = model.body_tension_cost(40, -span * 0.3, span, feet_on=2)
    assert stretched > relaxed
    assert scrunched > relaxed


def test_cutting_feet_increases_body_tension_cost():
    model = DifficultyModel()
    span = 40.0
    two_feet = model.body_tension_cost(40, span * 0.5, span, feet_on=2)
    one_foot = model.body_tension_cost(40, span * 0.5, span, feet_on=1)
    no_feet = model.body_tension_cost(40, span * 0.5, span, feet_on=0)
    assert no_feet > one_foot > two_feet


def test_reach_cost_grows_convexly():
    model = DifficultyModel()
    low = model.reach_cost(0.3)
    mid = model.reach_cost(0.6)
    high = model.reach_cost(0.9)
    assert 0 <= low < mid < high
    # Convexity: the increment from mid->high should exceed low->mid.
    assert (high - mid) > (mid - low)
