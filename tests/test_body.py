import pytest

from kilterbeta.beta.body import BodyModel, hip_hand_weight, solve_stance, support_triangle_area
from kilterbeta.domain.holds import Limb


def test_body_model_defaults_scale_with_height():
    short = BodyModel(height=60.0)
    tall = BodyModel(height=76.0)
    assert short.arm_length < tall.arm_length
    assert short.leg_length < tall.leg_length
    assert short.ape_span < tall.ape_span


def test_explicit_overrides_win_over_height_scaling():
    body = BodyModel(height=69.0, arm_length_in=30.0)
    assert body.arm_length == 30.0
    # leg_length still derives from height since it was not overridden.
    assert body.leg_length == pytest.approx(0.540 * 69.0)


def test_max_reach_uses_correct_segment():
    body = BodyModel()
    assert body.max_reach(Limb.LH) == body.arm_length
    assert body.max_reach(Limb.RF) == body.leg_length


def test_hip_hand_weight_increases_with_angle():
    assert hip_hand_weight(0) < hip_hand_weight(35) < hip_hand_weight(70)


def test_solve_stance_hip_between_hands_and_feet_on_vertical_wall():
    body = BodyModel()
    contacts = {
        Limb.LH: (30.0, 60.0),
        Limb.RH: (50.0, 60.0),
        Limb.LF: (32.0, 20.0),
        Limb.RF: (48.0, 20.0),
    }
    stance = solve_stance(contacts, body, angle_deg=0.0)
    assert 20.0 < stance.hip[1] < 60.0
    assert 25.0 < stance.hip[0] < 55.0


def test_solve_stance_hip_rides_up_as_angle_steepens():
    # Hands well above the feet, as on a stretched-out overhang move: this is
    # the geometry where "hips move toward the hands as angle increases"
    # actually produces a higher hip, rather than a compressed stance where
    # the hand- and foot-implied hip estimates are close enough that rounding
    # in the relaxation could go either way.
    body = BodyModel()
    contacts = {
        Limb.LH: (30.0, 80.0),
        Limb.RH: (50.0, 80.0),
        Limb.LF: (32.0, 10.0),
        Limb.RF: (48.0, 10.0),
    }
    vertical = solve_stance(contacts, body, angle_deg=0.0)
    steep = solve_stance(contacts, body, angle_deg=70.0)
    assert steep.hip[1] > vertical.hip[1]


def test_solve_stance_feet_cut_hangs_hip_below_hands():
    body = BodyModel()
    contacts = {Limb.LH: (30.0, 60.0), Limb.RH: (50.0, 60.0)}
    stance = solve_stance(contacts, body, angle_deg=40.0)
    assert stance.hip[1] < 60.0


def test_solve_stance_requires_at_least_one_contact():
    body = BodyModel()
    with pytest.raises(ValueError):
        solve_stance({}, body, angle_deg=0.0)


def test_stance_origin_for_hands_uses_shoulders_not_hip():
    body = BodyModel()
    contacts = {
        Limb.LH: (30.0, 60.0),
        Limb.RH: (50.0, 60.0),
        Limb.LF: (32.0, 20.0),
        Limb.RF: (48.0, 20.0),
    }
    stance = solve_stance(contacts, body, angle_deg=20.0)
    assert stance.origin_for(Limb.LH) == stance.shoulder_left
    assert stance.origin_for(Limb.RH) == stance.shoulder_right
    assert stance.origin_for(Limb.LF) == stance.hip


def test_support_triangle_area_zero_for_fewer_than_three_points():
    assert support_triangle_area([]) == 0.0
    assert support_triangle_area([(0, 0)]) == 0.0
    assert support_triangle_area([(0, 0), (1, 1)]) == 0.0


def test_support_triangle_area_positive_for_real_triangle():
    area = support_triangle_area([(0, 0), (4, 0), (0, 3)])
    assert area == pytest.approx(6.0)


def test_support_triangle_area_larger_for_wider_stance():
    narrow = support_triangle_area([(0, 0), (1, 0), (0.5, 10)])
    wide = support_triangle_area([(0, 0), (10, 0), (5, 10)])
    assert wide > narrow
