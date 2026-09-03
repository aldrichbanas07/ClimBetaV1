import pytest

from kilterbeta.beta.search import BetaSearch, HandState, SearchConfig
from kilterbeta.domain.holds import HoldRole, HoldType, Limb
from kilterbeta.domain.moves import MoveKind

from .conftest import make_hold


def _run(holds, angle=40, **overrides):
    config = SearchConfig(angle=angle, **overrides)
    return BetaSearch(holds, config).run()


def test_empty_climb_rejected():
    with pytest.raises(ValueError):
        BetaSearch([], SearchConfig())


def test_duplicate_hold_id_rejected():
    holds = [make_hold(1, 0, 0), make_hold(1, 10, 10)]
    with pytest.raises(ValueError):
        BetaSearch(holds, SearchConfig())


def test_jug_ladder_solves_without_truncation(jug_ladder):
    result = _run(jug_ladder)
    assert not result.truncated
    assert result.moves
    hand_moves = [m for m in result.moves if m.limb.is_hand]
    assert hand_moves[-1].hold_id == 7  # the finish hold


def test_jug_ladder_ends_with_both_hands_matched_on_finish(jug_ladder):
    result = _run(jug_ladder)
    last_two_hands = [m for m in result.moves if m.limb.is_hand][-2:]
    assert {m.hold_id for m in last_two_hands} == {7}


def test_minimal_two_hold_climb_solves(two_hold_climb):
    result = _run(two_hold_climb)
    assert not result.truncated
    hand_moves = [m for m in result.moves if m.limb.is_hand]
    assert hand_moves[-1].hold_id == 2


def test_no_hand_holds_raises():
    holds = [make_hold(1, 0, 0, role=HoldRole.FOOT), make_hold(2, 10, 10, role=HoldRole.FOOT)]
    with pytest.raises(ValueError):
        BetaSearch(holds, SearchConfig()).run()


def test_steeper_angle_never_makes_a_jug_ladder_cheaper(jug_ladder):
    costs = [_run(jug_ladder, angle=a).total_cost for a in (0, 20, 40, 55, 70)]
    assert all(b >= a - 1e-6 for a, b in zip(costs, costs[1:]))


def test_moves_never_exceed_dynamic_reach_limit(jug_ladder):
    result = _run(jug_ladder)
    for m in result.moves:
        assert m.reach_utilisation <= 1.15  # dynamic_reach_factor default + slack


def test_feet_never_move_dynamically(jug_ladder):
    result = _run(jug_ladder, angle=60)
    for m in result.moves:
        if m.limb.is_foot:
            assert m.reach_utilisation <= 1.0 + 1e-6
            assert m.kind is not MoveKind.DYNAMIC


def test_greedy_strategy_also_reaches_finish(jug_ladder):
    result = _run(jug_ladder, strategy="greedy")
    hand_moves = [m for m in result.moves if m.limb.is_hand]
    assert hand_moves[-1].hold_id == 7


def test_uncrossed_start_is_used_when_available(jug_ladder):
    search = BetaSearch(jug_ladder, SearchConfig(angle=40))
    result = search.run()
    starts = [m for m in result.moves if m.kind is MoveKind.START]
    lh = next(m for m in starts if m.limb is Limb.LH)
    rh = next(m for m in starts if m.limb is Limb.RH)
    assert search.by_id[lh.hold_id].x <= search.by_id[rh.hold_id].x


def test_crimp_ladder_costs_more_than_jug_ladder_same_geometry():
    def climb(hold_type):
        return [
            make_hold(1, 40.0, 20.0, HoldType.JUG, HoldRole.START),
            make_hold(2, 60.0, 20.0, HoldType.JUG, HoldRole.START),
            make_hold(3, 44.0, 40.0, hold_type, HoldRole.HAND),
            make_hold(4, 58.0, 56.0, hold_type, HoldRole.HAND),
            make_hold(5, 50.0, 72.0, HoldType.JUG, HoldRole.FINISH),
            make_hold(10, 40.0, 2.0, HoldType.JUG, HoldRole.FOOT),
            make_hold(11, 60.0, 2.0, HoldType.JUG, HoldRole.FOOT),
            make_hold(12, 36.0, 28.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
            make_hold(13, 64.0, 44.0, HoldType.FOOT_CHIP, HoldRole.FOOT),
        ]

    jug_cost = _run(climb(HoldType.JUG)).total_cost
    crimp_cost = _run(climb(HoldType.CRIMP)).total_cost
    assert crimp_cost > jug_cost


def test_unreachable_finish_returns_partial_beta_with_warning():
    holds = [
        make_hold(1, 40.0, 10.0, HoldType.JUG, HoldRole.START),
        make_hold(2, 40.0, 300.0, HoldType.JUG, HoldRole.FINISH),  # miles away
    ]
    result = _run(holds)
    assert result.truncated
    assert any("finish" in w.lower() for w in result.warnings)
    assert result.moves  # still returns something


def test_expansion_cap_truncates_and_warns(jug_ladder):
    result = _run(jug_ladder, max_expansions=1)
    assert result.truncated
    assert any("cap" in w for w in result.warnings)


def test_hand_state_moved_updates_only_target_limb():
    state = HandState(lh=1, rh=2, last=None)
    nxt = state.moved(Limb.RH, 5)
    assert nxt.lh == 1
    assert nxt.rh == 5
    assert nxt.last is Limb.RH
