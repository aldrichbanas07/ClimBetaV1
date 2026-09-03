from kilterbeta.domain.holds import Hold, HoldRole, HoldType, Limb, role_allows


def test_limb_opposite_is_involutive():
    for limb in Limb:
        assert limb.opposite.opposite is limb


def test_limb_hand_foot_partition():
    for limb in Limb:
        assert limb.is_hand != limb.is_foot


def test_foot_only_hold_blocks_hands():
    assert role_allows(HoldRole.FOOT, Limb.LF)
    assert role_allows(HoldRole.FOOT, Limb.RF)
    assert not role_allows(HoldRole.FOOT, Limb.LH)
    assert not role_allows(HoldRole.FOOT, Limb.RH)


def test_start_hand_finish_allow_both_hands_and_feet():
    for role in (HoldRole.START, HoldRole.HAND, HoldRole.FINISH, HoldRole.ANY):
        for limb in Limb:
            assert role_allows(role, limb)


def test_hold_allows_delegates_to_role_allows():
    h = Hold(hold_id=1, x=0.0, y=0.0, hold_type=HoldType.JUG, role=HoldRole.FOOT)
    assert h.allows(Limb.LF)
    assert not h.allows(Limb.LH)


def test_matchable_by_size_or_type():
    small_crimp = Hold(hold_id=1, x=0, y=0, hold_type=HoldType.CRIMP, size=1.0)
    assert not small_crimp.is_matchable
    big_sloper = Hold(hold_id=2, x=0, y=0, hold_type=HoldType.SLOPER, size=1.0)
    assert big_sloper.is_matchable  # slopers are matchable by type regardless of size
    wide_crimp = Hold(hold_id=3, x=0, y=0, hold_type=HoldType.CRIMP, size=4.0)
    assert wide_crimp.is_matchable
