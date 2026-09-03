import pytest

from kilterbeta.beta.body import BodyModel
from kilterbeta.beta.generator import generate_beta
from kilterbeta.domain.holds import ALL_LIMBS, HoldRole, HoldType, Limb
from kilterbeta.domain.moves import SCHEMA_VERSION, MoveKind

from .conftest import make_hold


def test_generate_beta_echoes_schema_version(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    assert result.schema_version == SCHEMA_VERSION


def test_generate_beta_moves_are_serialisable_pydantic_models(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    dumped = result.model_dump()
    assert dumped["angle"] == 40
    assert len(dumped["moves"]) == len(result.moves)


def test_moves_indexed_sequentially_from_zero(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    assert [m.index for m in result.moves] == list(range(len(result.moves)))


def test_pose_and_extensions_reserved_for_later_phases(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    for move in result.moves:
        assert move.pose is None
        assert move.extensions == {}
        assert move.body.source == "heuristic-centroid"


def test_body_model_echoed_matches_input(jug_ladder):
    body = BodyModel(height=64.0)
    result = generate_beta(jug_ladder, angle=40, body=body)
    assert result.body_model["height"] == pytest.approx(64.0)
    assert result.body_model["arm_length"] == pytest.approx(body.arm_length, abs=0.01)


def test_contacts_reflect_full_stance_not_just_moved_limb(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    # By the last move both hands should be represented in a contacts list.
    late_move = result.moves[-1]
    contact_limbs = {c.limb for c in late_move.contacts}
    assert Limb.LH in contact_limbs or Limb.RH in contact_limbs


def test_difficulty_matches_breakdown_total(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    for move in result.moves:
        assert move.difficulty == pytest.approx(move.difficulty_breakdown.total)


def test_grade_estimate_uncalibrated_by_default(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    assert result.grade.calibrated is False
    assert result.grade.kilter_difficulty is not None


def test_crux_move_index_points_at_highest_difficulty_non_start_move(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    travelling = [m for m in result.moves if m.kind is not MoveKind.START]
    hardest = max(travelling, key=lambda m: m.difficulty)
    assert result.grade.crux_move_index == hardest.index


def test_holds_out_mirrors_input_climb(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    assert {h.hold_id for h in result.holds} == {h.hold_id for h in jug_ladder}


def test_generator_metadata_reports_strategy_and_truncation(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    assert result.generator["strategy"] == "astar"
    assert "nodes_expanded" in result.generator
    assert "difficulty_model" in result.generator


def test_finish_move_kind_marks_the_last_hand_move(jug_ladder):
    result = generate_beta(jug_ladder, angle=40)
    finishes = [m for m in result.moves if m.kind is MoveKind.FINISH]
    assert len(finishes) == 1
    assert finishes[0].limb.is_hand


def test_climb_id_and_name_passthrough(jug_ladder):
    result = generate_beta(jug_ladder, angle=40, climb_id="abc", climb_name="Test Climb")
    assert result.climb_id == "abc"
    assert result.climb_name == "Test Climb"
