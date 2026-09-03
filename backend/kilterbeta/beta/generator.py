"""Orchestration: holds + angle -> versioned ``BetaResponse``.

This is the single place where the internal search representation is
translated into the public wire schema. Keeping the translation in one
function means phase 2 can wrap it (``generate_beta`` -> add poses -> return)
without touching the search or the API.
"""

from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Dict, List, Optional, Sequence

from ..domain.holds import ALL_LIMBS, Hold
from ..domain.moves import (
    SCHEMA_VERSION,
    BetaMove,
    BetaResponse,
    BodyEstimate,
    Contact,
    DifficultyBreakdown,
    GradeEstimate,
    HoldOut,
    MoveKind,
    Point2D,
)
from .body import BodyModel, Stance
from .calibration import GradeCalibration
from .difficulty import DifficultyModel
from .search import BetaSearch, MoveEval, SearchConfig


def _contacts_out(contacts, by_id: Dict[int, Hold]) -> List[Contact]:
    """Wire-format four-limb stance, straight from the search's limb assignment."""
    if not contacts:
        return []
    out: List[Contact] = []
    for limb in ALL_LIMBS:
        hold_id = contacts.get(limb)
        if hold_id is None:
            continue
        hold = by_id.get(hold_id)
        if hold is None:
            continue
        out.append(
            Contact(
                limb=limb,
                hold_id=hold.hold_id,
                x=hold.x,
                y=hold.y,
                hold_type=hold.hold_type,
            )
        )
    return out


def _body_estimate(stance: Optional[Stance]) -> BodyEstimate:
    if stance is None:
        zero = Point2D(x=0.0, y=0.0)
        return BodyEstimate(hip=zero, shoulder_left=zero, shoulder_right=zero)
    return BodyEstimate(
        hip=Point2D(x=round(stance.hip[0], 2), y=round(stance.hip[1], 2)),
        shoulder_left=Point2D(x=round(stance.shoulder_left[0], 2), y=round(stance.shoulder_left[1], 2)),
        shoulder_right=Point2D(x=round(stance.shoulder_right[0], 2), y=round(stance.shoulder_right[1], 2)),
        source="heuristic-centroid",
    )


def _to_move(index: int, ev: MoveEval, by_id: Dict[int, Hold], model: DifficultyModel) -> BetaMove:
    hold = by_id[ev.hold_id]
    breakdown = DifficultyBreakdown(
        reach=round(model.w_reach * ev.reach, 4),
        target_hold=round(model.w_target_hold * ev.target_hold, 4),
        support_holds=round(model.w_support_holds * ev.support_holds, 4),
        body_tension=round(model.w_body_tension * ev.body_tension, 4),
        balance=round(model.w_balance * ev.balance, 4),
        penalties=round(model.w_penalties * ev.penalties, 4),
        total=round(ev.cost, 4),
        notes=list(ev.notes),
    )
    return BetaMove(
        index=index,
        limb=ev.limb,
        hold_id=ev.hold_id,
        x=hold.x,
        y=hold.y,
        hold_type=hold.hold_type,
        kind=ev.kind,
        from_hold_id=ev.from_hold_id,
        reach_distance=round(ev.reach_distance, 2),
        reach_utilisation=round(ev.reach_utilisation, 4),
        difficulty=round(ev.cost, 4),
        difficulty_breakdown=breakdown,
        contacts=_contacts_out(ev.contacts_after, by_id),
        body=_body_estimate(ev.stance_after),
        pose=None,          # PHASE 2 fills this in.
        extensions={},      # PHASE 3 namespaces its data here.
    )


def generate_beta(
    holds: Sequence[Hold],
    angle: int,
    body: Optional[BodyModel] = None,
    model: Optional[DifficultyModel] = None,
    calibration: Optional[GradeCalibration] = None,
    config: Optional[SearchConfig] = None,
    climb_id: Optional[str] = None,
    climb_name: Optional[str] = None,
) -> BetaResponse:
    """Plan a beta and package it in the versioned response schema."""
    body = body or BodyModel()
    model = model or DifficultyModel()
    calibration = calibration or GradeCalibration()
    config = config or SearchConfig(angle=angle)
    if config.angle != angle:
        config = dc_replace(config, angle=angle)

    search = BetaSearch(holds, config, body=body, model=model)
    result = search.run()

    by_id = {h.hold_id: h for h in holds}
    moves = [_to_move(i, ev, by_id, model) for i, ev in enumerate(result.moves)]

    # Grade off the *travelling* moves only: the opening hand placements are
    # not moves a climber has to execute, and including them dilutes the crux.
    scored = [m.difficulty for m in moves if m.kind is not MoveKind.START]
    if not scored:
        scored = [m.difficulty for m in moves]

    difficulty_value = calibration.difficulty(scored)
    crux_index: Optional[int] = None
    if moves:
        crux = max(
            (m for m in moves if m.kind is not MoveKind.START),
            key=lambda m: m.difficulty,
            default=None,
        )
        crux_index = crux.index if crux is not None else None

    grade = GradeEstimate(
        difficulty_score=round(sum(scored), 4),
        kilter_difficulty=round(difficulty_value, 2),
        boulder_grade=calibration.grade_name(difficulty_value),
        crux_move_index=crux_index,
        calibrated=calibration.calibrated,
    )

    return BetaResponse(
        schema_version=SCHEMA_VERSION,
        climb_id=climb_id,
        climb_name=climb_name,
        angle=angle,
        moves=moves,
        grade=grade,
        holds=[
            HoldOut(
                hold_id=h.hold_id,
                x=h.x,
                y=h.y,
                hold_type=h.hold_type,
                role=h.role,
                size=h.size,
                placement_id=h.placement_id,
                name=h.name,
            )
            for h in holds
        ],
        body_model=body.as_dict(),
        generator={
            "strategy": result.strategy,
            "nodes_expanded": result.expansions,
            "elapsed_ms": round(result.elapsed_ms, 2),
            "truncated": result.truncated,
            "reached_finish": not result.truncated,
            "config": {
                "beam_per_limb": config.beam_per_limb,
                "max_expansions": config.max_expansions,
                "max_moves": config.max_moves,
                "heuristic_weight": config.heuristic_weight,
                "require_finish_match": config.require_finish_match,
                "allow_dynamic": config.allow_dynamic,
            },
            "difficulty_model": model.as_dict(),
            "calibration": {
                "calibrated": calibration.calibrated,
                "n_samples": calibration.n_samples,
                "rmse": calibration.rmse,
            },
        },
        warnings=list(result.warnings),
    )
