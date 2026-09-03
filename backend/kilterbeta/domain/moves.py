"""Versioned beta move-list schema.

This module is the **public contract** between the beta generator, the HTTP
API, and the frontend. Phase 2 (inverse kinematics) and phase 3 (video pose
comparison) are expected to *attach data to* these structures, not reshape
them.

Stability rules
---------------
1. ``SCHEMA_VERSION`` is semver. Additive, optional fields bump the MINOR
   version. Renaming/removing/retyping a field bumps MAJOR.
2. Every response echoes the version in ``BetaResponse.schema_version`` so a
   client can branch on it.
3. Reserved-for-later fields are declared *now* and serialised as ``null``, so
   phase 2 turns them on without changing the shape of the document:
       - ``BetaMove.pose``       -> full skeletal pose at this move (phase 2)
       - ``BetaMove.extensions`` -> free-form namespaced add-ons
       - ``BodyEstimate.source`` -> flips from "heuristic-centroid" to "ik"
4. ``BetaMove.contacts`` deliberately carries the *complete four-limb stance*
   after the move, not just the limb that moved. Inverse kinematics needs the
   whole stance to solve a pose, and video alignment needs it to score one.
   Recomputing it client-side would be lossy, so it is transmitted.
5. ``BetaResponse.body_model`` carries the anthropometry the search actually
   used. Phase 2 IK *must* solve against these same numbers or the poses will
   not match the sequence that was planned.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .holds import HoldRole, HoldType, Limb

# --- Bump deliberately. See stability rules above. -------------------------
SCHEMA_VERSION = "1.0.0"


class MoveKind(str, Enum):
    """Coarse classification of a single limb movement."""

    START = "start"          # initial placement, no travel cost
    STATIC = "static"        # controlled reach within comfortable range
    LONG = "long"            # near the limit of static reach
    DYNAMIC = "dynamic"      # beyond static reach; requires momentum
    MATCH = "match"          # limb joins another limb on the same hold
    BUMP = "bump"            # same limb moves again immediately
    FOOT_SWAP = "foot_swap"  # feet exchange holds
    FINISH = "finish"        # move that completes the climb


class DifficultyBreakdown(BaseModel):
    """Per-move cost decomposition, in arbitrary but internally consistent units.

    Exposed in full because it is what makes the output explainable in the UI
    ("this move is hard because of the *hold*, not the *reach*"), and because
    calibration fits against these components individually.
    """

    model_config = ConfigDict(extra="forbid")

    reach: float = Field(..., description="Strain from the span of the move")
    target_hold: float = Field(..., description="Difficulty of grabbing the destination hold")
    support_holds: float = Field(..., description="Difficulty of holding the stance being moved from")
    body_tension: float = Field(..., description="Core/overhang tax, independent of hold type")
    balance: float = Field(..., description="Barn-door / rotational instability proxy")
    penalties: float = Field(0.0, description="Cross-through, match, bump, cut-foot, etc.")
    total: float = Field(..., description="Weighted sum of the above")

    notes: List[str] = Field(
        default_factory=list, description="Human-readable reasons contributing to the cost"
    )


class Contact(BaseModel):
    """One limb resting on one hold."""

    model_config = ConfigDict(extra="forbid")

    limb: Limb
    hold_id: int
    x: float
    y: float
    hold_type: HoldType


class HoldOut(BaseModel):
    """A hold belonging to a climb, as sent to clients for rendering."""

    model_config = ConfigDict(extra="forbid")

    hold_id: int
    x: float
    y: float
    hold_type: HoldType
    role: HoldRole
    size: float = 3.0
    placement_id: Optional[int] = None
    name: Optional[str] = None


class Point2D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float
    y: float


class BodyEstimate(BaseModel):
    """Coarse body reference points at a move.

    Phase 1 fills these from a weighted-centroid heuristic. Phase 2 replaces
    the values with IK output and sets ``source="ik"``, additionally populating
    ``BetaMove.pose``. Consumers should read ``source`` before trusting these
    to any precision.
    """

    model_config = ConfigDict(extra="forbid")

    hip: Point2D
    shoulder_left: Point2D
    shoulder_right: Point2D
    source: str = Field(
        "heuristic-centroid",
        description='Provenance: "heuristic-centroid" (phase 1) or "ik" (phase 2)',
    )


class BetaMove(BaseModel):
    """A single limb movement in a generated beta."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., description="0-based position in the sequence")
    limb: Limb
    hold_id: int
    x: float
    y: float
    hold_type: HoldType
    kind: MoveKind

    from_hold_id: Optional[int] = Field(
        None, description="Hold this limb left, or null for its initial placement"
    )
    reach_distance: float = Field(
        ..., description="Inches from the limb's reach origin (shoulder/hip) to the target hold"
    )
    reach_utilisation: float = Field(
        ..., description="reach_distance / max reach for this limb; 1.0 == at the static limit"
    )

    difficulty: float = Field(..., description="Total cost of this move; equals difficulty_breakdown.total")
    difficulty_breakdown: DifficultyBreakdown

    contacts: List[Contact] = Field(
        ..., description="Complete four-limb stance immediately AFTER this move (feet may be absent)"
    )
    body: BodyEstimate

    # --- Reserved for later phases; always present, null/empty in phase 1. --
    pose: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "PHASE 2: skeletal joint angles and segment endpoints at this move "
            "(hip, shoulders, elbows, knees, flexion angles, drop-knee flag). Null in phase 1."
        ),
    )
    extensions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Namespaced add-on data, e.g. {'video': {...}} in phase 3.",
    )


class GradeEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty_score: float = Field(..., description="Aggregate beta cost (crux-weighted)")
    kilter_difficulty: Optional[float] = Field(
        None, description="Score mapped onto Kilter's numeric difficulty scale, if calibrated"
    )
    boulder_grade: Optional[str] = Field(None, description='e.g. "V5/6C+"')
    crux_move_index: Optional[int] = Field(None, description="Index of the single hardest move")
    calibrated: bool = Field(
        False,
        description="True when the score->grade map was fitted against real per-angle Kilter data",
    )


class BetaResponse(BaseModel):
    """Top-level payload of ``POST /generate-beta``."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    climb_id: Optional[str] = None
    climb_name: Optional[str] = None
    angle: int = Field(..., ge=0, le=70, description="Wall overhang in degrees from vertical")

    moves: List[BetaMove]
    grade: GradeEstimate
    holds: List[HoldOut] = Field(
        default_factory=list, description="The climb's holds, so a client can render without a second call"
    )

    body_model: Dict[str, float] = Field(
        ...,
        description="Anthropometry used by the search, in inches. Phase 2 IK must reuse these values.",
    )
    generator: Dict[str, Any] = Field(
        default_factory=dict,
        description="Algorithm provenance: strategy, cost weights, nodes expanded, timing.",
    )
    warnings: List[str] = Field(default_factory=list)
