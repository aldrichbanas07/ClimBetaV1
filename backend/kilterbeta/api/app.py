"""FastAPI application.

Endpoints
---------
``GET  /health``                 liveness + what data is loaded
``GET  /layouts``                board layouts available
``GET  /layouts/{id}/holds``     every hold on a board (for the faint underlay)
``GET  /climbs``                 browse/search climbs
``GET  /climbs/{id}``            one climb: holds, roles, per-angle grades
``POST /generate-beta``          THE endpoint: holds + angle -> move list
``GET  /schema/move``            the move-list JSON Schema and its version
``GET  /difficulty-model``       weights currently in use, for the UI to explain

The response shape of ``/generate-beta`` is ``domain.moves.BetaResponse`` and
is versioned; see that module's stability rules before changing it.
"""

from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from ..beta.body import BodyModel
from ..beta.calibration import GradeCalibration
from ..beta.difficulty import DifficultyModel, foot_load_share, hand_load_share, steepness
from ..beta.generator import generate_beta
from ..beta.search import SearchConfig
from ..config import settings
from ..db.connection import connect
from ..db.repository import Repository
from ..domain.holds import Hold, HoldRole, HoldType, Limb
from ..domain.moves import SCHEMA_VERSION, BetaResponse, HoldOut

app = FastAPI(
    title="Kilter Beta AI",
    version="0.1.0",
    description=(
        "Heuristic beta generation for Kilter Board climbs. "
        "Phase 1: hold geometry + hold type + wall angle -> limb sequence."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------- dependencies


def get_conn():
    """Per-request read-only connection.

    SQLite connections are not safe to share across threads, and FastAPI runs
    sync endpoints in a thread pool, so a connection per request it is. This is
    cheap for SQLite and keeps the code obvious.
    """
    if not settings.db_path.exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"database {settings.db_path} not found. "
                "Run: python -m kilterbeta.etl.cli init-sample"
            ),
        )
    conn = connect(settings.db_path, read_only=True)
    try:
        yield conn
    finally:
        conn.close()


def get_repo(conn: sqlite3.Connection = Depends(get_conn)) -> Repository:
    return Repository(conn)


_calibration: Optional[GradeCalibration] = None


def get_calibration() -> GradeCalibration:
    """Load the fitted score->grade map once, falling back to hand-set values."""
    global _calibration
    if _calibration is None:
        _calibration = GradeCalibration.load(settings.calibration_path)
    return _calibration


# ------------------------------------------------------------- request models


class HoldInput(BaseModel):
    """An ad-hoc hold, for callers who are not using a stored climb."""

    model_config = ConfigDict(extra="forbid")

    hold_id: Optional[int] = Field(None, description="Optional; auto-assigned if omitted")
    x: float
    y: float
    hold_type: HoldType = HoldType.UNKNOWN
    role: HoldRole = HoldRole.HAND
    size: float = 3.0


class BodyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    height: float = Field(69.0, gt=36.0, lt=96.0, description="Climber height in inches")
    arm_length: Optional[float] = Field(None, gt=0)
    leg_length: Optional[float] = Field(None, gt=0)
    torso_length: Optional[float] = Field(None, gt=0)
    shoulder_width: Optional[float] = Field(None, gt=0)
    hip_width: Optional[float] = Field(None, gt=0)

    def to_model(self) -> BodyModel:
        return BodyModel(
            height=self.height,
            arm_length_in=self.arm_length,
            leg_length_in=self.leg_length,
            torso_length_in=self.torso_length,
            shoulder_width_in=self.shoulder_width,
            hip_width_in=self.hip_width,
        )


class GenerateBetaRequest(BaseModel):
    """Supply exactly one of ``climb_id`` or ``hold_list``."""

    model_config = ConfigDict(extra="forbid")

    climb_id: Optional[str] = None
    hold_list: Optional[List[HoldInput]] = None
    angle: int = Field(settings.default_angle, ge=0, le=70)

    body: Optional[BodyInput] = None
    strategy: str = Field("astar", pattern="^(astar|greedy)$")
    beam_per_limb: int = Field(6, ge=2, le=20)
    max_expansions: int = Field(20000, ge=100, le=400000)
    require_finish_match: bool = True
    allow_dynamic: bool = True


class ClimbSummaryOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    climb_id: str
    layout_id: int
    name: str
    setter: Optional[str] = None
    description: Optional[str] = None
    setter_angle: Optional[int] = None
    hold_count: int
    source: str
    graded_angles: List[int] = Field(default_factory=list)


class LayoutOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    name: str
    product_name: Optional[str] = None
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    source: str


class ClimbDetailOut(BaseModel):
    model_config = ConfigDict(extra="forbid")

    climb: ClimbSummaryOut
    layout: Optional[LayoutOut]
    holds: List[HoldOut]
    stats: Dict[int, Dict[str, Optional[float]]] = Field(
        default_factory=dict, description="angle -> community difficulty, if known"
    )


def _hold_out(h: Hold) -> HoldOut:
    return HoldOut(
        hold_id=h.hold_id,
        x=h.x,
        y=h.y,
        hold_type=h.hold_type,
        role=h.role,
        size=h.size,
        placement_id=h.placement_id,
        name=h.name,
    )


# -------------------------------------------------------------------- routes


@app.get("/health")
def health() -> Dict[str, object]:
    out: Dict[str, object] = {
        "status": "ok",
        "schema_version": SCHEMA_VERSION,
        "database": str(settings.db_path),
        "database_present": settings.db_path.exists(),
        "calibrated": get_calibration().calibrated,
    }
    if settings.db_path.exists():
        conn = connect(settings.db_path, read_only=True)
        try:
            out["counts"] = Repository(conn).counts()
        finally:
            conn.close()
    return out


@app.get("/layouts", response_model=List[LayoutOut])
def list_layouts(repo: Repository = Depends(get_repo)) -> List[LayoutOut]:
    return [LayoutOut(**l.__dict__) for l in repo.list_layouts()]


@app.get("/layouts/{layout_id}/holds", response_model=List[HoldOut])
def layout_holds(layout_id: int, repo: Repository = Depends(get_repo)) -> List[HoldOut]:
    if repo.get_layout(layout_id) is None:
        raise HTTPException(status_code=404, detail=f"layout {layout_id} not found")
    return [_hold_out(h) for h in repo.layout_holds(layout_id)]


@app.get("/climbs", response_model=List[ClimbSummaryOut])
def list_climbs(
    layout_id: Optional[int] = None,
    search: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    repo: Repository = Depends(get_repo),
) -> List[ClimbSummaryOut]:
    return [
        ClimbSummaryOut(**c.__dict__)
        for c in repo.list_climbs(
            layout_id=layout_id, search=search, limit=limit, offset=offset
        )
    ]


@app.get("/climbs/{climb_id}", response_model=ClimbDetailOut)
def get_climb(climb_id: str, repo: Repository = Depends(get_repo)) -> ClimbDetailOut:
    climb = repo.get_climb(climb_id)
    if climb is None:
        raise HTTPException(status_code=404, detail=f"climb {climb_id} not found")
    layout = repo.get_layout(climb.layout_id)
    return ClimbDetailOut(
        climb=ClimbSummaryOut(**climb.__dict__),
        layout=LayoutOut(**layout.__dict__) if layout else None,
        holds=[_hold_out(h) for h in repo.climb_holds(climb_id)],
        stats=repo.climb_stats(climb_id),
    )


@app.post("/generate-beta", response_model=BetaResponse)
def post_generate_beta(
    req: GenerateBetaRequest,
    repo: Repository = Depends(get_repo),
) -> BetaResponse:
    """Generate an optimal-under-our-cost-model limb sequence.

    The response is ``BetaResponse`` at ``schema_version`` -- see
    ``/schema/move``. Phase 2 will populate each move's ``pose`` field; the
    rest of the document will not change shape.
    """
    if bool(req.climb_id) == bool(req.hold_list is not None):
        raise HTTPException(
            status_code=422, detail="supply exactly one of 'climb_id' or 'hold_list'"
        )

    climb_name: Optional[str] = None
    if req.climb_id:
        climb = repo.get_climb(req.climb_id)
        if climb is None:
            raise HTTPException(status_code=404, detail=f"climb {req.climb_id} not found")
        holds = repo.climb_holds(req.climb_id)
        climb_name = climb.name
    else:
        holds = [
            Hold(
                hold_id=h.hold_id if h.hold_id is not None else i + 1,
                x=h.x,
                y=h.y,
                hold_type=h.hold_type,
                role=h.role,
                size=h.size,
            )
            for i, h in enumerate(req.hold_list or [])
        ]
        seen = {h.hold_id for h in holds}
        if len(seen) != len(holds):
            raise HTTPException(status_code=422, detail="duplicate hold_id in hold_list")

    if len(holds) < 2:
        raise HTTPException(status_code=422, detail="a climb needs at least 2 holds")

    calibration = get_calibration()
    grades = repo.difficulty_grades()
    if grades:
        calibration.grades = grades

    config = SearchConfig(
        angle=req.angle,
        strategy=req.strategy,
        beam_per_limb=req.beam_per_limb,
        max_expansions=req.max_expansions,
        require_finish_match=req.require_finish_match,
        allow_dynamic=req.allow_dynamic,
    )

    try:
        return generate_beta(
            holds,
            angle=req.angle,
            body=(req.body.to_model() if req.body else BodyModel()),
            model=DifficultyModel(),
            calibration=calibration,
            config=config,
            climb_id=req.climb_id,
            climb_name=climb_name,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/schema/move")
def move_schema() -> Dict[str, object]:
    """The move-list schema, so clients can validate and detect version drift."""
    return {
        "schema_version": SCHEMA_VERSION,
        "reserved_for_phase_2": ["moves[].pose", "moves[].body.source == 'ik'"],
        "reserved_for_phase_3": ["moves[].extensions"],
        "json_schema": BetaResponse.model_json_schema(),
    }


@app.get("/difficulty-model")
def difficulty_model(angle: int = Query(40, ge=0, le=70)) -> Dict[str, object]:
    """Weights in use, plus each hold type's cost at the requested angle.

    Backs the UI panel that explains *why* a move scored the way it did.
    """
    model = DifficultyModel()
    return {
        "angle": angle,
        "steepness": round(steepness(angle), 4),
        "hand_load_share": round(hand_load_share(angle), 4),
        "foot_load_share": round(foot_load_share(angle), 4),
        "hands": {
            t.value: model.explain_hold(t, Limb.RH, angle) for t in HoldType
        },
        "feet": {
            t.value: model.explain_hold(t, Limb.RF, angle) for t in HoldType
        },
        "cost_weights": model.as_dict()["weights"],
    }
