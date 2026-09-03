"""Hold-type difficulty weights and the wall-angle model.

Two physically distinct channels are modelled, and kept separate on purpose so
that calibration can move one without disturbing the other:

**Channel A - hold quality under load** (``hold_cost``)
    How hard a specific hold type is for a specific limb, scaled by how much
    the wall angle punishes that type. Slopers and open-hand shapes degrade
    fast as the wall steepens because the resultant force rotates out of the
    hold's usable surface; jugs barely care. This is the hold-type x angle
    interaction that drives the whole tool.

**Channel B - global overhang tax** (``body_tension_cost``)
    Core/tension cost that has nothing to do with hold type: on a 50 degree
    wall, keeping your hips in and your feet on costs energy regardless of
    what you are holding.

All weights are in the same arbitrary "cost units" as
``DifficultyBreakdown``. They are dimensionless and only meaningful relative
to each other; ``calibration.py`` maps their sum onto a real grade scale.

Sign convention: bigger = harder. 0.0 would be a free hold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional

from ..domain.holds import HoldType, Limb

#: Wall angle treated as maximally steep. Kilter's app tops out at 70 degrees.
MAX_ANGLE_DEG = 70.0


@dataclass(frozen=True)
class HoldWeight:
    """Difficulty profile of one hold type.

    ``base`` is the cost at 0 degrees (vertical). ``angle_sensitivity`` is the
    extra multiple of ``base`` added at maximum steepness, applied through a
    sine ramp so it tracks the actual out-of-wall force component rather than
    growing linearly in degrees.
    """

    base: float
    angle_sensitivity: float


# --- Channel A: hands ------------------------------------------------------
# Rationale for the ordering the brief asked for (crimp/sloper/pinch harder
# than jug), and for the sensitivities:
#   jug     - positive and deep; steepness costs you endurance, not purchase
#   edge    - a decent incut rail; middling
#   pocket  - secure but restricts wrist/finger orientation
#   pinch   - depends on thumb strength, which fades as the load rotates out
#   crimp   - small; load rises steeply with overhang
#   sloper  - friction-only; the single most angle-sensitive shape
HAND_WEIGHTS: Dict[HoldType, HoldWeight] = {
    HoldType.JUG:       HoldWeight(base=0.10, angle_sensitivity=0.35),
    HoldType.EDGE:      HoldWeight(base=0.34, angle_sensitivity=0.85),
    HoldType.POCKET:    HoldWeight(base=0.46, angle_sensitivity=0.80),
    HoldType.PINCH:     HoldWeight(base=0.52, angle_sensitivity=0.95),
    HoldType.CRIMP:     HoldWeight(base=0.74, angle_sensitivity=1.30),
    HoldType.SLOPER:    HoldWeight(base=0.80, angle_sensitivity=1.70),
    HoldType.FOOT_CHIP: HoldWeight(base=1.15, angle_sensitivity=1.45),
    HoldType.UNKNOWN:   HoldWeight(base=0.48, angle_sensitivity=0.90),
}

# --- Channel A: feet -------------------------------------------------------
# Feet care about a crisp edge to stand on, not about being incut, so the
# ranking differs from hands: a small sharp chip is a *fine* foothold, while a
# rounded sloper is a smear that fails as soon as the wall steepens.
FOOT_WEIGHTS: Dict[HoldType, HoldWeight] = {
    HoldType.JUG:       HoldWeight(base=0.06, angle_sensitivity=0.45),
    HoldType.EDGE:      HoldWeight(base=0.10, angle_sensitivity=0.55),
    HoldType.FOOT_CHIP: HoldWeight(base=0.16, angle_sensitivity=0.60),
    HoldType.PINCH:     HoldWeight(base=0.18, angle_sensitivity=0.65),
    HoldType.CRIMP:     HoldWeight(base=0.22, angle_sensitivity=0.70),
    HoldType.POCKET:    HoldWeight(base=0.26, angle_sensitivity=0.75),
    HoldType.SLOPER:    HoldWeight(base=0.44, angle_sensitivity=1.55),
    HoldType.UNKNOWN:   HoldWeight(base=0.20, angle_sensitivity=0.70),
}


def steepness(angle_deg: float) -> float:
    """Normalised out-of-wall force component, 0.0 at vertical -> 1.0 at 70 deg.

    Uses sin(angle) normalised by sin(70) rather than angle/70 because the
    force pulling the climber off the wall goes as sin of the overhang, which
    is why the 40->55 degree range feels like a much bigger jump than 0->15.
    """
    a = max(0.0, min(float(angle_deg), MAX_ANGLE_DEG))
    return math.sin(math.radians(a)) / math.sin(math.radians(MAX_ANGLE_DEG))


def hand_load_share(angle_deg: float) -> float:
    """Fraction of body weight carried by the hands, 0.30 (vertical) -> ~0.96."""
    return 0.30 + 0.66 * steepness(angle_deg)


def foot_load_share(angle_deg: float) -> float:
    """How much useful downward purchase the feet still have."""
    return max(0.05, math.cos(math.radians(max(0.0, min(float(angle_deg), MAX_ANGLE_DEG)))))


@dataclass
class DifficultyModel:
    """Bundles the weight tables with the cost weights used to combine them.

    Everything tunable lives here so that ``calibration.py`` can fit a variant
    and the API can report exactly which numbers produced a beta.
    """

    hand_weights: Mapping[HoldType, HoldWeight] = field(default_factory=lambda: dict(HAND_WEIGHTS))
    foot_weights: Mapping[HoldType, HoldWeight] = field(default_factory=lambda: dict(FOOT_WEIGHTS))

    # Relative importance of each breakdown component in the final move cost.
    w_reach: float = 1.00
    w_target_hold: float = 1.15
    w_support_holds: float = 0.85
    w_body_tension: float = 0.90
    w_balance: float = 0.55
    w_penalties: float = 1.00

    # Reach strain shape: cost = (utilisation ** reach_exponent). Cubic keeps
    # comfortable moves nearly free while punishing moves near the limit.
    reach_exponent: float = 3.0
    reach_scale: float = 1.60

    # Fixed penalties.
    penalty_cross_through: float = 0.45
    penalty_match: float = 0.35
    #: Sharing a hold with your own hand is legal but awkward, and costs a lot
    #: more than matching hand-to-hand or foot-to-foot.
    penalty_hand_foot_match: float = 1.20
    penalty_bump: float = 0.15
    penalty_cut_feet: float = 0.75
    penalty_dynamic: float = 1.10
    penalty_high_step: float = 0.45
    penalty_downward_hand: float = 0.35

    def hold_cost(self, hold_type: HoldType, limb: Limb, angle_deg: float) -> float:
        """Channel A: cost of a limb using a hold of this type at this angle."""
        table = self.hand_weights if limb.is_hand else self.foot_weights
        w = table.get(hold_type) or table[HoldType.UNKNOWN]
        return w.base * (1.0 + w.angle_sensitivity * steepness(angle_deg))

    #: Hand-to-foot separation, as a fraction of torso+leg length, at which the
    #: body is in its most economical position.
    ideal_extension: float = 0.55

    def body_tension_cost(
        self,
        angle_deg: float,
        hand_foot_separation: float,
        body_span: float,
        feet_on: int,
    ) -> float:
        """Channel B: core cost of holding the position, hold types aside.

        Grows with steepness, sharply when feet come off, and with how far the
        body is from its economical extension.

        Note that the cost is two-sided. Being over-extended (hands far above
        feet, a long lever for the core) is expensive, but so is being
        scrunched -- and a stance with the feet *above* the hands is the most
        scrunched of all. Treating only over-extension as costly makes a
        one-sided model prefer putting a foot at chest height, because that
        drives the separation to zero and looks maximally relaxed.
        """
        s = steepness(angle_deg)

        ratio = 0.0
        if body_span > 1e-6:
            ratio = hand_foot_separation / body_span
        ideal = self.ideal_extension
        if ratio >= ideal:
            # Over-extended: strain rises toward full stretch.
            strain = (ratio - ideal) / max(1.4 - ideal, 1e-6)
        else:
            # Scrunched, or feet above hands (negative ratio).
            strain = (ideal - ratio) / (ideal + 0.4) * 0.8
        strain = max(0.0, min(strain, 1.4))

        if feet_on == 0:
            stance_factor = 2.2  # fully cut; everything goes through the core
        elif feet_on == 1:
            stance_factor = 1.35
        else:
            stance_factor = 1.0

        return s * stance_factor * (0.35 + 0.75 * strain)

    def reach_cost(self, utilisation: float) -> float:
        """Strain from the span of a move, as a fraction of available reach."""
        u = max(0.0, utilisation)
        return self.reach_scale * (u ** self.reach_exponent)

    def as_dict(self) -> Dict[str, object]:
        """Provenance blob for the API response."""
        return {
            "weights": {
                "reach": self.w_reach,
                "target_hold": self.w_target_hold,
                "support_holds": self.w_support_holds,
                "body_tension": self.w_body_tension,
                "balance": self.w_balance,
                "penalties": self.w_penalties,
            },
            "reach_exponent": self.reach_exponent,
            "reach_scale": self.reach_scale,
            "hand_weights": {
                k.value: {"base": v.base, "angle_sensitivity": v.angle_sensitivity}
                for k, v in self.hand_weights.items()
            },
            "foot_weights": {
                k.value: {"base": v.base, "angle_sensitivity": v.angle_sensitivity}
                for k, v in self.foot_weights.items()
            },
        }

    # --- Introspection helper used by the UI's "why is this hard" panel ----
    def explain_hold(self, hold_type: HoldType, limb: Limb, angle_deg: float) -> Dict[str, float]:
        table = self.hand_weights if limb.is_hand else self.foot_weights
        w = table.get(hold_type) or table[HoldType.UNKNOWN]
        s = steepness(angle_deg)
        return {
            "base": w.base,
            "angle_sensitivity": w.angle_sensitivity,
            "steepness": round(s, 4),
            "multiplier": round(1.0 + w.angle_sensitivity * s, 4),
            "cost": round(w.base * (1.0 + w.angle_sensitivity * s), 4),
        }


#: Minimum conceivable cost of any single move, used as the A* heuristic's
#: per-move lower bound. Must stay <= the true minimum or A* loses admissibility.
def min_move_cost(model: DifficultyModel, angle_deg: float) -> float:
    cheapest_hand = min(
        model.hold_cost(t, Limb.RH, angle_deg) for t in model.hand_weights
    )
    return model.w_target_hold * cheapest_hand


DEFAULT_MODEL = DifficultyModel()


def default_model() -> DifficultyModel:
    """A fresh, mutable copy of the default weights."""
    return DifficultyModel()
