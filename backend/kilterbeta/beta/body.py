"""Anthropometry and the coarse body-position estimate.

Phase 1 needs just enough of a body to answer two questions:

    1. Where is the climber's hip/shoulder right now?  (reach origins)
    2. Can a given limb reach a given hold from there?  (feasibility + strain)

The hip solve here is deliberately a *positional constraint relaxation* rather
than a centroid one-liner: it already speaks the language phase 2's inverse
kinematics will use (segment lengths, reach origins, constraint violations), so
phase 2 can extend ``solve_stance`` to full joint angles instead of replacing
it. Everything is planar (the wall plane); out-of-wall depth is phase 2's
problem.

Segment ratios follow the standard Drillis & Contini proportions, tuned so
that fingertip-to-fingertip span comes out at roughly the climber's height
(the usual "ape index 0" assumption).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from ..domain.holds import FEET, HANDS, Limb

Vec = Tuple[float, float]


@dataclass(frozen=True)
class BodyModel:
    """Climber anthropometry, in inches.

    All lengths are derived from ``height`` unless explicitly overridden, so a
    caller can say ``BodyModel(height=63)`` and get a coherent small climber.
    Phase 2 IK must be handed the *same instance* (it is echoed in
    ``BetaResponse.body_model``) or its poses will not match the planned beta.
    """

    height: float = 69.0

    # Overrides; resolved in __post_init__ style properties below.
    arm_length_in: Optional[float] = None      # shoulder joint -> fingertip
    leg_length_in: Optional[float] = None      # hip joint -> toe
    torso_length_in: Optional[float] = None    # hip joint -> shoulder joint
    shoulder_width_in: Optional[float] = None  # between shoulder joints
    hip_width_in: Optional[float] = None       # between hip joints

    #: Fraction of ``arm_length`` a hand may exceed before the move stops being
    #: static and becomes a dynamic/jump move. 1.0 disables dynamic moves.
    dynamic_reach_factor: float = 1.12

    @property
    def arm_length(self) -> float:
        return self.arm_length_in if self.arm_length_in is not None else 0.390 * self.height

    @property
    def leg_length(self) -> float:
        return self.leg_length_in if self.leg_length_in is not None else 0.540 * self.height

    @property
    def torso_length(self) -> float:
        return self.torso_length_in if self.torso_length_in is not None else 0.288 * self.height

    @property
    def shoulder_width(self) -> float:
        return self.shoulder_width_in if self.shoulder_width_in is not None else 0.210 * self.height

    @property
    def hip_width(self) -> float:
        return self.hip_width_in if self.hip_width_in is not None else 0.191 * self.height

    @property
    def ape_span(self) -> float:
        """Fingertip-to-fingertip span."""
        return 2 * self.arm_length + self.shoulder_width

    def max_reach(self, limb: Limb) -> float:
        """Maximum static distance from the limb's reach origin to a hold."""
        return self.arm_length if limb.is_hand else self.leg_length

    def as_dict(self) -> Dict[str, float]:
        """Serialised form echoed to clients and consumed by phase 2."""
        return {
            "height": round(self.height, 2),
            "arm_length": round(self.arm_length, 2),
            "leg_length": round(self.leg_length, 2),
            "torso_length": round(self.torso_length, 2),
            "shoulder_width": round(self.shoulder_width, 2),
            "hip_width": round(self.hip_width, 2),
            "ape_span": round(self.ape_span, 2),
            "dynamic_reach_factor": self.dynamic_reach_factor,
        }


@dataclass
class Stance:
    """A resolved four-limb stance: contact points plus body reference points.

    ``contacts`` maps limb -> (x, y). A foot may be absent (cut feet / start
    off the ground); hands are always expected to be present after the start.
    """

    contacts: Dict[Limb, Vec]
    hip: Vec
    shoulder_left: Vec
    shoulder_right: Vec
    #: Sum of constraint violation in inches after relaxation. A large value
    #: means the stance is geometrically strained (or impossible).
    residual: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)

    def origin_for(self, limb: Limb) -> Vec:
        """The point a limb reaches *from*: a shoulder for hands, a hip for feet."""
        if limb is Limb.LH:
            return self.shoulder_left
        if limb is Limb.RH:
            return self.shoulder_right
        # Both feet swing from the hip centre. Hip width is small next to leg
        # length, and ``solve_stance`` uses the same origin, so keeping them
        # identical avoids a reach/feasibility mismatch.
        return self.hip

    @property
    def hand_points(self):
        return [self.contacts[l] for l in HANDS if l in self.contacts]

    @property
    def foot_points(self):
        return [self.contacts[l] for l in FEET if l in self.contacts]


def _mean(points) -> Optional[Vec]:
    pts = list(points)
    if not pts:
        return None
    n = float(len(pts))
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def hip_hand_weight(angle_deg: float) -> float:
    """How strongly the hips are drawn toward the hands rather than the feet.

    On a vertical wall the hips sit over the feet (weight ~0.30). As the wall
    steepens the climber's mass hangs under the hands and the hips ride up and
    in (weight -> ~0.72 at 70 degrees). This single scalar is what makes the
    whole stance geometry angle-dependent.
    """
    s = math.sin(math.radians(angle_deg))
    return 0.30 + 0.48 * s


def solve_stance(
    contacts: Mapping[Limb, Vec],
    body: BodyModel,
    angle_deg: float,
    iterations: int = 12,
) -> Stance:
    """Estimate hip and shoulder positions for a set of limb contacts.

    Starts from an angle-weighted centroid of the contacts, then runs a few
    Jacobi relaxation sweeps that pull the hip toward any contact whose segment
    is over-extended and push it away from any that is impossibly compressed.
    Deliberately shaped like a positional IK solver so phase 2 can grow joint
    angles out of it.
    """
    contacts = {l: (float(p[0]), float(p[1])) for l, p in contacts.items() if p is not None}
    hands = _mean(contacts[l] for l in HANDS if l in contacts)
    feet = _mean(contacts[l] for l in FEET if l in contacts)

    w_hand = hip_hand_weight(angle_deg)
    if hands is None and feet is None:
        raise ValueError("solve_stance requires at least one contact")
    if hands is None:
        hip = (feet[0], feet[1] + body.leg_length * 0.8)
    elif feet is None:
        # Feet cut: hips hang below the hands, closer in on steep ground.
        hip = (hands[0], hands[1] - body.torso_length * (1.0 - 0.25 * w_hand))
    else:
        hip = (
            w_hand * hands[0] + (1.0 - w_hand) * feet[0],
            w_hand * (hands[1] - body.torso_length) + (1.0 - w_hand) * (feet[1] + body.leg_length * 0.72),
        )

    half_shoulder = body.shoulder_width / 2.0
    torso = body.torso_length

    residual = 0.0
    for _ in range(max(1, iterations)):
        dx = dy = 0.0
        residual = 0.0
        n = 0
        for limb, point in contacts.items():
            if limb.is_hand:
                sx = hip[0] + (-half_shoulder if limb.is_left else half_shoulder)
                origin = (sx, hip[1] + torso)
                limit = body.arm_length
            else:
                origin = (hip[0], hip[1])
                limit = body.leg_length

            vx, vy = point[0] - origin[0], point[1] - origin[1]
            d = math.hypot(vx, vy)
            if d < 1e-9:
                continue
            # Over-extended: drag the hip toward the hold. Excessively folded
            # (hold much closer than half a segment): nudge the hip away.
            if d > limit:
                excess = d - limit
                residual += excess
                dx += vx / d * excess
                dy += vy / d * excess
                n += 1
            elif d < 0.35 * limit:
                deficit = 0.35 * limit - d
                dx -= vx / d * deficit * 0.5
                dy -= vy / d * deficit * 0.5
                n += 1

        if n == 0:
            break
        # Under-relax for stability; contacts pull in competing directions.
        hip = (hip[0] + 0.6 * dx / n, hip[1] + 0.6 * dy / n)

    return Stance(
        contacts=dict(contacts),
        hip=hip,
        shoulder_left=(hip[0] - half_shoulder, hip[1] + torso),
        shoulder_right=(hip[0] + half_shoulder, hip[1] + torso),
        residual=residual,
    )


def distance(a: Vec, b: Vec) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def support_triangle_area(points) -> float:
    """Area of the polygon formed by the supporting contacts.

    Used as a barn-door proxy: three nearly-collinear contacts give an area
    near zero and cannot resist rotation about that line. For 2 or fewer
    contacts the area is 0 by definition.
    """
    pts = list(points)
    if len(pts) < 3:
        return 0.0
    # Shoelace over however many contacts we have (3 or 4).
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0
