"""Beta search: A* (and a greedy fallback) over feasible hand assignments.

Why hands drive the search
--------------------------
The obvious formulation -- search over all four limbs at once -- has a state
space of ``(hand holds)^2 x (foot holds + 1)^2``, and because foot shuffles are
individually cheap, a shortest-path search burns essentially all its budget
permuting feet before it ever advances a hand. Measured on a 16-hold problem
that was ~7000 expansions and six seconds.

So the search is structured the way a climber thinks about it:

* **A\\* over hand states** ``(left hand hold, right hand hold, last hand moved)``.
  This is the "feasible hand assignments" space, and it is small: a few
  hundred states even for a busy climb.
* **Feet are resolved per hand state** by a bounded sub-search
  (``_resolve_feet``) that picks the stance minimising foot-hold cost, body
  tension and balance, subject to leg reach. Its result is cached per hand
  pair.
* **Foot repositioning is still costed and still emitted**: changing hand state
  usually changes the best stance, and those foot moves appear in the output
  with their own difficulty scores, priced into the transition.

That makes the cost of a hand move depend on the stance you are actually
standing in, which is the whole point -- moving off a crimp with your feet cut
on a 50 degree wall should score very differently from the same span on a slab.

State
-----
``HandState(lh, rh, last)``. ``last`` is carried because it changes successor
costs (bumping the same hand twice in a row is penalised), so it belongs in
the state rather than in the path.

Search
------
A* with an admissible-by-construction heuristic: a lower bound on the number
of remaining hand moves multiplied by the cheapest conceivable move cost. Two
safety valves bound the work: a per-limb candidate beam and a hard expansion
cap. Both are reported in the response so a caller can tell whether the answer
was truncated.

No pydantic here; the inner loop deals in floats and dataclasses.
``generator.py`` translates the result into the versioned wire schema.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..domain.holds import ALL_LIMBS, FEET, HANDS, Hold, HoldRole, HoldType, Limb
from ..domain.moves import MoveKind
from .body import BodyModel, Stance, distance, solve_stance, support_triangle_area
from .difficulty import (
    DifficultyModel,
    foot_load_share,
    hand_load_share,
    min_move_cost,
    steepness,
)

Vec = Tuple[float, float]
#: limb -> hold_id. The search's contact representation.
Contacts = Dict[Limb, int]

#: Reach utilisation below which a move counts as comfortably static.
STATIC_THRESHOLD = 0.78

#: Minimum hold width (inches) for a hand and a foot to share one hold. Board
#: holds smaller than this cannot take a shoe and fingers at the same time.
HAND_FOOT_SHARE_MIN_SIZE = 4.5


@dataclass(frozen=True)
class SearchConfig:
    angle: int = 40
    strategy: str = "astar"            # "astar" | "greedy"
    beam_per_limb: int = 6             # hand candidates considered per state
    foot_candidates: int = 3           # footholds shortlisted per side
    max_expansions: int = 20000        # hard cap on A* node expansions
    max_moves: int = 60                # abandon paths longer than this
    heuristic_weight: float = 1.0      # >1 = weighted A*: faster, possibly suboptimal
    require_finish_match: bool = True  # Kilter convention: match the finish jug
    allow_dynamic: bool = True
    allow_hand_foot_share: bool = False  # a foot on the hold a hand is already on
    max_start_states: int = 4


@dataclass
class MoveEval:
    """Result of costing one limb movement."""

    limb: Limb
    hold_id: int
    from_hold_id: Optional[int]
    kind: MoveKind
    reach_distance: float
    reach_utilisation: float
    cost: float
    reach: float
    target_hold: float
    support_holds: float
    body_tension: float
    balance: float
    penalties: float
    notes: List[str] = field(default_factory=list)
    #: Filled in by ``_finalise`` for the chosen path only.
    stance_after: Optional[Stance] = None
    contacts_after: Optional[Contacts] = None


@dataclass(frozen=True)
class HandState:
    lh: int
    rh: int
    last: Optional[Limb] = None

    def hold_of(self, limb: Limb) -> int:
        return self.lh if limb is Limb.LH else self.rh

    def moved(self, limb: Limb, hold_id: int) -> "HandState":
        if limb is Limb.LH:
            return HandState(lh=hold_id, rh=self.rh, last=Limb.LH)
        return HandState(lh=self.lh, rh=hold_id, last=Limb.RH)


@dataclass
class Transition:
    """One hand move, plus the foot repositioning that follows it.

    Order matters and matches how the costs were computed: the hand move is
    measured from the stance belonging to the state being left, then the feet
    come up into the stance the new position wants.
    """

    hand_move: MoveEval
    foot_moves: List[MoveEval] = field(default_factory=list)

    @property
    def cost(self) -> float:
        return self.hand_move.cost + sum(f.cost for f in self.foot_moves)

    def as_list(self) -> List[MoveEval]:
        return [self.hand_move] + list(self.foot_moves)


@dataclass
class SearchResult:
    moves: List[MoveEval]
    total_cost: float
    expansions: int
    elapsed_ms: float
    truncated: bool
    warnings: List[str] = field(default_factory=list)
    strategy: str = "astar"


class BetaSearch:
    """Plans a limb sequence for one climb at one angle."""

    def __init__(
        self,
        holds: Sequence[Hold],
        config: SearchConfig,
        body: Optional[BodyModel] = None,
        model: Optional[DifficultyModel] = None,
    ) -> None:
        if not holds:
            raise ValueError("cannot plan a beta for a climb with no holds")

        self.holds: List[Hold] = list(holds)
        self.by_id: Dict[int, Hold] = {h.hold_id: h for h in self.holds}
        if len(self.by_id) != len(self.holds):
            raise ValueError("duplicate hold_id in climb")

        self.config = config
        self.body = body or BodyModel()
        self.model = model or DifficultyModel()
        self.angle = float(config.angle)

        # --- Vectorised geometry, built once and reused every expansion. ---
        self.xs = np.array([h.x for h in self.holds], dtype=np.float64)
        self.ys = np.array([h.y for h in self.holds], dtype=np.float64)
        self._hand_idx = np.array(
            [i for i, h in enumerate(self.holds) if h.allows(Limb.RH)], dtype=np.int64
        )
        self._foot_idx = np.array(
            [i for i, h in enumerate(self.holds) if h.allows(Limb.RF)], dtype=np.int64
        )

        self.start_holds = [h for h in self.holds if h.role is HoldRole.START]
        self.finish_holds = [h for h in self.holds if h.role is HoldRole.FINISH]
        self._finish_ids = frozenset(
            h.hold_id for h in (self.finish_holds or self._fallback_finish_holds())
        )

        # Distance from every hold to the nearest finish hold. This is the
        # progress metric: it works on traverses, where height does not.
        if self._finish_ids:
            fx = np.array([self.by_id[f].x for f in self._finish_ids], dtype=np.float64)
            fy = np.array([self.by_id[f].y for f in self._finish_ids], dtype=np.float64)
            self._finish_dist = np.hypot(
                self.xs[:, None] - fx[None, :], self.ys[:, None] - fy[None, :]
            ).min(axis=1)
        else:
            self._finish_dist = np.zeros(len(self.holds), dtype=np.float64)
        self._finish_dist_by_id = {
            h.hold_id: float(self._finish_dist[i]) for i, h in enumerate(self.holds)
        }

        self._min_move = min_move_cost(self.model, self.angle)
        self._steep = steepness(self.angle)
        self._hand_load = hand_load_share(self.angle)
        self._foot_load = foot_load_share(self.angle)
        self._body_span = self.body.torso_length + self.body.leg_length
        # Generous upper bound on vertical progress per hand move; over-
        # estimating it keeps the A* heuristic admissible.
        self._max_gain = self.body.arm_length + self.body.torso_length + self.body.leg_length

        self._stance_cache: Dict[Tuple, Stance] = {}
        self._feet_cache: Dict[Tuple[int, int], Contacts] = {}
        #: Count of foot placements the plan wanted but could not legally
        #: reach; surfaced as a response warning rather than swallowed.
        self.unreachable_feet = 0

    # --------------------------------------------------------- fallbacks

    def _fallback_hand_holds(self) -> List[Hold]:
        """Lowest hand-usable holds, for climbs with no tagged start holds."""
        usable = [h for h in self.holds if h.allows(Limb.LH)]
        usable.sort(key=lambda h: h.y)
        return usable[:3]

    def _fallback_finish_holds(self) -> List[Hold]:
        usable = [h for h in self.holds if h.allows(Limb.LH)]
        if not usable:
            return []
        top = max(h.y for h in usable)
        return [h for h in usable if h.y >= top - 6.0]

    # ---------------------------------------------------------- geometry

    def _positions(self, contacts: Contacts) -> Dict[Limb, Vec]:
        out: Dict[Limb, Vec] = {}
        for limb, hold_id in contacts.items():
            if hold_id is None:
                continue
            h = self.by_id[hold_id]
            out[limb] = (h.x, h.y)
        return out

    def _stance(self, contacts: Contacts, iterations: int = 12) -> Stance:
        key = (
            contacts.get(Limb.LH),
            contacts.get(Limb.RH),
            contacts.get(Limb.LF),
            contacts.get(Limb.RF),
            iterations,
        )
        cached = self._stance_cache.get(key)
        if cached is not None:
            return cached
        stance = solve_stance(
            self._positions(contacts), self.body, self.angle, iterations=iterations
        )
        if len(self._stance_cache) < 100_000:
            self._stance_cache[key] = stance
        return stance

    def _reachable(self, limb: Limb, origin: Vec, reach: float) -> np.ndarray:
        """Indices of holds this limb could legally reach from ``origin``."""
        pool = self._hand_idx if limb.is_hand else self._foot_idx
        if pool.size == 0:
            return pool
        d = np.hypot(self.xs[pool] - origin[0], self.ys[pool] - origin[1])
        return pool[d <= reach]

    # ------------------------------------------------------ foot planning

    #: Bonus applied to a stance from which some onward hand move exists. Large
    #: enough that any stance which unlocks progress beats any that does not,
    #: so that among *usable* stances the most comfortable still wins. See
    #: ``_resolve_feet`` and ``_unlocks_progress``.
    PROGRESS_WEIGHT = 2.0
    #: Extra, deliberately small, preference for stances with reach to spare.
    PROGRESS_MARGIN_WEIGHT = 0.3
    #: Inches of progress toward the finish that counts as "an onward move".
    PROGRESS_EPSILON = 2.0

    def _resolve_feet(self, hand_state: HandState) -> Contacts:
        """Choose the feet for a hand state. Deterministic and cached.

        Feet cannot be chosen by comfort alone. The cheapest stance to *hold*
        is a low, relaxed one -- but the whole reason a climber moves their
        feet up is to reach the next hold. Scoring stances purely on comfort
        makes the planner stand around pleasantly and then find every onward
        move out of reach.

        So the score is comfort *minus* how much upward reach the stance
        unlocks (``PROGRESS_WEIGHT``). That single term is what turns
        "stand somewhere nice" into "get your foot up so you can reach", and
        it is why foot choice can stay outside the A* state: the choice is a
        deterministic function of the hand pair, so no alternative is ever
        lost to a visited-state check.
        """
        key = (hand_state.lh, hand_state.rh)
        cached = self._feet_cache.get(key)
        if cached is not None:
            return cached

        hands: Contacts = {Limb.LH: hand_state.lh, Limb.RH: hand_state.rh}
        hip = self._stance(hands, iterations=8).hip
        leg = self.body.leg_length

        # Look slightly beyond leg reach of the hands-only hip: placing a foot
        # moves the hip, which can bring an out-of-range hold into range.
        idx = self._reachable(Limb.LF, hip, leg * 1.15)

        by_side: Dict[bool, List[Tuple[float, float, int]]] = {True: [], False: []}
        for i in idx:
            hold = self.holds[int(i)]
            if hold.hold_id in self._finish_ids:
                # A foot resolver has no notion of "the climb ends here" -- it
                # just sees a comfortable, reachable jug. But the finish
                # normally has to take both hands, so parking a foot there
                # pre-emptively blocks the climb from ever finishing. Simplest
                # fix: feet never treat the finish as a foothold candidate.
                continue
            d = distance(hip, hold.position)
            comfort = self.model.hold_cost(
                hold.hold_type, Limb.RF, self.angle
            ) + self.model.reach_cost(d / leg) * 0.5
            if hold.y > hip[1]:
                comfort += self.model.penalty_high_step * ((hold.y - hip[1]) / leg)
            # Keep height as a second key so tall stances survive the cut.
            if hold.x <= hip[0] + self.body.hip_width:
                by_side[True].append((comfort, hold.y, hold.hold_id))
            if hold.x >= hip[0] - self.body.hip_width:
                by_side[False].append((comfort, hold.y, hold.hold_id))

        k = max(1, self.config.foot_candidates)
        per_side: Dict[bool, List[Optional[int]]] = {}
        for is_left, entries in by_side.items():
            chosen: List[Optional[int]] = []
            for hid in [e[2] for e in sorted(entries, key=lambda e: e[0])[:k]]:
                if hid not in chosen:
                    chosen.append(hid)
            for hid in [e[2] for e in sorted(entries, key=lambda e: -e[1])[:k]]:
                if hid not in chosen:
                    chosen.append(hid)
            chosen.append(None)  # this foot off the wall
            per_side[is_left] = chosen

        best: Optional[Contacts] = None
        best_score = float("inf")
        for lf in per_side[True]:
            for rf in per_side[False]:
                if lf is not None and lf == rf and not self.by_id[lf].is_matchable:
                    continue
                contacts: Contacts = dict(hands)
                if lf is not None:
                    contacts[Limb.LF] = lf
                if rf is not None:
                    contacts[Limb.RF] = rf

                unlocks, margin = self._unlocks_progress(contacts, hand_state)
                score = (
                    self._stance_cost(contacts)
                    - self.PROGRESS_WEIGHT * unlocks
                    - self.PROGRESS_MARGIN_WEIGHT * margin
                )
                if score < best_score:
                    best_score = score
                    best = contacts

        result = best if best is not None else dict(hands)
        if len(self._feet_cache) < 100_000:
            self._feet_cache[key] = result
        return result

    def _unlocks_progress(
        self, contacts: Contacts, hand_state: HandState
    ) -> Tuple[float, float]:
        """Does this stance leave an onward move available?

        Returns ``(unlocks, margin)``: ``unlocks`` is 1.0 if either hand can
        reach a hold measurably closer to the finish than where it already is,
        and ``margin`` is how much closer, normalised.

        Progress is measured as *distance to the finish*, not height, so this
        works on a traverse as well as on a ladder. It is deliberately
        saturating: rewarding raw reachable height instead makes the planner
        high-step onto hand holds to maximise a number, which is how you get a
        beta that starts by putting a foot at chest level.
        """
        best_unlock = 0.0
        best_margin = 0.0
        for limb in HANDS:
            support = {l: hid for l, hid in contacts.items() if l is not limb}
            if not any(l.is_hand for l in support):
                continue
            origin = self._stance(support).origin_for(limb)
            reach = self.body.max_reach(limb) * (
                self.body.dynamic_reach_factor if self.config.allow_dynamic else 1.0
            )
            idx = self._reachable(limb, origin, reach)
            if idx.size == 0:
                continue
            here = self._finish_dist_by_id.get(hand_state.hold_of(limb))
            if here is None:
                continue
            closest = float(self._finish_dist[idx].min())
            gain = here - closest
            if gain > self.PROGRESS_EPSILON:
                best_unlock = 1.0
                best_margin = max(best_margin, min(gain / self._max_gain, 1.0))
        return best_unlock, best_margin

    def _stance_cost(self, contacts: Contacts) -> float:
        """Static cost of *holding* a stance. Used to choose between stances."""
        stance = self._stance(contacts, iterations=8)
        feet = [l for l in FEET if l in contacts]
        hands = [l for l in HANDS if l in contacts]

        support = 0.0
        if hands:
            per = self._hand_load / len(hands)
            for l in hands:
                support += self.model.hold_cost(
                    self.by_id[contacts[l]].hold_type, l, self.angle
                ) * per
        if feet:
            per = self._foot_load / len(feet) * 0.6
            for l in feet:
                support += self.model.hold_cost(
                    self.by_id[contacts[l]].hold_type, l, self.angle
                ) * per

        positions = self._positions(contacts)
        hand_y = [positions[l][1] for l in hands] or [stance.hip[1]]
        if feet:
            foot_y = [positions[l][1] for l in feet]
            separation = sum(hand_y) / len(hand_y) - sum(foot_y) / len(foot_y)
        else:
            separation = self._body_span
        tension = self.model.body_tension_cost(
            self.angle, separation, self._body_span, len(feet)
        )

        area = support_triangle_area(list(positions.values()))
        area_norm = area / max(self.body.shoulder_width * self.body.torso_length, 1.0)
        balance = 0.35 / (1.0 + area_norm) * (0.5 + 0.5 * self._steep)

        pen = 0.0
        if not feet:
            pen += self.model.penalty_cut_feet * (0.5 + 0.5 * self._steep)
        elif len(feet) == 1:
            pen += self.model.penalty_cut_feet * 0.35 * (0.5 + 0.5 * self._steep)
        # Crossed feet are almost never what you want.
        if Limb.LF in contacts and Limb.RF in contacts:
            if positions[Limb.LF][0] > positions[Limb.RF][0] + 1.0:
                pen += self.model.penalty_cross_through
        # Geometrically strained stances (over-extended segments).
        pen += stance.residual / max(self.body.leg_length, 1.0)

        return (
            self.model.w_support_holds * support
            + self.model.w_body_tension * tension
            + self.model.w_balance * balance
            + self.model.w_penalties * pen
        )

    # ---------------------------------------------------------- cost model

    def _cost_move(
        self,
        support: Contacts,
        limb: Limb,
        target: Hold,
        from_hold: Optional[int],
        last_limb: Optional[Limb],
        explain: bool = False,
    ) -> Optional[MoveEval]:
        """Cost one limb movement out of a support stance, or None if illegal.

        ``support`` must already exclude the moving limb: reach and load are
        measured from the position you are actually in *while* moving.
        """
        if from_hold == target.hold_id:
            return None
        if not target.allows(limb):
            return None
        if limb.is_hand and not any(l.is_hand for l in support):
            return None  # cannot release both hands at once
        if not support:
            return None

        occupants = [l for l, hid in support.items() if hid == target.hold_id]
        if occupants:
            mixed_share = any(o.is_hand != limb.is_hand for o in occupants)
            if mixed_share:
                # Standing on the hold your own hand is on. Board holds are
                # small and this is rare enough in practice that the default
                # ruleset forbids it outright; allowing it let the search park
                # both feet on the finish jug.
                if not self.config.allow_hand_foot_share:
                    return None
                if target.size < HAND_FOOT_SHARE_MIN_SIZE:
                    return None
            elif not target.is_matchable:
                # Too small to share, unless it is the finish and the ruleset
                # says both hands must end there.
                if not (target.role is HoldRole.FINISH and limb.is_hand):
                    return None

        stance = self._stance(support)
        origin = stance.origin_for(limb)
        max_reach = self.body.max_reach(limb)
        dist = distance(origin, target.position)
        util = dist / max_reach if max_reach > 1e-9 else float("inf")

        # Only hands can move dynamically. A foot either reaches the hold or it
        # does not -- there is no such thing as jumping a foot onto a chip.
        limit = 1.0
        if limb.is_hand and self.config.allow_dynamic:
            limit = self.body.dynamic_reach_factor
        if util > limit:
            return None

        notes: List[str] = []

        # Dynamic is checked before match: a match that is beyond static reach
        # is still a dynamic move and must be priced as one. The match penalty
        # is applied separately below, from ``occupants``.
        if util > 1.0:
            kind = MoveKind.DYNAMIC
        elif occupants:
            kind = MoveKind.MATCH
        elif util > STATIC_THRESHOLD:
            kind = MoveKind.LONG
        elif last_limb is limb:
            kind = MoveKind.BUMP
        elif limb.is_foot and from_hold is not None:
            kind = MoveKind.FOOT_SWAP
        else:
            kind = MoveKind.STATIC

        # --- reach --------------------------------------------------------
        reach_c = self.model.reach_cost(util)
        if explain and util > STATIC_THRESHOLD:
            notes.append(
                f"Long reach: {dist:.0f} in, {util * 100:.0f}% of this limb's span"
            )

        # --- target hold (channel A) --------------------------------------
        target_c = self.model.hold_cost(target.hold_type, limb, self.angle)
        if explain:
            notes.append(
                f"{target.hold_type.value} for {limb.value} at {self.angle:.0f} deg "
                f"costs {target_c:.2f}"
            )

        # --- support holds (channel A, load-weighted) ---------------------
        support_hands = [l for l in support if l.is_hand]
        support_feet = [l for l in support if l.is_foot]
        support_c = 0.0
        if support_hands:
            per = self._hand_load / len(support_hands)
            for l in support_hands:
                support_c += self.model.hold_cost(
                    self.by_id[support[l]].hold_type, l, self.angle
                ) * per
        if support_feet:
            per = self._foot_load / len(support_feet) * 0.6
            for l in support_feet:
                support_c += self.model.hold_cost(
                    self.by_id[support[l]].hold_type, l, self.angle
                ) * per
        if explain and limb.is_hand and len(support_hands) == 1:
            held = self.by_id[support[support_hands[0]]]
            notes.append(
                f"Pulling off a single {held.hold_type.value} with "
                f"{self._hand_load * 100:.0f}% of body weight on it"
            )

        # --- body tension (channel B) -------------------------------------
        positions = self._positions(support)
        hand_pts = [positions[l] for l in support_hands] or [origin]
        mean_hand_y = sum(p[1] for p in hand_pts) / len(hand_pts)
        if support_feet:
            foot_pts = [positions[l] for l in support_feet]
            separation = mean_hand_y - sum(p[1] for p in foot_pts) / len(foot_pts)
        else:
            separation = self._body_span
        tension_c = self.model.body_tension_cost(
            self.angle, separation, self._body_span, len(support_feet)
        )
        if explain and not support_feet:
            notes.append("Feet are off: the whole move goes through the core")

        # --- balance / barn-door ------------------------------------------
        support_pts = list(positions.values())
        area = support_triangle_area(support_pts)
        area_norm = area / max(self.body.shoulder_width * self.body.torso_length, 1.0)
        sxs = [p[0] for p in support_pts]
        lateral = max(0.0, max(min(sxs) - target.x, target.x - max(sxs))) / max(
            self.body.shoulder_width, 1.0
        )
        balance_c = (0.35 + lateral) / (1.0 + area_norm) * (0.5 + 0.5 * self._steep)
        if explain and lateral > 0.75:
            notes.append(
                f"Reaching {lateral:.1f} shoulder-widths outside your support: barn-door risk"
            )

        # --- penalties ----------------------------------------------------
        pen = 0.0
        if kind is MoveKind.DYNAMIC:
            # Scaled two ways: by how far past static reach it is, and by how
            # bad the hold you are catching is. Deadpointing to a jug is a
            # normal move; deadpointing to a sloper is a different sport.
            overshoot = (util - 1.0) / max(self.body.dynamic_reach_factor - 1.0, 1e-6)
            pen += self.model.penalty_dynamic * (0.5 + overshoot) * (1.0 + target_c)
            if explain:
                notes.append(
                    f"Beyond static reach: dynamic move to a {target.hold_type.value}"
                )
        if occupants:
            mixed = any(o.is_hand != limb.is_hand for o in occupants)
            pen += self.model.penalty_hand_foot_match if mixed else self.model.penalty_match
            if explain:
                notes.append(
                    f"Sharing this hold with your {occupants[0].value}"
                    if mixed
                    else f"Matching alongside {occupants[0].value}"
                )
        if last_limb is limb and not occupants:
            pen += self.model.penalty_bump
            if explain:
                notes.append("Bumping the same limb twice in a row")
        if not support_feet:
            pen += self.model.penalty_cut_feet * (0.5 + 0.5 * self._steep)

        # Cross-through: the moving limb ends up past its opposite number.
        other = limb.opposite
        other_hold = support.get(other)
        if other_hold is not None and other_hold != target.hold_id:
            other_x = self.by_id[other_hold].x
            overlap = (other_x - target.x) if limb.is_left else (target.x - other_x)
            if overlap < -1.0:
                depth = min(-overlap / max(self.body.shoulder_width, 1.0), 2.0)
                pen += self.model.penalty_cross_through * depth
                if explain:
                    notes.append(f"Cross-through: {limb.value} reaching past {other.value}")

        if limb.is_foot and target.y > stance.hip[1]:
            rise = (target.y - stance.hip[1]) / max(self.body.leg_length, 1.0)
            pen += self.model.penalty_high_step * min(rise * 2.0, 2.0)
            if explain:
                notes.append("High step above the hip")

        if limb.is_hand and from_hold is not None:
            drop = self.by_id[from_hold].y - target.y
            if drop > 2.0:
                pen += self.model.penalty_downward_hand * min(drop / 12.0, 2.0)
                if explain:
                    notes.append("Reaching back down: usually wasted motion")

        total = (
            self.model.w_reach * reach_c
            + self.model.w_target_hold * target_c
            + self.model.w_support_holds * support_c
            + self.model.w_body_tension * tension_c
            + self.model.w_balance * balance_c
            + self.model.w_penalties * pen
        )

        return MoveEval(
            limb=limb,
            hold_id=target.hold_id,
            from_hold_id=from_hold,
            kind=kind,
            reach_distance=dist,
            reach_utilisation=util,
            cost=total,
            reach=reach_c,
            target_hold=target_c,
            support_holds=support_c,
            body_tension=tension_c,
            balance=balance_c,
            penalties=pen,
            notes=notes,
        )

    # ------------------------------------------------------- transitions

    def _foot_transition(
        self,
        from_feet: Contacts,
        to_feet: Contacts,
        hand_state: HandState,
        explain: bool = False,
    ) -> List[MoveEval]:
        """Cost and order the foot moves between two stances.

        A foot that cannot legally reach its intended hold from here simply
        stays put and is counted in ``self.unreachable_feet``, which surfaces
        as a response warning. That is a deliberate approximation: the target
        stance is a *preference*, and refusing the whole transition over one
        awkward foot would strand the search. Exact stance feasibility is
        phase 2's job, once there is a real skeleton to solve against.
        """
        moves: List[MoveEval] = []
        current: Contacts = {l: hid for l, hid in from_feet.items() if l.is_foot}
        current[Limb.LH] = hand_state.lh
        current[Limb.RH] = hand_state.rh

        pending = [
            l for l in FEET if to_feet.get(l) is not None and current.get(l) != to_feet.get(l)
        ]

        # Move the foot that is furthest out of place first; an unplaced foot
        # goes first of all.
        def displacement(limb: Limb) -> float:
            src = current.get(limb)
            if src is None:
                return float("inf")
            return distance(self.by_id[src].position, self.by_id[to_feet[limb]].position)

        pending.sort(key=displacement, reverse=True)

        for limb in pending:
            target = self.by_id[to_feet[limb]]
            support = {l: hid for l, hid in current.items() if l is not limb}
            ev = self._cost_move(
                support,
                limb,
                target,
                from_hold=current.get(limb),
                last_limb=None,
                explain=explain,
            )
            if ev is None:
                self.unreachable_feet += 1
                continue
            moves.append(ev)
            current[limb] = target.hold_id
        return moves

    def _transition(
        self,
        state: HandState,
        limb: Limb,
        target: Hold,
        explain: bool = False,
    ) -> Optional[Tuple[HandState, Transition]]:
        """Move one hand, then bring the feet up to suit the new position.

        The reach is measured from the stance belonging to the state we are
        leaving -- ``_resolve_feet(state)`` -- because that is where the
        climber is standing when they make the move. The feet then follow, at
        their own cost, into the stance the destination wants.
        """
        nxt = state.moved(limb, target.hold_id)
        other = limb.opposite
        feet = self._resolve_feet(state)

        support: Contacts = {other: state.hold_of(other)}
        for l in FEET:
            if feet.get(l) is not None:
                support[l] = feet[l]

        hand_move = self._cost_move(
            support,
            limb,
            target,
            from_hold=state.hold_of(limb),
            last_limb=state.last,
            explain=explain,
        )
        if hand_move is None:
            return None

        foot_moves = self._foot_transition(feet, self._resolve_feet(nxt), nxt, explain=explain)
        return nxt, Transition(hand_move=hand_move, foot_moves=foot_moves)

    def _successors(self, state: HandState) -> List[Tuple[HandState, Transition]]:
        out: List[Tuple[HandState, Transition]] = []
        feet = self._resolve_feet(state)
        for limb in HANDS:
            other = limb.opposite
            support: Contacts = {other: state.hold_of(other)}
            for l in FEET:
                if feet.get(l) is not None:
                    support[l] = feet[l]
            stance = self._stance(support)
            origin = stance.origin_for(limb)
            reach = self.body.max_reach(limb) * (
                self.body.dynamic_reach_factor if self.config.allow_dynamic else 1.0
            )

            # Shortlist by the cheap part of the cost (hold type + reach) before
            # paying for foot resolution on each one.
            shortlist: List[Tuple[float, Hold]] = []
            for i in self._reachable(limb, origin, reach):
                hold = self.holds[int(i)]
                if hold.hold_id == state.hold_of(limb):
                    continue
                d = distance(origin, hold.position)
                pre = self.model.w_target_hold * self.model.hold_cost(
                    hold.hold_type, limb, self.angle
                ) + self.model.w_reach * self.model.reach_cost(d / self.body.max_reach(limb))
                shortlist.append((pre, hold))
            shortlist.sort(key=lambda t: t[0])

            for _, hold in shortlist[: self.config.beam_per_limb]:
                result = self._transition(state, limb, hold)
                if result is not None:
                    out.append(result)
        return out

    # --------------------------------------------------------------- start

    def start_states(self) -> List[Tuple[HandState, List[MoveEval]]]:
        """Enumerate plausible opening hand configurations."""
        starts = self.start_holds or self._fallback_hand_holds()
        if not starts:
            raise ValueError("climb has no hand-usable holds")

        pairs: List[Tuple[Hold, Hold]] = []
        if len(starts) == 1:
            pairs.append((starts[0], starts[0]))  # matched start
        else:
            # Only uncrossed assignments: left hand on the more leftward hold.
            # Crossed starts are vanishingly rare in practice, and allowing
            # them lets the search open crossed purely so it can "un-cross" by
            # matching on the next move -- a cheaper path through the cost
            # model that no climber would ever do.
            ordered = sorted(starts, key=lambda h: h.x)
            for i, left in enumerate(ordered):
                for right in ordered[i + 1:]:
                    if distance(left.position, right.position) <= self.body.ape_span:
                        pairs.append((left, right))
            if not pairs:
                lowest = sorted(starts, key=lambda h: h.y)[0]
                pairs.append((lowest, lowest))

        def rank(p: Tuple[Hold, Hold]) -> float:
            lh, rh = p
            spread = distance(lh.position, rh.position) / max(self.body.ape_span, 1.0)
            cost = self.model.hold_cost(lh.hold_type, Limb.LH, self.angle) + self.model.hold_cost(
                rh.hold_type, Limb.RH, self.angle
            )
            return spread * 0.5 + cost

        pairs.sort(key=rank)
        out: List[Tuple[HandState, List[MoveEval]]] = []
        seen = set()
        for lh, rh in pairs:
            key = (lh.hold_id, rh.hold_id)
            if key in seen:
                continue
            seen.add(key)
            state = HandState(lh=lh.hold_id, rh=rh.hold_id, last=None)
            # Starting with the hands crossed has to be priced into the path
            # cost, not just into the ordering here, or A* will happily pick a
            # crossed start because ranking never reaches the frontier. Scaled
            # by how deeply crossed it is, as mid-climb crossings are.
            out.append(
                (state, [self._placement(Limb.LH, lh), self._placement(Limb.RH, rh)])
            )
            if len(out) >= self.config.max_start_states:
                break
        return out

    def _placement(self, limb: Limb, hold: Hold) -> MoveEval:
        """Zero-travel record for a hand starting on a start hold."""
        hold_c = self.model.hold_cost(hold.hold_type, limb, self.angle)
        pen = 0.0
        notes = [f"Start: {limb.value} on the {hold.hold_type.value} start hold"]
        return MoveEval(
            limb=limb,
            hold_id=hold.hold_id,
            from_hold_id=None,
            kind=MoveKind.START,
            reach_distance=0.0,
            reach_utilisation=0.0,
            cost=self.model.w_target_hold * hold_c + self.model.w_penalties * pen,
            reach=0.0,
            target_hold=hold_c,
            support_holds=0.0,
            body_tension=0.0,
            balance=0.0,
            penalties=pen,
            notes=notes,
        )

    # ------------------------------------------------------------ goal + h

    def is_goal(self, state: HandState) -> bool:
        if not self._finish_ids:
            return False
        hands = {state.lh, state.rh}
        if len(self._finish_ids) == 1:
            fid = next(iter(self._finish_ids))
            return hands == {fid} if self.config.require_finish_match else fid in hands
        return self._finish_ids.issubset(hands)

    def heuristic(self, state: HandState) -> float:
        """Admissible lower bound on the remaining cost.

        Counts the hand moves that must still happen -- at minimum, one per
        finish hold not yet occupied, and at minimum one per ``_max_gain``
        inches of vertical distance still to cover -- and prices each at the
        cheapest move the cost model can produce. ``_max_gain`` is deliberately
        an over-estimate of per-move progress so the bound stays below truth.
        """
        if not self._finish_ids or self.is_goal(state):
            return 0.0

        hands = [state.lh, state.rh]
        if len(self._finish_ids) == 1:
            fid = next(iter(self._finish_ids))
            on_it = [h for h in hands if h == fid]
            if self.config.require_finish_match:
                remaining = 2 - len(on_it)
            else:
                remaining = 0 if on_it else 1
            elsewhere = [h for h in hands if h != fid]
            if elsewhere:
                gap = min(abs(self.by_id[fid].y - self.by_id[h].y) for h in elsewhere)
                remaining = max(remaining, int(math.ceil(gap / self._max_gain)))
        else:
            missing = self._finish_ids - set(hands)
            remaining = len(missing)
            best_y = max(self.by_id[h].y for h in hands)
            gaps = [abs(self.by_id[m].y - best_y) for m in missing]
            if gaps:
                remaining = max(remaining, int(math.ceil(min(gaps) / self._max_gain)))

        return self.config.heuristic_weight * max(0, remaining) * self._min_move

    def _progress_gap(self, state: HandState) -> float:
        """Inches from the better-placed hand to the nearest finish hold.

        Used only to pick which partial beta to return when no complete
        solution was found. The A* heuristic cannot serve here: it is a coarse
        *admissible* bound and is flat across most of the state space, so
        minimising it would return the shallowest state rather than the
        furthest one.
        """
        if not self._finish_ids:
            return 0.0
        best = float("inf")
        for fid in self._finish_ids:
            f = self.by_id[fid]
            for hold_id in (state.lh, state.rh):
                h = self.by_id[hold_id]
                best = min(best, distance(h.position, f.position))
        return best

    # -------------------------------------------------------------- search

    def run(self) -> SearchResult:
        if self.config.strategy == "greedy":
            return self._run_greedy()
        return self._run_astar()

    def _base_warnings(self) -> List[str]:
        warnings: List[str] = []
        if not self.start_holds:
            warnings.append("No start holds tagged; used the lowest hand holds instead.")
        if not self.finish_holds:
            warnings.append("No finish holds tagged; used the highest hand holds instead.")
        return warnings

    def _run_astar(self) -> SearchResult:
        t0 = time.perf_counter()
        warnings = self._base_warnings()

        frontier: List[Tuple[float, int, float, HandState]] = []
        best_g: Dict[HandState, float] = {}
        came_from: Dict[HandState, Tuple[HandState, Transition]] = {}
        openings: Dict[HandState, List[MoveEval]] = {}
        depth: Dict[HandState, int] = {}
        tie = 0

        for state, opening in self.start_states():
            # Placing the feet from standing is itself part of the beta.
            initial_feet_moves = self._foot_transition({}, self._resolve_feet(state), state)
            g = sum(e.cost for e in opening) + sum(e.cost for e in initial_feet_moves)
            if state in best_g and best_g[state] <= g:
                continue
            best_g[state] = g
            openings[state] = opening + initial_feet_moves
            depth[state] = 0
            heapq.heappush(frontier, (g + self.heuristic(state), tie, g, state))
            tie += 1

        expansions = 0
        goal: Optional[HandState] = None
        truncated = False

        while frontier:
            _, _, g, state = heapq.heappop(frontier)
            if g > best_g.get(state, float("inf")) + 1e-9:
                continue
            if self.is_goal(state):
                goal = state
                break

            expansions += 1
            if expansions > self.config.max_expansions:
                truncated = True
                warnings.append(
                    f"Search hit the {self.config.max_expansions}-expansion cap; "
                    "returning the best partial beta found."
                )
                break

            d = depth.get(state, 0)
            if d >= self.config.max_moves:
                continue

            for nxt, transition in self._successors(state):
                ng = g + transition.cost
                if ng >= best_g.get(nxt, float("inf")) - 1e-9:
                    continue
                best_g[nxt] = ng
                came_from[nxt] = (state, transition)
                depth[nxt] = d + 1
                heapq.heappush(frontier, (ng + self.heuristic(nxt), tie, ng, nxt))
                tie += 1

        elapsed = (time.perf_counter() - t0) * 1000.0

        if goal is None:
            if not best_g:
                raise ValueError("no valid start position could be built for this climb")
            # Return the furthest progress made rather than nothing at all.
            goal = min(best_g, key=lambda s: (self._progress_gap(s), best_g[s]))
            truncated = True
            if not any("cap" in w for w in warnings):
                warnings.append(
                    "Could not reach the finish hold under this reach model; "
                    "returning the furthest sequence found."
                )

        moves = self._reconstruct(goal, came_from, openings)
        return SearchResult(
            moves=moves,
            total_cost=sum(m.cost for m in moves),
            expansions=expansions,
            elapsed_ms=elapsed,
            truncated=truncated,
            warnings=warnings,
            strategy="astar",
        )

    def _run_greedy(self) -> SearchResult:
        """Cheapest-progress-first walk. Fast, myopic; a useful baseline."""
        t0 = time.perf_counter()
        warnings = self._base_warnings()

        state, opening = self.start_states()[0]
        initial_feet = self._foot_transition({}, self._resolve_feet(state), state)

        chain: List[Transition] = []
        visited = {state}
        truncated = False
        top = max((self.by_id[f].y for f in self._finish_ids), default=float(self.ys.max()))

        for _ in range(self.config.max_moves):
            if self.is_goal(state):
                break
            options = [s for s in self._successors(state) if s[0] not in visited]
            if not options:
                truncated = True
                warnings.append("Greedy walk got stuck with no unvisited legal move.")
                break

            def score(item) -> float:
                _, transition = item
                hand_move = transition.hand_move
                origin = hand_move.from_hold_id or hand_move.hold_id
                gained = self.by_id[hand_move.hold_id].y - self.by_id[origin].y
                return transition.cost - 2.0 * max(0.0, gained) / max(top, 1.0)

            nxt, transition = min(options, key=score)
            chain.append(transition)
            state = nxt
            visited.add(state)
        else:
            truncated = True
            warnings.append(f"Greedy walk exceeded {self.config.max_moves} moves.")

        if not self.is_goal(state):
            truncated = True

        flat: List[MoveEval] = list(opening) + list(initial_feet)
        for t in chain:
            flat.extend(t.as_list())
        moves = self._finalise(flat)
        return SearchResult(
            moves=moves,
            total_cost=sum(m.cost for m in moves),
            expansions=len(chain),
            elapsed_ms=(time.perf_counter() - t0) * 1000.0,
            truncated=truncated,
            warnings=warnings,
            strategy="greedy",
        )

    # ------------------------------------------------------- path assembly

    def _reconstruct(
        self,
        goal: HandState,
        came_from: Dict[HandState, Tuple[HandState, Transition]],
        openings: Dict[HandState, List[MoveEval]],
    ) -> List[MoveEval]:
        chain: List[Transition] = []
        cur = goal
        guard = 0
        while cur in came_from:
            prev, transition = came_from[cur]
            chain.append(transition)
            cur = prev
            guard += 1
            if guard > self.config.max_moves * 4:
                break
        chain.reverse()

        flat: List[MoveEval] = list(openings.get(cur, []))
        for t in chain:
            flat.extend(t.as_list())
        return self._finalise(flat)

    def _finalise(self, moves: List[MoveEval]) -> List[MoveEval]:
        """Replay the path to attach the stance after each move, and re-cost
        it with human-readable explanations.

        Explanations are only generated here, so the search's inner loop never
        pays for string formatting.
        """
        if not moves:
            return moves

        # Trailing foot moves are always dead weight: the beta ends on the
        # finish hold, so nothing follows them that could benefit from the
        # better stance.
        while len(moves) > 2 and moves[-1].limb.is_foot:
            moves = moves[:-1]

        contacts: Contacts = {}
        last: Optional[Limb] = None
        out: List[MoveEval] = []

        for ev in moves:
            if ev.kind is MoveKind.START and ev.from_hold_id is None and ev.limb.is_hand:
                contacts[ev.limb] = ev.hold_id
                # A matched start records both hands on the same hold.
                other = ev.limb.opposite
                if other not in contacts:
                    contacts[other] = ev.hold_id
                out.append(ev)
                continue

            support = {l: hid for l, hid in contacts.items() if l is not ev.limb}
            explained = self._cost_move(
                support,
                ev.limb,
                self.by_id[ev.hold_id],
                from_hold=contacts.get(ev.limb),
                last_limb=last,
                explain=True,
            )
            out.append(explained if explained is not None else ev)
            contacts[ev.limb] = ev.hold_id
            last = ev.limb

        # Second pass: attach the resolved stance after each move.
        contacts = {}
        for i, ev in enumerate(out):
            if ev.kind is MoveKind.START and ev.from_hold_id is None and ev.limb.is_hand:
                contacts[ev.limb] = ev.hold_id
                other = ev.limb.opposite
                if other not in contacts:
                    contacts[other] = ev.hold_id
            else:
                contacts[ev.limb] = ev.hold_id
            ev.contacts_after = dict(contacts)
            ev.stance_after = self._stance(contacts)

        # A matched start leaves the opposite hand provisionally on the same
        # hold; correct that once the real second placement is known.
        starts = [e for e in out if e.kind is MoveKind.START]
        if len(starts) == 2 and starts[0].hold_id != starts[1].hold_id:
            fixed: Contacts = {starts[0].limb: starts[0].hold_id}
            starts[0].contacts_after = dict(fixed)
            starts[0].stance_after = self._stance(fixed)

        if out and self.is_goal(
            HandState(
                lh=out[-1].contacts_after.get(Limb.LH, -1),  # type: ignore[union-attr]
                rh=out[-1].contacts_after.get(Limb.RH, -1),  # type: ignore[union-attr]
            )
        ):
            # Mark the last *hand* move as the finish.
            for ev in reversed(out):
                if ev.limb.is_hand:
                    ev.kind = MoveKind.FINISH
                    break
        return out
