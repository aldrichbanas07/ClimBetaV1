"""Mapping raw beta cost onto a real climbing grade.

The search produces cost in arbitrary units. To say anything a climber
recognises we need a map from those units onto Kilter's numeric difficulty
scale (and from there to V / Font grades).

Two modes:

**Calibrated** -- if the ETL loaded real ``climb_stats`` rows, we have
(climb, angle) -> community difficulty for thousands of climbs. ``fit`` runs an
ordinary least-squares regression of a handful of beta features onto that
difficulty and persists the coefficients. Because it fits *per-angle*
observations, the angle dependence baked into the cost model gets checked
against reality rather than assumed.

**Uncalibrated** -- otherwise we fall back to a hand-set linear map, and every
response is flagged ``calibrated: false`` so nobody mistakes it for data.

Features were chosen to be the things climbers actually grade on: how hard the
hardest move is (boulder problems are graded by the crux), how sustained it is
(mean move cost), and how long it is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Fallback Kilter difficulty -> grade names, used when the app database's own
#: ``difficulty_grades`` table was not ingested.
FALLBACK_GRADES: Dict[int, str] = {
    10: "4a/V0", 11: "4b/V0", 12: "4c/V0", 13: "5a/V1", 14: "5b/V1",
    15: "5c/V2", 16: "6a/V3", 17: "6a+/V3", 18: "6b/V4", 19: "6b+/V4",
    20: "6c/V5", 21: "6c+/V5", 22: "7a/V6", 23: "7a+/V7", 24: "7b/V8",
    25: "7b+/V8", 26: "7c/V9", 27: "7c+/V10", 28: "8a/V11", 29: "8a+/V12",
    30: "8b/V13", 31: "8b+/V14", 32: "8c/V15", 33: "8c+/V16",
}

MIN_DIFFICULTY = 10.0
MAX_DIFFICULTY = 33.0

FEATURE_NAMES = ("crux_cost", "mean_cost", "log_moves", "bias")


def _nnls_with_free_intercept(
    X: np.ndarray,
    y: np.ndarray,
    free_index: int,
    iterations: int = 4000,
    tol: float = 1e-9,
) -> np.ndarray:
    """Least squares with all coefficients >= 0 except one.

    Projected gradient descent: plain enough to read, and adequate at this
    problem size (a handful of features, a few thousand rows). Avoids a scipy
    dependency for one function.
    """
    n_features = X.shape[1]
    # Start from the unconstrained solution, then project into the feasible set.
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    mask = np.ones(n_features, dtype=bool)
    mask[free_index] = False
    beta = np.where(mask, np.maximum(beta, 0.0), beta)

    gram = X.T @ X
    rhs = X.T @ y
    # 1/L step size, L = largest eigenvalue of the Gram matrix.
    norm = float(np.linalg.norm(gram, 2))
    step = 1.0 / norm if norm > 0 else 1e-3

    for _ in range(iterations):
        grad = gram @ beta - rhs
        nxt = beta - step * grad
        nxt = np.where(mask, np.maximum(nxt, 0.0), nxt)
        if float(np.max(np.abs(nxt - beta))) < tol:
            beta = nxt
            break
        beta = nxt
    return beta


def beta_features(move_costs: Sequence[float]) -> np.ndarray:
    """Feature vector for one generated beta. Order matches FEATURE_NAMES."""
    costs = np.asarray([c for c in move_costs], dtype=np.float64)
    if costs.size == 0:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return np.array(
        [
            float(costs.max()),
            float(costs.mean()),
            math.log1p(costs.size),
            1.0,
        ]
    )


@dataclass
class GradeCalibration:
    """Linear map from beta features to Kilter numeric difficulty."""

    #: Coefficients aligned with FEATURE_NAMES.
    coefficients: List[float] = field(
        # Hand-set fallback, fit (non-negative least squares) against six
        # hand-judged reference climbs spanning V0-V9 at 0-70 degrees -- see
        # ``scripts/fit_default_calibration.py``. NOT real community data;
        # `calibrated` stays False until a real Kilter database is ingested
        # and `etl calibrate` is re-run.
        default_factory=lambda: [0.0, 3.64, 3.02, 1.70]
    )
    calibrated: bool = False
    n_samples: int = 0
    rmse: Optional[float] = None
    grades: Dict[int, str] = field(default_factory=lambda: dict(FALLBACK_GRADES))

    # ----------------------------------------------------------------- use

    def difficulty(self, move_costs: Sequence[float]) -> float:
        x = beta_features(move_costs)
        raw = float(np.dot(np.asarray(self.coefficients, dtype=np.float64), x))
        return float(min(max(raw, MIN_DIFFICULTY), MAX_DIFFICULTY))

    def grade_name(self, difficulty: float) -> Optional[str]:
        if not self.grades:
            return None
        key = int(round(difficulty))
        if key in self.grades:
            return self.grades[key]
        available = sorted(self.grades)
        key = min(available, key=lambda k: abs(k - difficulty))
        return self.grades[key]

    # ----------------------------------------------------------------- fit

    def fit(
        self,
        samples: Sequence[Tuple[Sequence[float], float]],
        grades: Optional[Dict[int, str]] = None,
    ) -> "GradeCalibration":
        """Fit against (move_costs, observed_difficulty) pairs.

        ``samples`` should span a range of angles *and* a good number of
        distinct climbs. Per-angle rows from the same climb are strongly
        correlated, so 500 rows from 20 climbs carries far less information
        than the row count suggests.

        All coefficients except the bias are constrained non-negative. An
        unconstrained fit on too few climbs happily returns a *negative* crux
        weight -- i.e. "the harder the hardest move, the easier the climb" --
        which fits the noise and predicts nonsense on anything new.
        """
        usable = [(mc, d) for mc, d in samples if mc is not None and d is not None]
        if len(usable) < len(FEATURE_NAMES) + 2:
            raise ValueError(
                f"need at least {len(FEATURE_NAMES) + 2} samples to fit; got {len(usable)}"
            )

        X = np.vstack([beta_features(mc) for mc, _ in usable])
        y = np.asarray([d for _, d in usable], dtype=np.float64)

        # Bias is the last column and is left free to take any sign.
        coeffs = _nnls_with_free_intercept(X, y, free_index=len(FEATURE_NAMES) - 1)
        residuals = X @ coeffs - y

        self.coefficients = [float(c) for c in coeffs]
        self.calibrated = True
        self.n_samples = len(usable)
        self.rmse = float(np.sqrt(np.mean(residuals ** 2)))
        if grades:
            self.grades = dict(grades)
        return self

    # ---------------------------------------------------------------- i/o

    def to_dict(self) -> Dict[str, object]:
        return {
            "coefficients": self.coefficients,
            "feature_names": list(FEATURE_NAMES),
            "calibrated": self.calibrated,
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "grades": {str(k): v for k, v in self.grades.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "GradeCalibration":
        grades = {int(k): v for k, v in (data.get("grades") or {}).items()}
        out = cls(
            calibrated=bool(data.get("calibrated", False)),
            n_samples=int(data.get("n_samples", 0)),
            rmse=data.get("rmse"),
            grades=grades or dict(FALLBACK_GRADES),
        )
        # Only override the fallback coefficients if the file actually has a
        # full, correctly-sized set; a truncated file should not break scoring.
        coeffs = [float(c) for c in data.get("coefficients") or []]
        if len(coeffs) == len(FEATURE_NAMES):
            out.coefficients = coeffs
        return out

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "GradeCalibration":
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError, KeyError):
            return cls()
