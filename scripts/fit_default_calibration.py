"""Derive the hand-set fallback coefficients in ``GradeCalibration.coefficients``.

This is NOT the real calibration path -- that is ``kilterbeta.etl.cli
calibrate``, which fits against actual Kilter ``climb_stats`` community
difficulty once a real database has been ingested. This script only produces
a reasonable *placeholder* by fitting against six hand-judged reference grades
for the synthetic sample climbs, so the demo shows plausible numbers before
any real data exists.

Run after changing the difficulty model, to see whether the fallback
coefficients drift:

    PYTHONPATH=backend python scripts/fit_default_calibration.py

Then copy the printed coefficients into ``GradeCalibration.coefficients``'s
default if they moved meaningfully.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

import numpy as np

from kilterbeta.beta.calibration import FEATURE_NAMES, _nnls_with_free_intercept, beta_features
from kilterbeta.beta.generator import generate_beta
from kilterbeta.config import settings
from kilterbeta.db.connection import connect
from kilterbeta.db.repository import Repository
from kilterbeta.domain.moves import MoveKind

# Hand-judged Kilter difficulty for the synthetic demo climbs, chosen to span
# a wide grade and angle range. These are opinions, not measurements -- the
# whole point of real calibration is to replace them.
TARGETS = {
    "sample-001": {0: 10, 20: 12, 40: 14, 55: 16, 70: 18},   # Jug Ladder
    "sample-002": {0: 14, 20: 17, 40: 20, 55: 23, 70: 25},   # Crimp Ladder
    "sample-003": {0: 15, 20: 18, 40: 21, 55: 24, 70: 27},   # Sloper Traverse
    "sample-004": {0: 12, 20: 15, 40: 18, 55: 20, 70: 22},   # Big Moves
    "sample-005": {0: 14, 20: 17, 40: 19, 55: 21, 70: 23},   # Pinch Power
    "sample-006": {0: 13, 20: 15, 40: 17, 55: 19, 70: 21},   # Technical Feet
}


def main() -> int:
    if not settings.db_path.exists():
        print(f"error: {settings.db_path} not found. Run init-sample first.", file=sys.stderr)
        return 2

    conn = connect(settings.db_path, read_only=True)
    repo = Repository(conn)

    rows, ys, meta = [], [], []
    for climb_id, per_angle in TARGETS.items():
        holds = repo.climb_holds(climb_id)
        for angle, target in per_angle.items():
            beta = generate_beta(holds, angle=angle, climb_id=climb_id)
            costs = [m.difficulty for m in beta.moves if m.kind is not MoveKind.START]
            rows.append(beta_features(costs))
            ys.append(target)
            meta.append((climb_id, angle, beta.generator["truncated"]))

    X = np.vstack(rows)
    y = np.asarray(ys, dtype=np.float64)
    coeffs = _nnls_with_free_intercept(X, y, free_index=len(FEATURE_NAMES) - 1)
    pred = X @ coeffs
    err = pred - y

    print("feature ranges:")
    for i, name in enumerate(FEATURE_NAMES):
        print(f"  {name:12s} {X[:, i].min():7.3f} .. {X[:, i].max():7.3f}")
    print()
    print("coefficients:", [round(float(c), 4) for c in coeffs])
    print(f"RMSE {float(np.sqrt((err ** 2).mean())):.3f}  max|err| {float(np.abs(err).max()):.2f}")
    print()
    for (climb_id, angle, truncated), p, t in zip(meta, pred, y):
        flag = " TRUNCATED" if truncated else ""
        print(f"  {climb_id:12s} a={angle:2d} target={t:4.1f} pred={p:5.1f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
