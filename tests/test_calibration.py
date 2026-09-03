import json

import numpy as np
import pytest

from kilterbeta.beta.calibration import GradeCalibration, beta_features


def test_beta_features_empty_costs():
    f = beta_features([])
    assert f[-1] == 1.0  # bias term always present
    assert f[0] == 0.0 and f[1] == 0.0


def test_beta_features_crux_is_max_not_mean():
    f = beta_features([1.0, 1.0, 5.0])
    assert f[0] == 5.0
    assert f[1] == pytest.approx(7.0 / 3.0)


def test_uncalibrated_by_default():
    calib = GradeCalibration()
    assert calib.calibrated is False
    assert calib.difficulty([1.0, 2.0, 3.0]) >= 0


def test_difficulty_clamped_to_valid_range():
    calib = GradeCalibration(coefficients=[1000.0, 0.0, 0.0, 0.0])
    assert calib.difficulty([1.0]) <= 33.0
    calib_low = GradeCalibration(coefficients=[0.0, 0.0, 0.0, -1000.0])
    assert calib_low.difficulty([1.0]) >= 10.0


def test_grade_name_nearest_match():
    calib = GradeCalibration()
    name = calib.grade_name(20.4)
    assert name is not None


def test_fit_requires_minimum_samples():
    calib = GradeCalibration()
    with pytest.raises(ValueError):
        calib.fit([([1.0], 15.0)])


def test_fit_produces_nonnegative_weighted_features():
    # Synthetic, noiseless data: difficulty is exactly a positive combination
    # of crux and mean cost.
    rng_costs = [
        [1.0, 1.0],
        [2.0, 1.5],
        [3.0, 2.0],
        [4.0, 2.5],
        [5.0, 3.0],
        [6.0, 3.5],
        [7.0, 4.0],
        [8.0, 4.5],
    ]
    samples = []
    for costs in rng_costs:
        crux = max(costs)
        mean = sum(costs) / len(costs)
        difficulty = 10.0 + 2.0 * crux + 1.0 * mean
        samples.append((costs, difficulty))

    calib = GradeCalibration().fit(samples)
    assert calib.calibrated is True
    assert calib.n_samples == len(samples)
    # crux and mean_cost coefficients must be non-negative.
    assert calib.coefficients[0] >= 0.0
    assert calib.coefficients[1] >= 0.0
    assert calib.coefficients[2] >= 0.0
    assert calib.rmse is not None and calib.rmse < 2.0


def test_save_and_load_roundtrip(tmp_path):
    calib = GradeCalibration(coefficients=[1.0, 2.0, 3.0, 4.0], calibrated=True, n_samples=42)
    path = tmp_path / "calibration.json"
    calib.save(path)

    loaded = GradeCalibration.load(path)
    assert loaded.coefficients == calib.coefficients
    assert loaded.calibrated is True
    assert loaded.n_samples == 42


def test_load_missing_file_returns_default(tmp_path):
    calib = GradeCalibration.load(tmp_path / "does_not_exist.json")
    assert calib.calibrated is False


def test_load_corrupt_file_returns_default(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("not json{{{", encoding="utf-8")
    calib = GradeCalibration.load(path)
    assert calib.calibrated is False


def test_load_truncated_coefficients_keeps_fallback(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"coefficients": [1.0, 2.0], "calibrated": True}), encoding="utf-8")
    calib = GradeCalibration.load(path)
    # Truncated coefficient list must not silently corrupt scoring.
    assert len(calib.coefficients) == len(GradeCalibration().coefficients)
