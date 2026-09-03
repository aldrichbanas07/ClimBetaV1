"""API tests against a small temp database built with the real ETL path."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kilterbeta.config import Settings
from kilterbeta.db.connection import connect, init_schema
from kilterbeta.domain.holds import HoldRole
from kilterbeta.etl import loader, sample_source

from .conftest import make_hold


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "api_test.sqlite3"
    conn = connect(db_path)
    init_schema(conn)

    holds = [
        make_hold(1, 40.0, 20.0, role=HoldRole.START),
        make_hold(2, 60.0, 20.0, role=HoldRole.START),
        make_hold(3, 50.0, 60.0, role=HoldRole.HAND),
        make_hold(4, 50.0, 100.0, role=HoldRole.FINISH),
        make_hold(5, 40.0, 2.0, role=HoldRole.FOOT),
        make_hold(6, 60.0, 2.0, role=HoldRole.FOOT),
    ]
    loader.upsert_layout(conn, 1, "Test Layout", holds, source="sample")
    loader.upsert_holds(conn, 1, holds)
    loader.upsert_climb(
        conn, "climb-1", 1, "Test Climb",
        [(h.hold_id, h.role) for h in holds],
        setter="tester",
    )
    conn.commit()
    conn.close()

    test_settings = Settings(
        db_path=db_path,
        calibration_path=tmp_path / "calibration.json",
        hold_types_csv=tmp_path / "hold_types.csv",
        sample_dir=tmp_path / "sample",
    )

    import kilterbeta.api.app as app_module

    monkeypatch.setattr(app_module, "settings", test_settings)
    monkeypatch.setattr("kilterbeta.config.settings", test_settings)
    monkeypatch.setattr(app_module, "_calibration", None)

    return TestClient(app_module.app)


def test_health_reports_database_present(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["database_present"] is True
    assert body["counts"]["climbs"] == 1


def test_list_layouts(client):
    resp = client.get("/layouts")
    assert resp.status_code == 200
    layouts = resp.json()
    assert len(layouts) == 1
    assert layouts[0]["name"] == "Test Layout"


def test_layout_holds_returns_all_holds(client):
    resp = client.get("/layouts/1/holds")
    assert resp.status_code == 200
    assert len(resp.json()) == 6


def test_layout_holds_404_for_missing_layout(client):
    resp = client.get("/layouts/999/holds")
    assert resp.status_code == 404


def test_list_climbs(client):
    resp = client.get("/climbs")
    assert resp.status_code == 200
    climbs = resp.json()
    assert len(climbs) == 1
    assert climbs[0]["climb_id"] == "climb-1"


def test_get_climb_detail(client):
    resp = client.get("/climbs/climb-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["climb"]["name"] == "Test Climb"
    assert len(body["holds"]) == 6


def test_get_climb_404(client):
    resp = client.get("/climbs/does-not-exist")
    assert resp.status_code == 404


def test_generate_beta_by_climb_id(client):
    resp = client.post("/generate-beta", json={"climb_id": "climb-1", "angle": 40})
    assert resp.status_code == 200
    body = resp.json()
    assert body["angle"] == 40
    assert body["climb_id"] == "climb-1"
    assert len(body["moves"]) > 0
    assert body["schema_version"]


def test_generate_beta_by_hold_list(client):
    resp = client.post(
        "/generate-beta",
        json={
            "angle": 30,
            "hold_list": [
                {"hold_id": 1, "x": 40.0, "y": 20.0, "hold_type": "jug", "role": "start"},
                {"hold_id": 2, "x": 60.0, "y": 20.0, "hold_type": "jug", "role": "start"},
                {"hold_id": 3, "x": 50.0, "y": 80.0, "hold_type": "jug", "role": "finish"},
            ],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["moves"]


def test_generate_beta_rejects_both_climb_id_and_hold_list(client):
    resp = client.post(
        "/generate-beta",
        json={"climb_id": "climb-1", "hold_list": [], "angle": 40},
    )
    assert resp.status_code == 422


def test_generate_beta_rejects_neither(client):
    resp = client.post("/generate-beta", json={"angle": 40})
    assert resp.status_code == 422


def test_generate_beta_rejects_out_of_range_angle(client):
    resp = client.post("/generate-beta", json={"climb_id": "climb-1", "angle": 90})
    assert resp.status_code == 422


def test_generate_beta_404_for_missing_climb(client):
    resp = client.post("/generate-beta", json={"climb_id": "nope", "angle": 40})
    assert resp.status_code == 404


def test_generate_beta_rejects_too_few_holds(client):
    resp = client.post(
        "/generate-beta",
        json={
            "angle": 40,
            "hold_list": [{"hold_id": 1, "x": 0.0, "y": 0.0, "role": "start"}],
        },
    )
    assert resp.status_code == 422


def test_generate_beta_rejects_duplicate_hold_ids(client):
    resp = client.post(
        "/generate-beta",
        json={
            "angle": 40,
            "hold_list": [
                {"hold_id": 1, "x": 0.0, "y": 0.0, "role": "start"},
                {"hold_id": 1, "x": 10.0, "y": 10.0, "role": "finish"},
            ],
        },
    )
    assert resp.status_code == 422


def test_move_schema_endpoint_reports_version(client):
    resp = client.get("/schema/move")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"]
    assert "json_schema" in body


def test_difficulty_model_endpoint(client):
    resp = client.get("/difficulty-model", params={"angle": 45})
    assert resp.status_code == 200
    body = resp.json()
    assert body["angle"] == 45
    assert "jug" in body["hands"]
    assert body["hands"]["jug"]["cost"] < body["hands"]["crimp"]["cost"]
