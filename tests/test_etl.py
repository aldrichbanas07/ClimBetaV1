import sqlite3

import pytest

from kilterbeta.db.connection import connect, get_meta, init_schema, set_meta
from kilterbeta.db.repository import Repository
from kilterbeta.domain.holds import HoldRole, HoldType
from kilterbeta.etl import loader, sample_source
from kilterbeta.etl.hold_types import classify, parse_hold_type
from kilterbeta.etl.kilter_source import parse_frames, role_from_name


def test_parse_hold_type_aliases():
    assert parse_hold_type("Jug") is HoldType.JUG
    assert parse_hold_type("bucket") is HoldType.JUG
    assert parse_hold_type("mono") is HoldType.POCKET
    assert parse_hold_type("screw-on") is HoldType.FOOT_CHIP
    assert parse_hold_type(None) is HoldType.UNKNOWN
    assert parse_hold_type("nonsense-value") is HoldType.UNKNOWN


def test_classify_uses_override_first():
    overrides = {5: __import__(
        "kilterbeta.etl.hold_types", fromlist=["Classification"]
    ).Classification(HoldType.CRIMP, 1.5, "manual")}
    result = classify(5, overrides, set_name="screw ons", default_role="foot")
    assert result.hold_type is HoldType.CRIMP
    assert result.source == "manual"


def test_classify_falls_back_to_set_name_hint():
    result = classify(9, {}, set_name="Screw-On Kit", default_role="hand")
    assert result.hold_type is HoldType.FOOT_CHIP
    assert result.source == "heuristic"


def test_classify_falls_back_to_default_role():
    result = classify(9, {}, set_name=None, default_role="foot")
    assert result.hold_type is HoldType.FOOT_CHIP


def test_classify_defaults_to_unknown():
    result = classify(9, {}, set_name=None, default_role=None)
    assert result.hold_type is HoldType.UNKNOWN
    assert result.source == "default"


def test_parse_frames_extracts_placement_role_pairs():
    assert parse_frames("p1123r15p1145r13") == [(1123, 15), (1145, 13)]


def test_parse_frames_handles_empty():
    assert parse_frames(None) == []
    assert parse_frames("") == []


def test_role_from_name_prefers_name_over_id():
    assert role_from_name("Start", role_id=99) is HoldRole.START
    assert role_from_name("foot only", role_id=99) is HoldRole.FOOT


def test_role_from_name_falls_back_to_known_ids():
    assert role_from_name(None, role_id=12) is HoldRole.START
    assert role_from_name(None, role_id=15) is HoldRole.FOOT


def test_role_from_name_defaults_to_hand():
    assert role_from_name(None, role_id=None) is HoldRole.HAND
    assert role_from_name("mystery", role_id=None) is HoldRole.HAND


def test_sample_board_has_no_duplicate_hold_ids():
    board = sample_source.build_board()
    ids = [h.hold_id for h in board]
    assert len(ids) == len(set(ids))


def test_sample_board_is_deterministic():
    a = sample_source.build_board()
    b = sample_source.build_board()
    assert [(h.hold_id, h.hold_type) for h in a] == [(h.hold_id, h.hold_type) for h in b]


def test_sample_climbs_resolve_to_holds_on_the_board():
    board = sample_source.build_board()
    board_ids = {h.hold_id for h in board}
    for spec in sample_source.sample_climbs():
        resolved = sample_source.resolve_climb(spec, board)
        assert len(resolved) >= 3
        assert all(hid in board_ids for hid, _ in resolved)
        roles = {role for _, role in resolved}
        assert HoldRole.START in roles
        assert HoldRole.FINISH in roles


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.sqlite3"
    conn = connect(path)
    init_schema(conn)
    yield conn
    conn.close()


def test_loader_upsert_layout_and_holds_roundtrip(db):
    board = sample_source.build_board()[:5]
    loader.upsert_layout(db, 1, "Test Layout", board, source="sample")
    loader.upsert_holds(db, 1, board)
    db.commit()

    repo = Repository(db)
    layout = repo.get_layout(1)
    assert layout is not None
    assert layout.name == "Test Layout"

    holds = repo.layout_holds(1)
    assert len(holds) == 5


def test_loader_upsert_is_idempotent(db):
    board = sample_source.build_board()[:5]
    loader.upsert_layout(db, 1, "Test Layout", board)
    loader.upsert_holds(db, 1, board)
    loader.upsert_layout(db, 1, "Test Layout Renamed", board)
    loader.upsert_holds(db, 1, board)
    db.commit()

    repo = Repository(db)
    assert repo.get_layout(1).name == "Test Layout Renamed"
    assert len(repo.layout_holds(1)) == 5  # not duplicated


def test_loader_upsert_climb_replaces_hold_set(db):
    board = sample_source.build_board()[:5]
    loader.upsert_layout(db, 1, "Test Layout", board)
    loader.upsert_holds(db, 1, board)

    ids = [h.hold_id for h in board]
    loader.upsert_climb(db, "c1", 1, "Climb", [(ids[0], HoldRole.START), (ids[1], HoldRole.FINISH)])
    db.commit()
    assert len(Repository(db).climb_holds("c1")) == 2

    loader.upsert_climb(db, "c1", 1, "Climb", [(ids[2], HoldRole.START), (ids[3], HoldRole.FINISH)])
    db.commit()
    holds = Repository(db).climb_holds("c1")
    assert {h.hold_id for h in holds} == {ids[2], ids[3]}


def test_meta_set_and_get(db):
    set_meta(db, "foo", "bar")
    db.commit()
    assert get_meta(db, "foo") == "bar"
    assert get_meta(db, "missing") is None


def test_connect_read_only_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        connect(tmp_path / "nope.sqlite3", read_only=True)
