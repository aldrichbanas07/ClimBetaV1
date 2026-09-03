-- Clean analysis schema for Kilter Board beta generation.
--
-- This is deliberately NOT a copy of the Kilter app's schema. The app database
-- is normalised for the app's needs (products, sets, LED positions, image
-- layers); we keep only what beta generation and rendering require, in one
-- shape that both the sample board and real ingested data fit.
--
-- Coordinates are inches, y increasing upward -- matching the app's holes
-- table so real data needs no transform. See domain/holds.py.

PRAGMA foreign_keys = ON;

-- One physical board configuration (e.g. Kilter Original 12x12).
CREATE TABLE IF NOT EXISTS layouts (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    product_name  TEXT,
    -- Bounding box of the hold field, cached for the frontend's SVG viewBox.
    min_x         REAL,
    max_x         REAL,
    min_y         REAL,
    max_y         REAL,
    source        TEXT    NOT NULL DEFAULT 'sample'  -- 'sample' | 'kilter'
);

-- Every hold position on a layout.
--
-- hold_id is our stable internal key. For ingested Kilter data it is the
-- placements.id, because that is what climb frame strings reference; keeping
-- them equal means a generated beta can be traced to physical hardware.
CREATE TABLE IF NOT EXISTS holds (
    hold_id           INTEGER PRIMARY KEY,
    layout_id         INTEGER NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    placement_id      INTEGER,
    hole_id           INTEGER,
    set_id            INTEGER,
    name              TEXT,
    x                 REAL    NOT NULL,
    y                 REAL    NOT NULL,
    -- Hold type is OURS, not Kilter's: the app database has no notion of
    -- crimp/jug/sloper. See etl/hold_types.py.
    hold_type         TEXT    NOT NULL DEFAULT 'unknown',
    hold_type_source  TEXT    NOT NULL DEFAULT 'default', -- manual|heuristic|default
    size              REAL    NOT NULL DEFAULT 3.0,
    default_role      TEXT
);

CREATE INDEX IF NOT EXISTS idx_holds_layout ON holds(layout_id);
CREATE INDEX IF NOT EXISTS idx_holds_xy     ON holds(layout_id, y, x);

CREATE TABLE IF NOT EXISTS climbs (
    climb_id     TEXT    PRIMARY KEY,   -- uuid for real data
    layout_id    INTEGER NOT NULL REFERENCES layouts(id) ON DELETE CASCADE,
    name         TEXT    NOT NULL,
    setter       TEXT,
    description  TEXT,
    -- The angle the setter intended, if any. Nullable: most Kilter climbs are
    -- graded at many angles and have no single canonical one.
    setter_angle INTEGER,
    is_listed    INTEGER NOT NULL DEFAULT 1,
    hold_count   INTEGER NOT NULL DEFAULT 0,
    -- Raw Kilter frames string, kept verbatim for provenance/debugging.
    frames       TEXT,
    source       TEXT    NOT NULL DEFAULT 'sample'
);

CREATE INDEX IF NOT EXISTS idx_climbs_layout ON climbs(layout_id, is_listed);

CREATE TABLE IF NOT EXISTS climb_holds (
    climb_id  TEXT    NOT NULL REFERENCES climbs(climb_id) ON DELETE CASCADE,
    hold_id   INTEGER NOT NULL REFERENCES holds(hold_id)  ON DELETE CASCADE,
    role      TEXT    NOT NULL,   -- start|hand|finish|foot|any
    PRIMARY KEY (climb_id, hold_id)
);

CREATE INDEX IF NOT EXISTS idx_climb_holds_climb ON climb_holds(climb_id);

-- Per-angle community difficulty. This table is the whole reason calibration
-- is possible: it gives ground truth for the angle dependence we model.
CREATE TABLE IF NOT EXISTS climb_stats (
    climb_id             TEXT    NOT NULL REFERENCES climbs(climb_id) ON DELETE CASCADE,
    angle                INTEGER NOT NULL,
    display_difficulty   REAL,
    benchmark_difficulty REAL,
    ascensionist_count   INTEGER,
    quality_average      REAL,
    PRIMARY KEY (climb_id, angle)
);

CREATE INDEX IF NOT EXISTS idx_climb_stats_angle ON climb_stats(angle, ascensionist_count);

-- Kilter's numeric difficulty -> human grade names.
CREATE TABLE IF NOT EXISTS difficulty_grades (
    difficulty   INTEGER PRIMARY KEY,
    boulder_name TEXT,
    route_name   TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
