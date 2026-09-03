"""Paths and runtime settings.

Everything is overridable by environment variable so the same code runs
against the sample board and against a real Kilter app database.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: Repository root (…/kilterbeta/config.py -> backend/ -> repo root).
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.environ.get("KILTERBETA_DATA_DIR", ROOT / "data"))


@dataclass(frozen=True)
class Settings:
    #: Our cleaned analysis database, produced by the ETL.
    db_path: Path = DATA_DIR / "kilterbeta.sqlite3"
    #: Optional source: a copy of the Kilter Board app's own db.sqlite3.
    kilter_db_path: Path = DATA_DIR / "kilter" / "db.sqlite3"
    #: Fitted score->grade coefficients, if calibration has been run.
    calibration_path: Path = DATA_DIR / "calibration.json"
    #: Hand-curated hold-type overrides.
    hold_types_csv: Path = DATA_DIR / "hold_types.csv"
    #: Where the ETL writes an inspectable dump of the sample board.
    sample_dir: Path = DATA_DIR / "sample"

    default_angle: int = 40
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    @classmethod
    def from_env(cls) -> "Settings":
        def p(key: str, default: Path) -> Path:
            raw = os.environ.get(key)
            return Path(raw) if raw else default

        base = cls()
        return cls(
            db_path=p("KILTERBETA_DB", base.db_path),
            kilter_db_path=p("KILTER_APP_DB", base.kilter_db_path),
            calibration_path=p("KILTERBETA_CALIBRATION", base.calibration_path),
            hold_types_csv=p("KILTERBETA_HOLD_TYPES", base.hold_types_csv),
            sample_dir=p("KILTERBETA_SAMPLE_DIR", base.sample_dir),
            default_angle=int(os.environ.get("KILTERBETA_DEFAULT_ANGLE", base.default_angle)),
            cors_origins=os.environ.get("KILTERBETA_CORS_ORIGINS", base.cors_origins),
        )

    @property
    def cors_origin_list(self):
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings.from_env()
