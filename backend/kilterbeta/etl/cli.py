"""ETL command line.

    python -m kilterbeta.etl.cli init-sample          # build the demo database
    python -m kilterbeta.etl.cli ingest-kilter        # ingest a real app db
    python -m kilterbeta.etl.cli export-hold-types    # curation template
    python -m kilterbeta.etl.cli calibrate            # fit score -> grade
    python -m kilterbeta.etl.cli stats                # what's in the database
    python -m kilterbeta.etl.cli layouts              # list layouts in an app db
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

from ..config import settings
from ..db.connection import connect, init_schema, set_meta
from ..db.repository import Repository
from ..domain.holds import HoldRole
from . import loader, sample_source
from .hold_types import load_overrides, write_template


# --------------------------------------------------------------- init-sample


def cmd_init_sample(args: argparse.Namespace) -> int:
    db_path = Path(args.db or settings.db_path)
    conn = connect(db_path)
    try:
        init_schema(conn)

        board = sample_source.build_board()
        # Sample hold types are authored in code, but a curated CSV still wins
        # so the override path is exercised on the demo data too.
        overrides = load_overrides(Path(args.hold_types or settings.hold_types_csv))
        type_sources = {}
        patched = []
        for h in board:
            override = overrides.get(h.hold_id)
            if override is not None:
                from dataclasses import replace

                patched.append(replace(h, hold_type=override.hold_type, size=override.size))
                type_sources[h.hold_id] = "manual"
            else:
                patched.append(h)
                type_sources[h.hold_id] = "sample"
        board = patched

        loader.upsert_layout(
            conn,
            sample_source.SAMPLE_LAYOUT_ID,
            sample_source.SAMPLE_LAYOUT_NAME,
            board,
            product_name="Synthetic demo board",
            source="sample",
        )
        n_holds = loader.upsert_holds(
            conn, sample_source.SAMPLE_LAYOUT_ID, board, type_sources=type_sources
        )

        specs = sample_source.sample_climbs()
        n_stats = 0
        for spec in specs:
            climb_holds = sample_source.resolve_climb(spec, board)
            loader.upsert_climb(
                conn,
                climb_id=spec.climb_id,
                layout_id=sample_source.SAMPLE_LAYOUT_ID,
                name=spec.name,
                holds=climb_holds,
                setter=spec.setter,
                description=spec.description,
                setter_angle=spec.setter_angle,
                source="sample",
            )
            if not args.no_synthetic_stats:
                per_angle = {
                    angle: {
                        "display_difficulty": d,
                        "benchmark_difficulty": None,
                        # Flagged as synthetic via the layout's source column;
                        # a nominal count keeps the calibration filter happy.
                        "ascensionist_count": 25,
                        "quality_average": 3.0,
                    }
                    for angle, d in sample_source.synthetic_stats(
                        spec, [h for h, _ in climb_holds]
                    ).items()
                }
                n_stats += loader.upsert_stats(conn, spec.climb_id, per_angle)

        from ..beta.calibration import FALLBACK_GRADES

        loader.upsert_grades(conn, {d: (name, None) for d, name in FALLBACK_GRADES.items()})
        set_meta(conn, "sample_stats_are_synthetic", "1" if not args.no_synthetic_stats else "0")
        loader.record_ingest(conn, "sample", f"{n_holds} holds, {len(specs)} climbs")
        conn.commit()

        if args.dump:
            _dump_sample(board, specs, Path(args.dump_dir or settings.sample_dir))

        counts = Repository(conn).counts()
        print(f"Sample database written to {db_path}")
        print(f"  holds={counts['holds']} climbs={counts['climbs']} stats={counts['climb_stats']}")
        if not args.no_synthetic_stats:
            print("  NOTE: per-angle difficulties are SYNTHETIC placeholders, not community data.")
        return 0
    finally:
        conn.close()


def _dump_sample(board, specs, out_dir: Path) -> None:
    """Write the generated board to CSV/JSON so it can be eyeballed or edited."""
    out_dir.mkdir(parents=True, exist_ok=True)
    holds_csv = out_dir / "holds.csv"
    with holds_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["hold_id", "x", "y", "hold_type", "default_role", "size", "name"])
        for h in board:
            w.writerow([h.hold_id, h.x, h.y, h.hold_type.value, h.role.value, h.size, h.name])

    climbs_json = out_dir / "climbs.json"
    payload = []
    for spec in specs:
        resolved = sample_source.resolve_climb(spec, board)
        payload.append(
            {
                "climb_id": spec.climb_id,
                "name": spec.name,
                "description": spec.description,
                "setter_angle": spec.setter_angle,
                "holds": [{"hold_id": hid, "role": role.value} for hid, role in resolved],
            }
        )
    climbs_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  dumped {holds_csv.name} and {climbs_json.name} to {out_dir}")


# ------------------------------------------------------------ ingest-kilter


def cmd_ingest_kilter(args: argparse.Namespace) -> int:
    from .kilter_source import KilterSource

    src_path = Path(args.source or settings.kilter_db_path)
    db_path = Path(args.db or settings.db_path)

    try:
        source = KilterSource(src_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    with source:
        layout_id = args.layout_id or source.default_layout_id()
        layout = next((l for l in source.layouts() if l.layout_id == layout_id), None)
        if layout is None:
            print(f"error: layout {layout_id} not found in {src_path}", file=sys.stderr)
            return 2

        overrides = load_overrides(Path(args.hold_types or settings.hold_types_csv))
        holds, type_sources, set_ids = source.holds(layout_id, overrides=overrides)
        if not holds:
            print(f"error: layout {layout_id} has no placements with coordinates", file=sys.stderr)
            return 2

        conn = connect(db_path)
        try:
            init_schema(conn)
            loader.upsert_layout(
                conn,
                layout_id,
                layout.name,
                holds,
                product_name=layout.product_name,
                source="kilter",
            )
            n_holds = loader.upsert_holds(
                conn, layout_id, holds, type_sources=type_sources, set_ids=set_ids
            )

            known_holds = {h.hold_id for h in holds}
            n_climbs = 0
            skipped_unknown = 0
            for climb in source.climbs(
                layout_id, limit=args.limit, min_ascents=args.min_ascents
            ):
                pairs = [
                    (hid, role)
                    for hid, role in climb["holds"]  # type: ignore[index]
                    if hid in known_holds
                ]
                if len(pairs) < 2:
                    skipped_unknown += 1
                    continue
                loader.upsert_climb(
                    conn,
                    climb_id=climb["climb_id"],       # type: ignore[arg-type]
                    layout_id=layout_id,
                    name=climb["name"],               # type: ignore[arg-type]
                    holds=pairs,
                    setter=climb["setter"],           # type: ignore[arg-type]
                    description=climb["description"], # type: ignore[arg-type]
                    setter_angle=climb["setter_angle"],  # type: ignore[arg-type]
                    frames=climb["frames"],           # type: ignore[arg-type]
                    source="kilter",
                )
                n_climbs += 1

            ingested = {
                row["climb_id"]
                for row in conn.execute("SELECT climb_id FROM climbs").fetchall()
            }
            n_stats = 0
            for climb_id, angle, data in source.stats(layout_id):
                if climb_id in ingested:
                    n_stats += loader.upsert_stats(conn, climb_id, {angle: data})

            grades = source.difficulty_grades()
            if grades:
                loader.upsert_grades(conn, grades)

            loader.record_ingest(
                conn, "kilter", f"layout={layout_id} holds={n_holds} climbs={n_climbs}"
            )
            conn.commit()

            unknown = conn.execute(
                "SELECT COUNT(*) AS n FROM holds WHERE layout_id = ? AND hold_type = 'unknown'",
                (layout_id,),
            ).fetchone()["n"]

            print(f"Ingested layout {layout_id} ({layout.name}) into {db_path}")
            print(f"  holds={n_holds} climbs={n_climbs} stats_rows={n_stats} grades={len(grades)}")
            if skipped_unknown:
                print(f"  skipped {skipped_unknown} climbs referencing unknown placements")
            if unknown:
                pct = 100.0 * unknown / max(n_holds, 1)
                print(
                    f"  WARNING: {unknown} holds ({pct:.0f}%) have hold_type='unknown'.\n"
                    f"  The Kilter database does not store hold type. Run\n"
                    f"    python -m kilterbeta.etl.cli export-hold-types --out {settings.hold_types_csv}\n"
                    f"  label the CSV, then re-run this ingest to pick the labels up."
                )
            return 0
        finally:
            conn.close()


# -------------------------------------------------------------- other cmds


def cmd_layouts(args: argparse.Namespace) -> int:
    from .kilter_source import KilterSource

    try:
        with KilterSource(Path(args.source or settings.kilter_db_path)) as source:
            print(f"{'id':>6}  {'placements':>10}  name")
            for l in source.layouts():
                product = f" [{l.product_name}]" if l.product_name else ""
                print(f"{l.layout_id:>6}  {l.n_placements:>10}  {l.name}{product}")
        return 0
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def cmd_export_hold_types(args: argparse.Namespace) -> int:
    db_path = Path(args.db or settings.db_path)
    out = Path(args.out or settings.hold_types_csv)
    conn = connect(db_path, read_only=True)
    try:
        where = "WHERE layout_id = ?" if args.layout_id else ""
        params = (args.layout_id,) if args.layout_id else ()
        rows = conn.execute(
            f"""
            SELECT h.hold_id, h.name, h.x, h.y, h.hold_type AS current_type, h.size
            FROM holds h {where} ORDER BY h.y DESC, h.x
            """,
            params,
        ).fetchall()

        def gen():
            for r in rows:
                if args.only_unknown and r["current_type"] != "unknown":
                    continue
                yield {
                    "hold_id": r["hold_id"],
                    "hold_type": "",  # to be filled in by hand
                    "size": "",
                    "name": r["name"] or "",
                    "set_name": "",
                    "x": r["x"],
                    "y": r["y"],
                    "current_type": r["current_type"],
                }

        n = write_template(out, gen())
        print(f"Wrote {n} rows to {out}")
        print("Fill in the 'hold_type' column (jug/edge/crimp/sloper/pinch/pocket/foot_chip),")
        print("then re-run the ingest to apply them.")
        return 0
    finally:
        conn.close()


def cmd_calibrate(args: argparse.Namespace) -> int:
    """Fit the score -> Kilter-difficulty map against per-angle grade data."""
    from ..beta.calibration import GradeCalibration
    from ..beta.generator import generate_beta
    from ..beta.search import SearchConfig
    from ..domain.moves import MoveKind

    db_path = Path(args.db or settings.db_path)
    conn = connect(db_path, read_only=True)
    try:
        repo = Repository(conn)
        grades = repo.difficulty_grades()
        samples: List = []
        seen_climbs = 0
        failures = 0

        rows = list(repo.calibration_samples(min_ascents=args.min_ascents, limit=args.limit))
        if not rows:
            print(
                "error: no climb_stats rows with enough ascents to fit against.\n"
                "Ingest a real Kilter database first, or run init-sample "
                "(which writes synthetic stats).",
                file=sys.stderr,
            )
            return 2

        holds_cache = {}
        for climb_id, angle, difficulty in rows:
            if climb_id not in holds_cache:
                holds_cache[climb_id] = repo.climb_holds(climb_id)
                seen_climbs += 1
            holds = holds_cache[climb_id]
            if len(holds) < 3:
                continue
            try:
                beta = generate_beta(
                    holds,
                    angle=angle,
                    config=SearchConfig(angle=angle, max_expansions=args.max_expansions),
                )
            except (ValueError, AssertionError):
                failures += 1
                continue
            costs = [m.difficulty for m in beta.moves if m.kind is not MoveKind.START]
            if costs:
                samples.append((costs, difficulty))

        if len(samples) < 6:
            print(
                f"error: only {len(samples)} usable samples; need at least 6 to fit.",
                file=sys.stderr,
            )
            return 2

        calib = GradeCalibration().fit(samples, grades=grades or None)
        out = Path(args.out or settings.calibration_path)
        calib.save(out)
        print(f"Fitted on {calib.n_samples} (climb, angle) samples from {seen_climbs} climbs")
        print(f"  coefficients: {[round(c, 4) for c in calib.coefficients]}")
        print(f"  RMSE: {calib.rmse:.3f} Kilter difficulty points "
              f"(~{(calib.rmse or 0) / 2:.1f} V-grades)")
        if failures:
            print(f"  {failures} climbs could not be planned and were skipped")
        print(f"  saved to {out}")
        return 0
    finally:
        conn.close()


def cmd_stats(args: argparse.Namespace) -> int:
    db_path = Path(args.db or settings.db_path)
    if not db_path.exists():
        print(f"error: {db_path} does not exist. Run init-sample first.", file=sys.stderr)
        return 2
    conn = connect(db_path, read_only=True)
    try:
        repo = Repository(conn)
        for table, n in repo.counts().items():
            print(f"{table:>12}: {n}")
        print("\nlayouts:")
        for l in repo.list_layouts():
            print(
                f"  {l.id}  {l.name}  [{l.source}]  "
                f"x {l.min_x:.0f}..{l.max_x:.0f}  y {l.min_y:.0f}..{l.max_y:.0f}"
            )
        rows = conn.execute(
            "SELECT hold_type, COUNT(*) AS n FROM holds GROUP BY hold_type ORDER BY n DESC"
        ).fetchall()
        print("\nhold types:")
        for r in rows:
            print(f"  {r['hold_type']:>10}: {r['n']}")
        return 0
    finally:
        conn.close()


# ---------------------------------------------------------------- argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kilterbeta-etl", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("init-sample", help="build the synthetic demo database")
    s.add_argument("--db", help=f"output database (default {settings.db_path})")
    s.add_argument("--hold-types", help="hold-type override CSV")
    s.add_argument("--no-synthetic-stats", action="store_true",
                   help="skip the invented per-angle difficulties")
    s.add_argument("--dump", action="store_true", help="also write CSV/JSON of the board")
    s.add_argument("--dump-dir", help="where to write the dump")
    s.set_defaults(func=cmd_init_sample)

    s = sub.add_parser("ingest-kilter", help="ingest a real Kilter app database")
    s.add_argument("--source", help=f"path to the app's db.sqlite3 (default {settings.kilter_db_path})")
    s.add_argument("--db", help="output database")
    s.add_argument("--layout-id", type=int, help="layout to ingest (default: the largest)")
    s.add_argument("--limit", type=int, help="cap the number of climbs")
    s.add_argument("--min-ascents", type=int, default=0,
                   help="skip climbs whose best per-angle ascent count is below this")
    s.add_argument("--hold-types", help="hold-type override CSV")
    s.set_defaults(func=cmd_ingest_kilter)

    s = sub.add_parser("layouts", help="list layouts in a Kilter app database")
    s.add_argument("--source", help="path to the app's db.sqlite3")
    s.set_defaults(func=cmd_layouts)

    s = sub.add_parser("export-hold-types", help="write a hold-type curation CSV")
    s.add_argument("--db", help="input database")
    s.add_argument("--out", help="output CSV")
    s.add_argument("--layout-id", type=int)
    s.add_argument("--only-unknown", action="store_true",
                   help="only export holds still typed 'unknown'")
    s.set_defaults(func=cmd_export_hold_types)

    s = sub.add_parser("calibrate", help="fit beta score -> Kilter difficulty")
    s.add_argument("--db", help="input database")
    s.add_argument("--out", help="output calibration JSON")
    s.add_argument("--min-ascents", type=int, default=10)
    s.add_argument("--limit", type=int, default=1500, help="max (climb, angle) samples")
    s.add_argument("--max-expansions", type=int, default=4000,
                   help="per-climb search budget while fitting")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("stats", help="summarise the analysis database")
    s.add_argument("--db", help="input database")
    s.set_defaults(func=cmd_stats)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
