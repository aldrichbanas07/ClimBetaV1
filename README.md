# Kilter Climbing Beta AI

Phase 1 of a climbing beta-analysis tool: given a Kilter Board climb (a set
of holds + a wall angle), generate a heuristic, physically-motivated hand/foot
sequence, and render it on an interactive board.

No ML here by design — this phase is a calibrated cost model plus a search
algorithm. See [ROADMAP](#roadmap) for what phases 2 and 3 add on top without
requiring this phase's schema to change.

## Quick start

**Backend**

```bash
python -m venv .venv
.venv/Scripts/activate            # or: source .venv/bin/activate
pip install -r requirements.txt

# Build the synthetic sample board + 6 demo climbs into SQLite:
PYTHONPATH=backend python -m kilterbeta.etl.cli init-sample

# Run the API (docs at http://localhost:8000/docs):
PYTHONPATH=backend python -m uvicorn kilterbeta.api.app:app --port 8000
```

**Frontend**

```bash
cd frontend
npm install
cp .env.example .env       # points VITE_API_BASE at localhost:8000
npm run dev                # http://localhost:5173
```

Pick a climb, drag the angle slider, watch the beta and grade update.

**Tests**

```bash
pytest tests/ -q            # 104 tests: domain, body model, difficulty
                             # model, search, generator, ETL, API
```

## How it works

```
backend/kilterbeta/
  domain/
    holds.py     Hold, HoldType, HoldRole, Limb — the shared vocabulary
    moves.py     BetaMove / BetaResponse — the VERSIONED wire schema (see below)
  beta/
    body.py          Anthropometry + a positional stance solver (hip/shoulder
                      estimate from wherever the limbs currently are)
    difficulty.py     Hold-type x angle cost model (Channel A) and a
                      angle-driven body-tension cost independent of hold type
                      (Channel B)
    search.py         A*/greedy search over HAND states (LH, RH, last-moved);
                      feet are resolved deterministically per hand pair by a
                      progress-aware sub-search, so the state space stays
                      small (dozens of nodes, not tens of thousands)
    calibration.py    Beta cost -> Kilter numeric difficulty -> V/Font grade.
                      Uncalibrated by default (flagged `calibrated: false`);
                      `etl calibrate` fits real coefficients once real
                      community grade data has been ingested
    generator.py      Ties search output to the versioned BetaResponse
  db/            SQLite schema + repository (read-only query layer)
  etl/
    sample_source.py   A synthetic, deterministic 12x12 board + 6 demo climbs
    kilter_source.py   Reads a real Kilter Board app db.sqlite3 (schema-drift
                        tolerant: works across app versions that add/rename
                        columns)
    hold_types.py       Hold-TYPE classification (crimp/jug/sloper/...) — see
                        "Why hold type needs a classification layer" below
    cli.py              init-sample | ingest-kilter | export-hold-types |
                        calibrate | stats | layouts
  api/app.py     FastAPI: /generate-beta, /climbs, /layouts, /schema/move, ...

frontend/src/
  types.ts, api.ts       Typed mirror of the backend schema + fetch client
  holdStyle.ts           Hold-type -> {color, shape} (see palette note below)
  components/
    Board.tsx            SVG board: hold underlay + coloured climb holds +
                          numbered beta path with limb badges
    MoveList.tsx          Per-move limb/kind/difficulty, expandable cost
                          breakdown, crux highlight
    Legend.tsx, HoldMarker.tsx
  App.tsx                Wiring: climb picker, debounced angle slider, fetch
                          orchestration
```

### Why hold type needs a classification layer

The Kilter Board app's own database has no notion of "crimp" vs "jug" vs
"sloper" — it stores hole geometry and how a hold is used *within one climb*
(start/hand/finish/foot), nothing about the hold's physical shape. Since hold
type is the main input to the difficulty model, `etl/hold_types.py` supplies
it via: a hand-curated CSV override (`etl export-hold-types` generates a
starter template) → a set-name heuristic (screw-on sets are footholds) → the
placement's default role → `unknown` (treated as an average hold, never an
error). Swapping in real Kilter data does not require this file to change —
only the CSV needs filling in for full accuracy.

### The search

Searching over all four limbs jointly is combinatorially expensive and mostly
wasted on foot shuffling. Instead:

- **A\*** runs over hand states only: `(left hand hold, right hand hold, last
  hand moved)`.
- For each hand state, feet are resolved by a small deterministic sub-search
  that balances *comfort* (a low, relaxed stance) against *whether it leaves
  an onward hand move available* — the latter is what stops the planner from
  parking a foot comfortably and then finding nothing reachable.
- A move's cost decomposes into reach strain, target-hold difficulty,
  load-weighted support-hold difficulty, an angle-driven body-tension term,
  a barn-door/balance term, and move-specific penalties (dynamic, match,
  bump, cross-through, high-step, hand-foot-share). All of it is returned in
  `difficulty_breakdown` so the UI can show *why* a move is hard.

On the 6 sample climbs this solves in single-digit milliseconds to ~200ms per
angle — fast enough for the angle slider to feel live.

### The versioned schema

`domain/moves.py`'s `BetaResponse`/`BetaMove` is the contract between the
generator, the API, and the frontend, and is explicitly built for phase 2 to
extend rather than replace:

- `BetaMove.pose` and `BetaMove.extensions` exist now, always `null`/`{}` in
  phase 1 — phase 2 (inverse kinematics) fills `pose`; phase 3 (video
  comparison) namespaces data into `extensions`.
- `BetaMove.contacts` carries the *complete* four-limb stance after each
  move (not just the limb that moved), because IK needs the whole stance to
  solve a pose.
- `BetaResponse.body_model` echoes the exact anthropometry the search used,
  so phase 2's IK solves against the same segment lengths the beta was
  planned with.

`GET /schema/move` serves the live JSON Schema plus the reserved-field list,
so a client can check compatibility instead of assuming it.

### Grade calibration

Phase 1 ships **uncalibrated** by default — the score→grade coefficients are
a non-negative least-squares fit against six hand-judged reference climbs
(`scripts/fit_default_calibration.py`), not real community data, and every
response says so via `grade.calibrated: false`. Once a real Kilter database
is ingested (`etl ingest-kilter`), `etl calibrate` refits the same model
against real per-angle `climb_stats` difficulty and persists it to
`data/calibration.json`, which the API picks up automatically.

### Frontend palette

Hold-type colors are the dataviz skill's validated dark-mode categorical
palette (7 of its 8 slots, kept in the order that passes its CVD-safety
checks). Because every hold type is visible simultaneously on the board (a
spatial scatter, not a small stacked set), the palette's own *all-pairs*
validation fails past 3 categories — expected and documented in the skill.
Each hold type therefore also gets a distinct marker *shape*
(circle/bar/diamond/ring/...), so identity never depends on color alone. The
8th slot (red) is reserved exclusively for the beta path, so the climbing
line never collides with a hold color.

## Using real Kilter data

```bash
# 1. Get a copy of the Kilter Board app's db.sqlite3 (e.g. via `boardlib`)
#    and place it at data/kilter/db.sqlite3, or point KILTER_APP_DB at it.

# 2. See what layouts it has:
python -m kilterbeta.etl.cli layouts

# 3. Ingest one:
python -m kilterbeta.etl.cli ingest-kilter --layout-id <id>

# 4. Curate hold types (the app DB has geometry, not physical hold type):
python -m kilterbeta.etl.cli export-hold-types --only-unknown
#    ... fill in the CSV's hold_type column ...
python -m kilterbeta.etl.cli ingest-kilter --layout-id <id>   # re-run to apply

# 5. Fit real grade calibration:
python -m kilterbeta.etl.cli calibrate --min-ascents 10
```

## Roadmap (context, not built yet)

- **Phase 2**: attach a full-body IK pose to each move — a skeletal model
  (hip, shoulder, 2-segment limbs) solved against `BetaMove.contacts` and
  `BetaResponse.body_model`, populating `BetaMove.pose`. Purely geometric.
- **Phase 3**: MediaPipe pose estimation from climber video, aligned via DTW
  against phase 2's reference pose sequence, to score technique and give
  tips.

Phase 1's schema and API response shape are held stable specifically so
these phases can attach data rather than requiring a rewrite.
