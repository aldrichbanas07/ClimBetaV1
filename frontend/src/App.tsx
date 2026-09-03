import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";
import { ApiError, generateBeta, getClimb, getLayoutHolds, listClimbs } from "./api";
import { Board } from "./components/Board";
import { Legend } from "./components/Legend";
import { MoveList } from "./components/MoveList";
import type { BetaResponse, ClimbSummary, HoldOut } from "./types";
import { EXPECTED_SCHEMA_VERSION } from "./types";

const DEFAULT_ANGLE = 40;
const ANGLE_DEBOUNCE_MS = 200;

export default function App() {
  const [climbs, setClimbs] = useState<ClimbSummary[]>([]);
  const [selectedClimbId, setSelectedClimbId] = useState<string | null>(null);
  const [layoutHolds, setLayoutHolds] = useState<HoldOut[]>([]);
  const [climbHolds, setClimbHolds] = useState<HoldOut[]>([]);
  const [angle, setAngle] = useState(DEFAULT_ANGLE);
  const [pendingAngle, setPendingAngle] = useState(DEFAULT_ANGLE);
  const [beta, setBeta] = useState<BetaResponse | null>(null);
  const [selectedMoveIndex, setSelectedMoveIndex] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const debounceRef = useRef<number | undefined>(undefined);

  // Initial climb list.
  useEffect(() => {
    listClimbs()
      .then((list) => {
        setClimbs(list);
        if (list.length > 0) setSelectedClimbId(list[0].climb_id);
      })
      .catch((e) => setError(describeError(e)));
  }, []);

  // Climb detail (holds + layout) when the selection changes.
  useEffect(() => {
    if (!selectedClimbId) return;
    let cancelled = false;
    getClimb(selectedClimbId)
      .then((detail) => {
        if (cancelled) return;
        setClimbHolds(detail.holds);
        if (detail.climb.setter_angle != null) {
          setAngle(detail.climb.setter_angle);
          setPendingAngle(detail.climb.setter_angle);
        }
        if (detail.layout) {
          return getLayoutHolds(detail.layout.id).then((holds) => {
            if (!cancelled) setLayoutHolds(holds);
          });
        }
      })
      .catch((e) => setError(describeError(e)));
    return () => {
      cancelled = true;
    };
  }, [selectedClimbId]);

  // Debounce the angle slider so dragging doesn't spam the API.
  useEffect(() => {
    window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(() => setAngle(pendingAngle), ANGLE_DEBOUNCE_MS);
    return () => window.clearTimeout(debounceRef.current);
  }, [pendingAngle]);

  // Generate the beta whenever the climb or (debounced) angle changes.
  useEffect(() => {
    if (!selectedClimbId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    generateBeta({ climbId: selectedClimbId, angle })
      .then((result) => {
        if (cancelled) return;
        if (result.schema_version !== EXPECTED_SCHEMA_VERSION) {
          console.warn(
            `Beta schema version mismatch: frontend expects ${EXPECTED_SCHEMA_VERSION}, server returned ${result.schema_version}`,
          );
        }
        setBeta(result);
        setSelectedMoveIndex(null);
      })
      .catch((e) => setError(describeError(e)))
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [selectedClimbId, angle]);

  const bounds = useMemo(() => {
    const holds = layoutHolds.length > 0 ? layoutHolds : climbHolds;
    if (holds.length === 0) return { minX: 0, maxX: 100, minY: 0, maxY: 100 };
    return {
      minX: Math.min(...holds.map((h) => h.x)),
      maxX: Math.max(...holds.map((h) => h.x)),
      minY: Math.min(...holds.map((h) => h.y)),
      maxY: Math.max(...holds.map((h) => h.y)),
    };
  }, [layoutHolds, climbHolds]);

  const selectedClimb = climbs.find((c) => c.climb_id === selectedClimbId) ?? null;

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>Kilter Beta AI</h1>
        <span className="tagline">Phase 1 &mdash; heuristic beta generation</span>
      </header>

      <div className="app-body">
        <aside className="sidebar">
          <section className="control-group">
            <label htmlFor="climb-select">Climb</label>
            <select
              id="climb-select"
              value={selectedClimbId ?? ""}
              onChange={(e) => setSelectedClimbId(e.target.value)}
            >
              {climbs.map((c) => (
                <option key={c.climb_id} value={c.climb_id}>
                  {c.name} ({c.hold_count} holds)
                </option>
              ))}
            </select>
            {selectedClimb?.description && <p className="climb-description">{selectedClimb.description}</p>}
          </section>

          <section className="control-group">
            <label htmlFor="angle-slider">
              Wall angle: <strong>{pendingAngle}&deg;</strong>
            </label>
            <input
              id="angle-slider"
              type="range"
              min={0}
              max={70}
              step={5}
              value={pendingAngle}
              onChange={(e) => setPendingAngle(Number(e.target.value))}
            />
            <div className="angle-scale">
              <span>0&deg; slab</span>
              <span>70&deg; steep</span>
            </div>
          </section>

          {beta && (
            <section className="control-group grade-panel">
              <div className="grade-headline">
                <span className="grade-name">{beta.grade.boulder_grade ?? "—"}</span>
                {!beta.grade.calibrated && <span className="uncalibrated-flag">heuristic estimate</span>}
              </div>
              <div className="grade-meta">
                <span>Kilter difficulty: {beta.grade.kilter_difficulty?.toFixed(1) ?? "—"}</span>
                <span>Moves: {beta.moves.length}</span>
                <span>
                  Solved in {beta.generator.nodes_expanded} nodes / {beta.generator.elapsed_ms.toFixed(0)} ms
                </span>
              </div>
              {beta.generator.truncated && (
                <p className="warning-banner">
                  Search did not confirm a complete solution &mdash; showing the best sequence found.
                </p>
              )}
              {beta.warnings.map((w, i) => (
                <p className="warning-banner" key={i}>
                  {w}
                </p>
              ))}
            </section>
          )}

          <section className="control-group">
            <Legend />
          </section>
        </aside>

        <main className="board-panel">
          {error && <div className="error-banner">{error}</div>}
          {loading && <div className="loading-banner">Generating beta…</div>}
          <Board
            allHolds={layoutHolds}
            climbHolds={climbHolds}
            moves={beta?.moves ?? []}
            bounds={bounds}
            selectedMoveIndex={selectedMoveIndex}
            onSelectMove={setSelectedMoveIndex}
          />
        </main>

        <aside className="moves-panel">
          <h2>Beta sequence</h2>
          {beta ? (
            <MoveList
              moves={beta.moves}
              cruxIndex={beta.grade.crux_move_index}
              selectedIndex={selectedMoveIndex}
              onSelect={setSelectedMoveIndex}
            />
          ) : (
            <p className="muted">Select a climb to generate a beta.</p>
          )}
        </aside>
      </div>
    </div>
  );
}

function describeError(e: unknown): string {
  if (e instanceof ApiError) {
    if (e.status === 503) {
      return `${e.message} (run the ETL: python -m kilterbeta.etl.cli init-sample)`;
    }
    return `${e.message}`;
  }
  if (e instanceof Error) return e.message;
  return "Unknown error";
}
