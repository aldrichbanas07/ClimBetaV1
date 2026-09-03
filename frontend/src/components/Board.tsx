import { useMemo, useState } from "react";
import type { BetaMove, HoldOut } from "../types";
import { HOLD_STYLES, PATH_ACCENT, PATH_ACCENT_DIM, isFootLimb } from "../holdStyle";
import { HoldMarker } from "./HoldMarker";

interface Props {
  /** Every hold on the board, for the faint unused-hold underlay. */
  allHolds: HoldOut[];
  /** The current climb's holds (coloured, not faded). */
  climbHolds: HoldOut[];
  moves: BetaMove[];
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
  selectedMoveIndex: number | null;
  onSelectMove: (index: number | null) => void;
}

const PADDING = 14;
const HOLD_RADIUS = 6.5;

/**
 * 2D SVG render of the board: every hold as an underlay, the climb's own
 * holds highlighted by type, and the generated beta drawn as a numbered path
 * with limb badges.
 *
 * Coordinate note: the domain's y axis increases upward (see
 * backend/kilterbeta/domain/holds.py), but SVG y increases downward. Rather
 * than fight that in every marker's math, the whole board is wrapped in one
 * `scale(1, -1)` group with a compensating translate -- every child then
 * draws in "y increases up" coordinates directly.
 */
export function Board({ allHolds, climbHolds, moves, bounds, selectedMoveIndex, onSelectMove }: Props) {
  const [hoveredHold, setHoveredHold] = useState<number | null>(null);

  const width = bounds.maxX - bounds.minX + PADDING * 2;
  const height = bounds.maxY - bounds.minY + PADDING * 2;
  const viewBox = `${bounds.minX - PADDING} ${-(bounds.maxY + PADDING)} ${width} ${height}`;

  const climbHoldIds = useMemo(() => new Set(climbHolds.map((h) => h.hold_id)), [climbHolds]);
  const holdById = useMemo(() => {
    const map = new Map<number, HoldOut>();
    for (const h of allHolds) map.set(h.hold_id, h);
    for (const h of climbHolds) map.set(h.hold_id, h);
    return map;
  }, [allHolds, climbHolds]);

  const pathPoints = moves.map((m) => `${m.x},${m.y}`).join(" ");
  const hovered = hoveredHold != null ? holdById.get(hoveredHold) : undefined;

  return (
    <div className="board-wrap">
      <svg
        className="board-svg"
        viewBox={viewBox}
        preserveAspectRatio="xMidYMid meet"
        role="img"
        aria-label="Kilter Board with generated beta path"
      >
        <g transform="scale(1,-1)">
          {/* Board background */}
          <rect
            x={bounds.minX - PADDING}
            y={bounds.minY - PADDING}
            width={width}
            height={height}
            fill="var(--board-bg)"
          />

          {/* Faint underlay: every hold not part of this climb. */}
          {allHolds
            .filter((h) => !climbHoldIds.has(h.hold_id))
            .map((h) => {
              const style = HOLD_STYLES[h.hold_type];
              return (
                <HoldMarker
                  key={`bg-${h.hold_id}`}
                  cx={h.x}
                  cy={h.y}
                  r={HOLD_RADIUS * 0.75}
                  color={style.color}
                  shape={style.shape}
                  faded
                />
              );
            })}

          {/* This climb's holds, full colour. */}
          {climbHolds.map((h) => {
            const style = HOLD_STYLES[h.hold_type];
            const isHovered = hoveredHold === h.hold_id;
            return (
              <g
                key={h.hold_id}
                onMouseEnter={() => setHoveredHold(h.hold_id)}
                onMouseLeave={() => setHoveredHold(null)}
                style={{ cursor: "pointer" }}
              >
                {isHovered && (
                  <circle cx={h.x} cy={h.y} r={HOLD_RADIUS * 1.7} fill="none" stroke="white" strokeOpacity={0.4} strokeWidth={1.5} />
                )}
                <HoldMarker
                  cx={h.x}
                  cy={h.y}
                  r={h.role === "foot" ? HOLD_RADIUS * 0.85 : HOLD_RADIUS}
                  color={style.color}
                  shape={style.shape}
                  outlined={h.role === "foot"}
                />
                {(h.role === "start" || h.role === "finish") && (
                  <circle
                    cx={h.x}
                    cy={h.y}
                    r={HOLD_RADIUS + 3}
                    fill="none"
                    stroke={h.role === "start" ? "#3ddc6a" : "#c66be0"}
                    strokeWidth={2}
                  />
                )}
              </g>
            );
          })}

          {/* Beta path: connecting line, then numbered move badges. */}
          {moves.length > 1 && (
            <polyline
              points={pathPoints}
              fill="none"
              stroke={PATH_ACCENT}
              strokeWidth={1.75}
              strokeDasharray="1 5"
              strokeLinecap="round"
              opacity={0.85}
            />
          )}

          {moves.map((m, i) => {
            const isSelected = selectedMoveIndex === i;
            const badgeColor = isFootLimb(m.limb) ? PATH_ACCENT_DIM : PATH_ACCENT;
            const offsetY = -(HOLD_RADIUS + 11);
            return (
              <g
                key={`move-${i}`}
                transform={`translate(${m.x} ${m.y + offsetY})`}
                onClick={() => onSelectMove(isSelected ? null : i)}
                style={{ cursor: "pointer" }}
              >
                {isFootLimb(m.limb) ? (
                  <rect x={-9} y={-9} width={18} height={18} rx={3} fill={badgeColor} stroke={isSelected ? "white" : "rgba(0,0,0,0.4)"} strokeWidth={isSelected ? 2 : 1} transform="rotate(45)" />
                ) : (
                  <circle r={10} fill={badgeColor} stroke={isSelected ? "white" : "rgba(0,0,0,0.4)"} strokeWidth={isSelected ? 2 : 1} />
                )}
                <text
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize={10}
                  fontWeight={700}
                  fill="#1a1a19"
                  transform="scale(1,-1)"
                >
                  {i + 1}
                </text>
              </g>
            );
          })}
        </g>
      </svg>

      {hovered && (
        <div className="hold-tooltip">
          <strong>{HOLD_STYLES[hovered.hold_type].label}</strong>
          <span className="muted"> &middot; {hovered.role}</span>
        </div>
      )}
    </div>
  );
}
