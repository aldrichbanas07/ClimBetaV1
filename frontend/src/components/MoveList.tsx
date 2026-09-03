import type { BetaMove } from "../types";
import { HOLD_STYLES, LIMB_LABELS, isFootLimb } from "../holdStyle";

interface Props {
  moves: BetaMove[];
  cruxIndex: number | null;
  selectedIndex: number | null;
  onSelect: (index: number | null) => void;
}

const KIND_LABELS: Record<string, string> = {
  start: "Start",
  static: "Static",
  long: "Long reach",
  dynamic: "Dynamic",
  match: "Match",
  bump: "Bump",
  foot_swap: "Foot swap",
  finish: "Finish",
};

export function MoveList({ moves, cruxIndex, selectedIndex, onSelect }: Props) {
  return (
    <ol className="move-list">
      {moves.map((m, i) => {
        const style = HOLD_STYLES[m.hold_type];
        const isSelected = selectedIndex === i;
        const isCrux = cruxIndex === i;
        return (
          <li
            key={i}
            className={`move-row ${isSelected ? "selected" : ""} ${isCrux ? "crux" : ""}`}
            onClick={() => onSelect(isSelected ? null : i)}
          >
            <span className="move-index">{i + 1}</span>
            <span className={`limb-tag ${isFootLimb(m.limb) ? "foot" : "hand"}`} title={LIMB_LABELS[m.limb]}>
              {m.limb}
            </span>
            <span className="hold-swatch" style={{ background: style.color }} title={style.label} />
            <span className="move-kind">{KIND_LABELS[m.kind] ?? m.kind}</span>
            <span className="move-spacer" />
            {isCrux && <span className="crux-badge">CRUX</span>}
            <span className="move-difficulty" title="Difficulty score for this move">
              {m.difficulty.toFixed(2)}
            </span>

            {isSelected && (
              <div className="move-detail" onClick={(e) => e.stopPropagation()}>
                <div className="move-detail-grid">
                  <span>Reach</span>
                  <span>
                    {m.reach_distance.toFixed(1)}" ({Math.round(m.reach_utilisation * 100)}% of span)
                  </span>
                  <span>Target hold</span>
                  <span>{m.difficulty_breakdown.target_hold.toFixed(2)}</span>
                  <span>Reach cost</span>
                  <span>{m.difficulty_breakdown.reach.toFixed(2)}</span>
                  <span>Support holds</span>
                  <span>{m.difficulty_breakdown.support_holds.toFixed(2)}</span>
                  <span>Body tension</span>
                  <span>{m.difficulty_breakdown.body_tension.toFixed(2)}</span>
                  <span>Balance</span>
                  <span>{m.difficulty_breakdown.balance.toFixed(2)}</span>
                  <span>Penalties</span>
                  <span>{m.difficulty_breakdown.penalties.toFixed(2)}</span>
                </div>
                {m.difficulty_breakdown.notes.length > 0 && (
                  <ul className="move-notes">
                    {m.difficulty_breakdown.notes.map((n, ni) => (
                      <li key={ni}>{n}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}
