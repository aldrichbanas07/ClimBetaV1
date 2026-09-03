import { HOLD_STYLES } from "../holdStyle";
import { HoldMarker } from "./HoldMarker";

const ORDER: (keyof typeof HOLD_STYLES)[] = [
  "jug",
  "crimp",
  "edge",
  "pocket",
  "pinch",
  "sloper",
  "foot_chip",
  "unknown",
];

export function Legend() {
  return (
    <div className="legend">
      {ORDER.map((type) => {
        const style = HOLD_STYLES[type];
        return (
          <div className="legend-item" key={type}>
            <svg width={18} height={18} viewBox="-9 -9 18 18">
              <g transform="scale(1,-1)">
                <HoldMarker cx={0} cy={0} r={6.5} color={style.color} shape={style.shape} />
              </g>
            </svg>
            <span>{style.label}</span>
          </div>
        );
      })}
      <div className="legend-item">
        <svg width={18} height={18}>
          <circle cx={9} cy={9} r={5} fill="none" stroke="#3ddc6a" strokeWidth={2} />
        </svg>
        <span>Start</span>
      </div>
      <div className="legend-item">
        <svg width={18} height={18}>
          <circle cx={9} cy={9} r={5} fill="none" stroke="#c66be0" strokeWidth={2} />
        </svg>
        <span>Finish</span>
      </div>
    </div>
  );
}
