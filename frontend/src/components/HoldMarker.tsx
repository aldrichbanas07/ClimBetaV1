import type { HoldShape } from "../holdStyle";

interface Props {
  cx: number;
  cy: number;
  r: number;
  color: string;
  shape: HoldShape;
  faded?: boolean;
  outlined?: boolean;
}

/**
 * Renders one hold as an SVG shape. The shape encodes hold type redundantly
 * with color (see holdStyle.ts) so identity survives a colorblind viewer or
 * a low-quality projector -- never "color alone".
 */
export function HoldMarker({ cx, cy, r, color, shape, faded, outlined }: Props) {
  const opacity = faded ? 0.28 : 1;
  const fill = outlined ? "none" : color;
  const stroke = outlined ? color : "rgba(0,0,0,0.35)";
  const strokeWidth = outlined ? 2 : 1.5;
  const common = { fill, stroke, strokeWidth, opacity };

  switch (shape) {
    case "circle":
      return <circle cx={cx} cy={cy} r={r} {...common} />;
    case "ring":
      return (
        <>
          <circle cx={cx} cy={cy} r={r} {...common} />
          <circle cx={cx} cy={cy} r={r * 0.42} fill="var(--board-bg)" opacity={opacity} />
        </>
      );
    case "diamond": {
      const d = `M ${cx} ${cy - r} L ${cx + r} ${cy} L ${cx} ${cy + r} L ${cx - r} ${cy} Z`;
      return <path d={d} {...common} />;
    }
    case "bar": {
      const w = r * 2.1;
      const h = r * 0.9;
      return (
        <rect
          x={cx - w / 2}
          y={cy - h / 2}
          width={w}
          height={h}
          rx={h * 0.3}
          transform={`rotate(${(cx * 37 + cy * 13) % 180} ${cx} ${cy})`}
          {...common}
        />
      );
    }
    case "rounded-rect":
      return (
        <rect
          x={cx - r}
          y={cy - r * 0.75}
          width={r * 2}
          height={r * 1.5}
          rx={r * 0.35}
          {...common}
        />
      );
    case "blob": {
      // Slightly irregular ellipse to read as "rounded, no positive edge".
      return (
        <ellipse
          cx={cx}
          cy={cy}
          rx={r * 1.15}
          ry={r * 0.85}
          transform={`rotate(${(cx * 23) % 40 - 20} ${cx} ${cy})`}
          {...common}
        />
      );
    }
    case "triangle": {
      const d = `M ${cx} ${cy - r * 0.9} L ${cx + r * 0.9} ${cy + r * 0.7} L ${cx - r * 0.9} ${cy + r * 0.7} Z`;
      return <path d={d} {...common} />;
    }
    case "square-outline":
      return (
        <rect
          x={cx - r * 0.8}
          y={cy - r * 0.8}
          width={r * 1.6}
          height={r * 1.6}
          fill="none"
          stroke={color}
          strokeWidth={1.5}
          strokeDasharray="3 2"
          opacity={opacity}
        />
      );
    default:
      return <circle cx={cx} cy={cy} r={r} {...common} />;
  }
}
