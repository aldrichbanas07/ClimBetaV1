import type { HoldType, Limb } from "./types";

/**
 * Hold-type -> {color, shape}.
 *
 * Colors are the first 7 slots of the validated dark-mode categorical
 * palette (kept in the palette's fixed, CVD-safety-bearing order; see
 * dataviz skill reference), leaving the 8th slot (red) reserved exclusively
 * for the beta path so the climbing line never collides with a hold color.
 *
 * With 7 categories rendered simultaneously as a spatial scatter (every hold
 * on the board at once, not a small stacked/adjacent set), the palette's own
 * validator fails the stricter "all-pairs" check past 3 slots -- expected,
 * and documented in dataviz/references/palette.md. Shape carries identity as
 * well, per "never color alone": each hold type also gets a distinct marker
 * silhouette, so confusable colors are never the only cue.
 *
 * `unknown` is deliberately NOT a categorical hue -- an unclassified hold is
 * "other", rendered as neutral gray outline per the palette's own guidance.
 */
export type HoldShape = "circle" | "bar" | "rounded-rect" | "ring" | "diamond" | "blob" | "triangle" | "square-outline";

export interface HoldVisual {
  color: string;
  shape: HoldShape;
  label: string;
}

export const HOLD_STYLES: Record<HoldType, HoldVisual> = {
  jug: { color: "#3987e5", shape: "circle", label: "Jug" },
  crimp: { color: "#d95926", shape: "bar", label: "Crimp" },
  edge: { color: "#199e70", shape: "rounded-rect", label: "Edge" },
  pocket: { color: "#c98500", shape: "ring", label: "Pocket" },
  pinch: { color: "#d55181", shape: "diamond", label: "Pinch" },
  sloper: { color: "#008300", shape: "blob", label: "Sloper" },
  foot_chip: { color: "#9085e9", shape: "triangle", label: "Foot chip" },
  unknown: { color: "#8a8a86", shape: "square-outline", label: "Unclassified" },
};

/** Reserved exclusively for the beta path -- never assigned to a hold type. */
export const PATH_ACCENT = "#e66767";
export const PATH_ACCENT_DIM = "#a85252"; // feet, on the same hue ramp as the accent

export const LIMB_LABELS: Record<Limb, string> = {
  LH: "Left hand",
  RH: "Right hand",
  LF: "Left foot",
  RF: "Right foot",
};

export function isFootLimb(limb: Limb): boolean {
  return limb === "LF" || limb === "RF";
}
