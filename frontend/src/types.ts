/**
 * Mirrors the backend's versioned move-list schema
 * (backend/kilterbeta/domain/moves.py). Keep in lockstep with that file --
 * SCHEMA_VERSION there is what tells you whether this file needs updating.
 *
 * Phase 2 will add real data to `pose` and `body.source`; phase 3 will start
 * populating `extensions`. Both are typed loosely on purpose so this file
 * does not need to change shape when that happens.
 */

export const EXPECTED_SCHEMA_VERSION = "1.0.0";

export type HoldType =
  | "jug"
  | "edge"
  | "crimp"
  | "sloper"
  | "pinch"
  | "pocket"
  | "foot_chip"
  | "unknown";

export type HoldRole = "start" | "hand" | "finish" | "foot" | "any";

export type Limb = "LH" | "RH" | "LF" | "RF";

export type MoveKind =
  | "start"
  | "static"
  | "long"
  | "dynamic"
  | "match"
  | "bump"
  | "foot_swap"
  | "finish";

export interface HoldOut {
  hold_id: number;
  x: number;
  y: number;
  hold_type: HoldType;
  role: HoldRole;
  size: number;
  placement_id: number | null;
  name: string | null;
}

export interface Contact {
  limb: Limb;
  hold_id: number;
  x: number;
  y: number;
  hold_type: HoldType;
}

export interface Point2D {
  x: number;
  y: number;
}

export interface BodyEstimate {
  hip: Point2D;
  shoulder_left: Point2D;
  shoulder_right: Point2D;
  source: string;
}

export interface DifficultyBreakdown {
  reach: number;
  target_hold: number;
  support_holds: number;
  body_tension: number;
  balance: number;
  penalties: number;
  total: number;
  notes: string[];
}

export interface BetaMove {
  index: number;
  limb: Limb;
  hold_id: number;
  x: number;
  y: number;
  hold_type: HoldType;
  kind: MoveKind;
  from_hold_id: number | null;
  reach_distance: number;
  reach_utilisation: number;
  difficulty: number;
  difficulty_breakdown: DifficultyBreakdown;
  contacts: Contact[];
  body: BodyEstimate;
  // Reserved for later phases.
  pose: Record<string, unknown> | null;
  extensions: Record<string, unknown>;
}

export interface GradeEstimate {
  difficulty_score: number;
  kilter_difficulty: number | null;
  boulder_grade: string | null;
  crux_move_index: number | null;
  calibrated: boolean;
}

export interface BetaResponse {
  schema_version: string;
  climb_id: string | null;
  climb_name: string | null;
  angle: number;
  moves: BetaMove[];
  grade: GradeEstimate;
  holds: HoldOut[];
  body_model: Record<string, number>;
  generator: {
    strategy: string;
    nodes_expanded: number;
    elapsed_ms: number;
    truncated: boolean;
    reached_finish: boolean;
    config: Record<string, unknown>;
    difficulty_model: Record<string, unknown>;
    calibration: Record<string, unknown>;
  };
  warnings: string[];
}

export interface LayoutOut {
  id: number;
  name: string;
  product_name: string | null;
  min_x: number;
  max_x: number;
  min_y: number;
  max_y: number;
  source: string;
}

export interface ClimbSummary {
  climb_id: string;
  layout_id: number;
  name: string;
  setter: string | null;
  description: string | null;
  setter_angle: number | null;
  hold_count: number;
  source: string;
  graded_angles: number[];
}

export interface ClimbDetail {
  climb: ClimbSummary;
  layout: LayoutOut | null;
  holds: HoldOut[];
  stats: Record<string, Record<string, number | null>>;
}
