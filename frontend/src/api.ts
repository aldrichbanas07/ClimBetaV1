import type { BetaResponse, ClimbDetail, ClimbSummary, HoldOut, LayoutOut } from "./types";

const BASE_URL = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!resp.ok) {
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      detail = body.detail ?? JSON.stringify(body);
    } catch {
      // response wasn't JSON; fall back to statusText
    }
    throw new ApiError(resp.status, detail);
  }
  return resp.json() as Promise<T>;
}

export function listLayouts(): Promise<LayoutOut[]> {
  return request<LayoutOut[]>("/layouts");
}

export function getLayoutHolds(layoutId: number): Promise<HoldOut[]> {
  return request<HoldOut[]>(`/layouts/${layoutId}/holds`);
}

export function listClimbs(layoutId?: number): Promise<ClimbSummary[]> {
  const qs = layoutId != null ? `?layout_id=${layoutId}` : "";
  return request<ClimbSummary[]>(`/climbs${qs}`);
}

export function getClimb(climbId: string): Promise<ClimbDetail> {
  return request<ClimbDetail>(`/climbs/${encodeURIComponent(climbId)}`);
}

export interface GenerateBetaParams {
  climbId: string;
  angle: number;
  bodyHeight?: number;
  strategy?: "astar" | "greedy";
}

export function generateBeta(params: GenerateBetaParams): Promise<BetaResponse> {
  const body: Record<string, unknown> = {
    climb_id: params.climbId,
    angle: params.angle,
    strategy: params.strategy ?? "astar",
  };
  if (params.bodyHeight != null) {
    body.body = { height: params.bodyHeight };
  }
  return request<BetaResponse>("/generate-beta", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function healthCheck(): Promise<Record<string, unknown>> {
  return request("/health");
}
