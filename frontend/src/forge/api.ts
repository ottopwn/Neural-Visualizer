import axios from 'axios';
import type { TransformerInfo, TransformerTrace } from './transformerTypes';
import type {
  ComponentHistory, ComponentRef, ComputationTrace, Comparison, EpochComparison, ExperimentRequest, ForgeGraph, Frame,
  HistoryRow, Inspection, Probe, SessionSummary, Timeline,
} from './types';

const BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
const http = axios.create({ baseURL: `${BASE}/api/forge`, timeout: 90000 });

export interface CreateSessionBody {
  neurons: number[];
  activations: string[];
  dataset: string;
  noise: number;
  custom_dataset?: { X: number[][]; y: number[] } | null;
  seed?: number;
}

export interface TrainBody {
  epochs: number;
  learning_rate: number;
  batch_size: number;
  reg_type: string;
  reg_rate: number;
}

export const createSession = (body: CreateSessionBody): Promise<SessionSummary> =>
  http.post('/sessions', body).then((r) => r.data);

export const trainSession = (id: string, body: TrainBody): Promise<{ summary: SessionSummary; new_rows: HistoryRow[] }> =>
  http.post(`/sessions/${id}/train`, body).then((r) => r.data);

export const fetchGraph = (id: string, req: ExperimentRequest): Promise<ForgeGraph> =>
  http.post(`/sessions/${id}/graph`, req).then((r) => r.data);

export const inspect = (id: string, ref: ComponentRef, req: ExperimentRequest): Promise<Inspection> =>
  http.post(`/sessions/${id}/inspect`, { ...req, ref }).then((r) => r.data);

export const fetchTrace = (id: string, req: ExperimentRequest & { learning_rate?: number | null }, signal?: AbortSignal): Promise<ComputationTrace> =>
  http.post(`/sessions/${id}/trace`, req, { signal }).then((r) => r.data);

export interface SessionLossLandscape {
  checkpoint_epoch: number;
  is_latest: boolean;
  alphas: number[];
  betas: number[];
  loss: number[][];
  center_loss: number;
  min_loss: number;
  max_loss: number;
  seed: number;
  note: string;
}

export const fetchLossLandscape = (id: string, checkpointEpoch: number | null, signal?: AbortSignal): Promise<SessionLossLandscape> =>
  http.post(`/sessions/${id}/loss-landscape`, { checkpoint_epoch: checkpointEpoch }, { signal }).then((r) => r.data);

export const compare = (id: string, req: ExperimentRequest): Promise<Comparison> =>
  http.post(`/sessions/${id}/compare`, req).then((r) => r.data);

// ── Training Time Machine (read-only views of stored checkpoints) ──────────
// Every call accepts an AbortSignal so a superseded request can be cancelled.

export const fetchTimeline = (id: string, signal?: AbortSignal): Promise<Timeline> =>
  http.get(`/sessions/${id}/timeline`, { signal }).then((r) => r.data);

export const fetchFrame = (id: string, checkpointEpoch: number | null, probe: Probe, signal?: AbortSignal): Promise<Frame> =>
  http.post(`/sessions/${id}/frame`, { checkpoint_epoch: checkpointEpoch, probe }, { signal }).then((r) => r.data);

export const fetchComponentHistory = (id: string, ref: ComponentRef, probe: Probe, signal?: AbortSignal): Promise<ComponentHistory> =>
  http.post(`/sessions/${id}/component-history`, { ref, probe }, { signal }).then((r) => r.data);

export const fetchEpochComparison = (
  id: string, body: { epoch_a: number; epoch_b: number | null; probe: Probe; ref: ComponentRef | null }, signal?: AbortSignal,
): Promise<EpochComparison> =>
  http.post(`/sessions/${id}/epoch-compare`, body, { signal }).then((r) => r.data);

// ── Transformer Lab (a tiny model trained locally on first use) ─────────────

export const fetchTransformerInfo = (signal?: AbortSignal): Promise<TransformerInfo> =>
  http.get('/transformer', { signal, timeout: 180000 }).then((r) => r.data);

export const fetchTransformerTrace = (
  body: { text: string; ablate_heads: [number, number][]; top_k?: number }, signal?: AbortSignal,
): Promise<TransformerTrace> =>
  http.post('/transformer/trace', body, { signal, timeout: 180000 }).then((r) => r.data);

/** True for errors caused by cancelling a superseded request. */
export const isCancel = (err: unknown): boolean => axios.isCancel(err);

/** Human-readable message from an API error (FastAPI puts it in `detail`). */
export function errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = err.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (!err.response) return 'Backend unreachable — start FastAPI on :8000';
    return `Request failed (${err.response.status})`;
  }
  return err instanceof Error ? err.message : String(err);
}
