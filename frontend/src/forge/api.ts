import axios from 'axios';
import type {
  ComponentRef, Comparison, ExperimentRequest, ForgeGraph, HistoryRow, Inspection, SessionSummary,
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

export const compare = (id: string, req: ExperimentRequest): Promise<Comparison> =>
  http.post(`/sessions/${id}/compare`, req).then((r) => r.data);

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
