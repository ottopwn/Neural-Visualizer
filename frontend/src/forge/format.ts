// Number formatting and colour scales shared by Forge visualisations.

export function fmt(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  if (!Number.isFinite(v)) return v > 0 ? '∞' : '-∞';
  const a = Math.abs(v);
  if (a !== 0 && (a < 10 ** -digits || a >= 1e5)) return v.toExponential(Math.max(1, digits - 1));
  return v.toFixed(digits);
}

export function fmtSigned(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const s = fmt(v, digits);
  return v > 0 ? `+${s}` : s;
}

export function pct(v: number, digits = 1): string {
  return `${(v * 100).toFixed(digits)}%`;
}

export type RGB = [number, number, number];

// Colour semantics used everywhere in Forge:
//   positive value  -> teal/green  (matches positive weights in the graph)
//   negative value  -> red
//   class 0 / class 1 -> red / blue (matches the dataset preview)
export const POS: RGB = [16, 185, 129];
export const NEG: RGB = [239, 68, 68];
export const CLASS0: RGB = [239, 68, 68];
export const CLASS1: RGB = [59, 130, 246];
export const NEUTRAL: RGB = [17, 24, 39];

function mix(a: RGB, b: RGB, t: number): RGB {
  const c = Math.max(0, Math.min(1, t));
  return [0, 1, 2].map((i) => Math.round(a[i] + (b[i] - a[i]) * c)) as RGB;
}

export function rgb(c: RGB, alpha = 1): string {
  return alpha >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

/** Diverging scale: -maxAbs -> red, 0 -> neutral, +maxAbs -> green. */
export function diverging(v: number, maxAbs: number, neutral: RGB = NEUTRAL): RGB {
  if (!(maxAbs > 0)) return neutral;
  const t = Math.min(1, Math.abs(v) / maxAbs);
  return mix(neutral, v >= 0 ? POS : NEG, Math.sqrt(t));
}

/** Sequential scale from neutral to accent for values in [lo, hi]. */
export function sequential(v: number, lo: number, hi: number, accent: RGB = [96, 165, 250], neutral: RGB = NEUTRAL): RGB {
  const t = hi - lo > 1e-12 ? (v - lo) / (hi - lo) : 0;
  return mix(neutral, accent, t);
}

/** Probability of class 1 -> red (class 0) .. neutral .. blue (class 1). */
export function classColor(p1: number, neutral: RGB = NEUTRAL): RGB {
  return p1 >= 0.5 ? mix(neutral, CLASS1, (p1 - 0.5) * 2) : mix(neutral, CLASS0, (0.5 - p1) * 2);
}

export function maxAbs(values: number[]): number {
  let m = 0;
  for (const v of values) if (Math.abs(v) > m) m = Math.abs(v);
  return m;
}

/** Indices sorted by |value| descending. */
export function rankByMagnitude(values: number[]): number[] {
  return values.map((_, i) => i).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));
}

/** Equal-width histogram of `values` (empty-safe; a constant input gets one centred bin range). */
export function histogramOf(values: number[], bins = 24): { edges: number[]; counts: number[] } {
  if (!values.length) return { edges: [0, 1], counts: [0] };
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
  const width = (hi - lo) / bins;
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor((v - lo) / width))] += 1;
  return { edges: Array.from({ length: bins + 1 }, (_, i) => lo + i * width), counts };
}

export function mean(values: number[]): number {
  return values.length ? values.reduce((a, b) => a + b, 0) / values.length : 0;
}
