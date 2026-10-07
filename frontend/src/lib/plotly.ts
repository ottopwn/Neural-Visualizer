// Plotly is large (~4.5 MB), so it is code-split and loaded on first use.
// Previously index.html pulled Plotly 2.27 from cdn.plot.ly into `window.Plotly`;
// the bundled npm package (already a dependency) keeps the app usable offline.

export interface PlotlyLike {
  react: (el: HTMLElement, data: unknown[], layout?: unknown, config?: unknown) => Promise<unknown>;
  purge: (el: HTMLElement) => void;
}

let pending: Promise<PlotlyLike> | null = null;

export function loadPlotly(): Promise<PlotlyLike> {
  pending ??= import('plotly.js-dist-min').then((m) => (m.default ?? m) as PlotlyLike);
  return pending;
}
