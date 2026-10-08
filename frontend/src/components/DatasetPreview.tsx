import { useEffect, useMemo, useRef, useState } from 'react';
import { getDatasetPreview } from '../api/client';
import { useNetworkStore } from '../store/networkStore';

type Data = { X: number[][]; y: number[] };

/**
 * The exact dataset a Build would train on: the backend generator is
 * deterministic (fixed seed), so this preview and the model's data agree.
 */
export function DatasetPreview() {
  const dataset = useNetworkStore((s) => s.trainingConfig.dataset);
  const noise = useNetworkStore((s) => s.trainingConfig.noise);
  const custom = useNetworkStore((s) => s.customDataset);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [fetched, setFetched] = useState<{ key: string; data: Data } | null>(null);
  const [failed, setFailed] = useState(false);
  const key = `${dataset}:${noise}`;

  useEffect(() => {
    if (custom) return;
    let alive = true;
    getDatasetPreview(useNetworkStore.getState().trainingConfig)
      .then((d) => { if (alive) { setFetched({ key, data: d }); setFailed(false); } })
      .catch(() => { if (alive) setFailed(true); });
    return () => { alive = false; };
  }, [dataset, noise, custom, key]);

  const data: Data | null = useMemo(() => (custom
    ? { X: custom.X.filter((r) => r.length >= 2), y: custom.y }
    : fetched?.key === key ? fetched.data : null), [custom, fetched, key]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !data || !data.X.length) return;
    const dpr = window.devicePixelRatio || 1;
    const W = canvas.clientWidth || 240;
    const H = canvas.clientHeight || 120;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const xs = data.X.map((p) => p[0]);
    const ys = data.X.map((p) => p[1]);
    const pad = 8;
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const sx = (v: number) => pad + ((v - x0) / (x1 - x0 || 1)) * (W - 2 * pad);
    const sy = (v: number) => H - pad - ((v - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    data.X.forEach((p, i) => {
      ctx.beginPath();
      ctx.arc(sx(p[0]), sy(p[1]), 2.4, 0, Math.PI * 2);
      ctx.fillStyle = data.y[i] === 1 ? 'rgba(59,130,246,0.9)' : 'rgba(239,68,68,0.9)';
      ctx.fill();
    });
  }, [data]);

  return (
    <figure className="rounded-md border p-2 m-0" style={{ borderColor: 'var(--border)', background: 'var(--bg-base)' }}>
      {failed && !custom ? (
        <p className="text-[11px] py-6 text-center" style={{ color: 'var(--text-faint)' }}>Preview needs the backend (port 8000).</p>
      ) : (
        <canvas ref={canvasRef} className="w-full block" style={{ height: 112 }} role="img"
          aria-label={custom ? 'Scatter plot of the uploaded dataset (first two features)' : `Scatter plot of the ${dataset} dataset`} />
      )}
      <figcaption className="flex items-center gap-3 mt-1 text-[11px]" style={{ color: 'var(--text-faint)' }}>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full inline-block" style={{ background: '#ef4444' }} />class 0</span>
        <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full inline-block" style={{ background: '#3b82f6' }} />class 1</span>
        <span className="ml-auto tnum">{data ? `${data.X.length} samples` : '…'}{custom ? ' · CSV' : ''}</span>
      </figcaption>
    </figure>
  );
}
