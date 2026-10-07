import { useEffect, useMemo, useRef, useState } from 'react';
import { classColor, diverging, sequential, type RGB } from '../../forge/format';
import { useForgeStore } from '../../forge/store';
import type { ResponseMap } from '../../forge/types';
import type { ScatterPoint } from './charts';

export function useDatasetPoints(): ScatterPoint[] | undefined {
  const session = useForgeStore((s) => s.session);
  return useMemo(() => {
    if (!session || session.structure.input_dim !== 2) return undefined;
    return session.dataset_X.map((p, i) => ({ x: p[0], y: p[1], cls: session.dataset_y[i] }));
  }, [session]);
}

/** Colour function appropriate for a response map's quantity. */
export function responseColor(map: ResponseMap, classIndex: number | null): (v: number) => RGB {
  if (classIndex !== null) return (v) => classColor(classIndex === 1 ? v : 1 - v);
  const [lo, hi] = map.value_range;
  if (lo < 0) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    return (v) => diverging(v, m);
  }
  return (v) => sequential(v, 0, hi);
}

/** Width of an element, tracked with a ResizeObserver. */
export function useElementWidth<T extends HTMLElement>(initial = 300): [React.RefObject<T | null>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([entry]) => setW(Math.max(120, Math.round(entry.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

/** Width and height of an element, tracked with a ResizeObserver. */
export function useElementSize<T extends HTMLElement>(initial = { w: 300, h: 300 }): [React.RefObject<T | null>, { w: number; h: number }] {
  const ref = useRef<T>(null);
  const [size, setSize] = useState(initial);
  useEffect(() => {
    const el = ref.current;
    if (!el || typeof ResizeObserver === 'undefined') return;
    const ro = new ResizeObserver(([entry]) => setSize({
      w: Math.max(120, Math.round(entry.contentRect.width)), h: Math.max(120, Math.round(entry.contentRect.height)),
    }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}
