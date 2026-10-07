import { useMemo } from 'react';
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
