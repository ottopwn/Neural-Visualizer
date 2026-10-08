// React glue for the Time Machine: visible components declare what they
// need, the store fetches it (cached, de-duplicated, stale-protected).

import { useEffect, useMemo } from 'react';
import { useTheme } from '../../../contexts/theme';
import { NEUTRAL, type RGB } from '../../../forge/format';
import { refKey } from '../../../forge/interventions';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { probeKey, sortedEpochs } from '../../../forge/timeline';

function useSessionKey(): string {
  const session = useForgeStore((s) => s.session);
  return session ? `${session.session_id}@${session.epoch}` : '';
}

/** Stored checkpoint epochs of the current session (sorted). */
export function useCheckpointEpochs(): number[] {
  const session = useForgeStore((s) => s.session);
  return useMemo(() => sortedEpochs(session?.checkpoints.map((c) => c.epoch) ?? []), [session]);
}

export function useTimelineData() {
  const key = useSessionKey();
  const loadTimeline = useTimeMachine((s) => s.loadTimeline);
  useEffect(() => { if (key) void loadTimeline(); }, [key, loadTimeline]);
  return useTimeMachine((s) => s.timeline);
}

/** Keep the frame of the playhead loaded (reloads when the probe changes). */
export function useFrame() {
  const key = useSessionKey();
  const probe = useForgeStore((s) => s.probe);
  const pk = probeKey(probe);
  const loadFrame = useTimeMachine((s) => s.loadFrame);
  useEffect(() => { if (key) void loadFrame(); }, [key, pk, loadFrame]);
  return useTimeMachine((s) => s.frame);
}

/** History of the selected component across all checkpoints. */
export function useComponentHistory() {
  const key = useSessionKey();
  const selection = useForgeStore((s) => s.selection);
  const probe = useForgeStore((s) => s.probe);
  const rk = refKey(selection);
  const pk = probeKey(probe);
  const loadHistory = useTimeMachine((s) => s.loadHistory);
  useEffect(() => { if (key) void loadHistory(); }, [key, rk, pk, loadHistory]);
  return useTimeMachine((s) => s.history);
}

/** A/B comparison, refreshed when epochs, probe or selected component change. */
export function useEpochComparison() {
  const key = useSessionKey();
  const selection = useForgeStore((s) => s.selection);
  const probe = useForgeStore((s) => s.probe);
  const compareMode = useTimeMachine((s) => s.compareMode);
  const compareA = useTimeMachine((s) => s.compareA);
  const compareB = useTimeMachine((s) => s.compareB);
  const loadComparison = useTimeMachine((s) => s.loadComparison);
  const rk = refKey(selection);
  const pk = probeKey(probe);
  useEffect(() => {
    if (key && compareMode) void loadComparison();
  }, [key, compareMode, compareA, compareB, rk, pk, loadComparison]);
  return useTimeMachine((s) => s.comparison);
}

/** Space = play/pause, ←/→ = step, Home/End = first/live (ignored while typing). */
export function useTransportKeys(enabled: boolean) {
  useEffect(() => {
    if (!enabled) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const tm = useTimeMachine.getState();
      if (e.key === ' ') { e.preventDefault(); tm.togglePlay(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); void tm.step(1); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); void tm.step(-1); }
      else if (e.key === 'Home') { e.preventDefault(); void tm.first(); }
      else if (e.key === 'End') { e.preventDefault(); void tm.goLive(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [enabled]);
}

/** Neutral colour of probability maps: dark instrument display, light on Paper. */
export function useNeutral(): RGB {
  const { theme } = useTheme();
  return theme === 'paper' ? [241, 245, 249] : NEUTRAL;
}
