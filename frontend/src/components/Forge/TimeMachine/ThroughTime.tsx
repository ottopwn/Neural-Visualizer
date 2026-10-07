import { Loader2, Spline } from 'lucide-react';
import { describeSeries, describeSilence } from '../../../forge/explainTime';
import { fmt, fmtSigned } from '../../../forge/format';
import { refKey } from '../../../forge/interventions';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { resolveEpoch } from '../../../forge/timeline';
import type { SeriesGroup } from '../../../forge/types';
import { Section, SeriesChart } from '../charts';
import { useComponentHistory } from './hooks';

const GROUP_COLOR: Record<SeriesGroup, string> = {
  parameter: '#a78bfa',
  probe: '#fde047',
  dataset: '#34d399',
  gradient: '#fbbf24',
};
const GROUP_TITLE: Record<SeriesGroup, string> = {
  parameter: 'Parameters',
  probe: 'On the probe input',
  dataset: 'Across the dataset',
  gradient: 'Gradients',
};
/** Series shown in Learn mode / in the compact Microscope view. */
const LEARN_KEYS = new Set(['bias', 'w_in_norm', 'a_probe', 'act_mean', 'active_frac', 'weight', 'contribution', 'w_norm', 'act_mean_abs', 'dead_frac']);
const COMPACT_KEYS = ['bias', 'a_probe', 'act_mean', 'weight', 'contribution', 'w_norm', 'dead_frac'];

/**
 * The selected neuron / layer / connection across every stored checkpoint.
 * Values come from the component-history endpoint (stored weights only, no
 * what-if overlay); the vertical line is the Time Machine's playhead.
 */
export function ThroughTime({ compact = false }: { compact?: boolean }) {
  const session = useForgeStore((s) => s.session);
  const selection = useForgeStore((s) => s.selection);
  const mode = useForgeStore((s) => s.mode);
  const nIv = useForgeStore((s) => s.interventions.length);
  const history = useComponentHistory();
  const loading = useTimeMachine((s) => s.historyLoading);
  const cursor = useTimeMachine((s) => s.cursor);
  const compareMode = useTimeMachine((s) => s.compareMode);
  const compareA = useTimeMachine((s) => s.compareA);
  const compareB = useTimeMachine((s) => s.compareB);
  const goTo = useTimeMachine((s) => s.goTo);
  if (!session) return null;

  if (!selection) {
    return (
      <div className="text-[12px] leading-relaxed p-3" style={{ color: 'var(--text-muted)' }}>
        <Spline size={16} className="mb-1.5" style={{ color: 'var(--tm-hist)' }} />
        Select a neuron, layer header or connection in the network to follow it through training.
        The same component stays selected while you move through epochs.
      </div>
    );
  }
  if (!history || refKey(history.ref) !== refKey(selection)) {
    return <div className="p-3 flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-faint)' }}><Loader2 size={12} className="animate-spin" />loading history…</div>;
  }

  const live = session.epoch;
  const current = resolveEpoch(cursor, live);
  const lab = mode === 'lab';
  const shown = compact
    ? history.series.filter((s) => COMPACT_KEYS.includes(s.key)).slice(0, 3)
    : history.series.filter((s) => lab || LEARN_KEYS.has(s.key));
  const groups = (['parameter', 'probe', 'dataset', 'gradient'] as SeriesGroup[])
    .map((g) => ({ g, series: shown.filter((s) => s.group === g) }))
    .filter((x) => x.series.length);
  const idx = history.epochs.indexOf(current);
  const markA = compareMode ? compareA : null;
  const markB = compareMode ? compareB ?? live : null;

  const rows = (
    <div className="space-y-2" style={{ opacity: loading ? 0.6 : 1 }}>
      {groups.map(({ g, series }) => (
        <div key={g}>
          {!compact && <div className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--text-faint)' }}>{GROUP_TITLE[g]}</div>}
          {series.map((s) => {
            const v = idx >= 0 ? s.values[idx] : null;
            const first = s.values.find((x) => x !== null) ?? null;
            return (
              <div key={s.key} className="mb-1">
                <div className="flex items-baseline justify-between text-[10.5px]">
                  <span style={{ color: 'var(--text-muted)' }}>{s.label}</span>
                  <span className="font-mono" style={{ color: 'var(--text-primary)' }}>
                    {fmt(v, 4)}
                    {lab && v !== null && first !== null && idx > 0 && (
                      <span className="ml-1.5" style={{ color: 'var(--text-faint)' }}>({fmtSigned(v - first, 3)} since start)</span>
                    )}
                  </span>
                </div>
                <SeriesChart epochs={history.epochs} values={s.values} current={current} markA={markA} markB={markB}
                  color={GROUP_COLOR[s.group]} height={compact ? 30 : 38} onPick={(e) => void goTo(e)}
                  ariaLabel={`${history.name} ${s.label} across stored checkpoints`} />
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );

  if (compact) {
    return (
      <Section title="Through training" hint={`${history.epochs.length} checkpoints · click to travel`}>
        {rows}
      </Section>
    );
  }

  const narrative = mode === 'learn'
    ? [describeSilence(history, current, session.structure.layers[history.ref.layer]?.activation ?? null), ...['act_mean', 'bias', 'weight', 'w_norm'].map((k) => describeSeries(history, k, current))]
      .filter((t): t is string => !!t)
    : [];

  return (
    <div className="space-y-2.5">
      <div className="flex items-baseline gap-2">
        <span className="font-mono font-semibold text-sm" style={{ color: 'var(--text-primary)' }}>{history.name}</span>
        <span className="text-[10px]" style={{ color: 'var(--text-faint)' }}>{history.epochs.length} stored checkpoints · stored weights only</span>
      </div>
      {nIv > 0 && (
        <p className="text-[10px] leading-snug" style={{ color: 'var(--tm-whatif-text)' }}>
          These curves show the stored checkpoints. Your {nIv} what-if edit{nIv > 1 ? 's are' : ' is'} applied only in the Microscope view.
        </p>
      )}
      {narrative.length > 0 && (
        <Section title="What changed">
          <ul className="space-y-1 text-[12px] leading-relaxed" style={{ color: 'var(--text-primary)' }}>
            {narrative.map((t, i) => <li key={i}>{t}</li>)}
          </ul>
        </Section>
      )}
      {rows}
      {history.notes.map((n, i) => <p key={i} className="text-[10px]" style={{ color: 'var(--text-faint)' }}>• {n}</p>)}
      <p className="text-[10px]" style={{ color: 'var(--text-faint)' }}>
        Dots are stored checkpoints; segments between them are straight lines, not intermediate states. Click a chart to jump there.
      </p>
    </div>
  );
}
