import { Loader2 } from 'lucide-react';
import { fmt, pct } from '../../../forge/format';
import { useForgeStore } from '../../../forge/store';
import { useTimeMachine } from '../../../forge/timeMachine';
import { resolveEpoch } from '../../../forge/timeline';
import type { LayerHealth } from '../../../forge/types';
import { Section, SeriesChart } from '../charts';
import { useTimelineData } from './hooks';

const LAYER_COLORS = ['#60a5fa', '#a78bfa', '#34d399', '#f472b6', '#fbbf24', '#22d3ee'];

/**
 * Training health at the playhead, from the timeline payload: every value is
 * either measured on the stored checkpoint (norms, activation statistics) or
 * logged during real training (gradient and update norms).
 */
export function HealthPanel() {
  const session = useForgeStore((s) => s.session);
  const select = useForgeStore((s) => s.select);
  const timeline = useTimelineData();
  const cursor = useTimeMachine((s) => s.cursor);
  const goTo = useTimeMachine((s) => s.goTo);
  if (!session) return null;
  if (!timeline) {
    return <div className="p-3 flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-faint)' }}><Loader2 size={12} className="animate-spin" />loading timeline…</div>;
  }
  const current = resolveEpoch(cursor, session.epoch);
  const ck = timeline.checkpoints.find((c) => c.epoch === current);
  const epochs = timeline.checkpoints.map((c) => c.epoch);
  const hidden = (ck?.layers ?? []).filter((l) => l.mean_abs_activation !== null);
  const lane = (title: string, pick: (l: LayerHealth) => number | null, layers: LayerHealth[], hint?: string) => (
    <Section title={title} hint={hint}>
      {layers.map((l, li) => {
        const values = timeline.checkpoints.map((c) => pick(c.layers[l.layer - 1]));
        return (
          <div key={l.layer} className="mb-1">
            <div className="flex justify-between text-[10px]">
              <span style={{ color: LAYER_COLORS[li % LAYER_COLORS.length] }}>{l.label}</span>
              <span className="font-mono" style={{ color: 'var(--text-primary)' }}>{fmt(pick(l), 4)}</span>
            </div>
            <SeriesChart epochs={epochs} values={values} current={current} color={LAYER_COLORS[li % LAYER_COLORS.length]}
              height={28} onPick={(e) => void goTo(e)} ariaLabel={`${title} of ${l.label} across checkpoints`} />
          </div>
        );
      })}
    </Section>
  );

  if (!ck) return <p className="p-3 text-[11px]" style={{ color: 'var(--text-faint)' }}>No stored checkpoint at epoch {current}.</p>;

  return (
    <div className="space-y-2.5">
      <Section title={`Layers at epoch ${current}`} hint="stored checkpoint">
        <div className="overflow-x-auto">
          <table className="w-full text-[10.5px] font-mono">
            <thead>
              <tr style={{ color: 'var(--text-faint)' }}>
                <th className="text-left font-normal">layer</th>
                <th className="text-right font-normal" title="‖W‖₂ (Frobenius)">‖W‖</th>
                <th className="text-right font-normal" title="Mean L2 norm of this layer's training gradient during the epoch">train ‖∇‖</th>
                <th className="text-right font-normal" title="‖Δθ‖ over the epoch / ‖θ‖">Δθ/θ</th>
                <th className="text-right font-normal" title="Neurons that output 0 for every sample">dead</th>
                <th className="text-right font-normal" title="Sigmoid/Tanh outputs within 1% of an asymptote">sat.</th>
              </tr>
            </thead>
            <tbody>
              {ck.layers.map((l) => (
                <tr key={l.layer} className="cursor-pointer hover:bg-[var(--bg-hover)]" onClick={() => void select({ kind: 'layer', layer: l.layer })}>
                  <td className="py-0.5" style={{ color: 'var(--text-muted)' }}>{l.label}</td>
                  <td className="text-right">{fmt(l.weight_norm, 3)}</td>
                  <td className="text-right" style={{ color: 'var(--tm-grad)' }}>{fmt(l.train_grad_norm, 3)}</td>
                  <td className="text-right">{l.update_ratio === null ? '—' : pct(l.update_ratio, 2)}</td>
                  <td className="text-right" style={{ color: l.dead_fraction ? 'var(--tm-neg)' : undefined }}>
                    {l.dead_fraction === null ? '—' : pct(l.dead_fraction, 0)}
                  </td>
                  <td className="text-right" style={{ color: (l.saturated_fraction ?? 0) > 0.5 ? 'var(--tm-whatif-text)' : undefined }}>
                    {l.saturated_fraction === null ? '—' : pct(l.saturated_fraction, 0)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {current === 0 && <p className="text-[10px] mt-1" style={{ color: 'var(--text-faint)' }}>Epoch 0 is the initialisation: no training step has run, so there is no gradient or update to report.</p>}
      </Section>

      {lane('Training gradient norm', (l) => l.train_grad_norm, ck.layers, 'mean over the epoch’s mini-batches')}
      {lane('Update size Δθ/θ', (l) => l.update_ratio, ck.layers, 'parameter change during the epoch, relative')}
      {hidden.length > 0 && lane('Dead neurons', (l) => l.dead_fraction, hidden, 'fraction of neurons silent on every sample')}
      {hidden.some((l) => l.saturated_fraction !== null) && lane('Saturation', (l) => l.saturated_fraction, hidden.filter((l) => l.saturated_fraction !== null), 'Sigmoid/Tanh outputs near ±1 (or 0/1)')}
      {hidden.length > 0 && lane('Mean |activation|', (l) => l.mean_abs_activation, hidden, 'over dataset × neurons')}

      <p className="text-[10px] leading-snug" style={{ color: 'var(--text-faint)' }}>
        Gradient and update norms were logged during the real training run (loss including regularisation, Adam steps).
        Norms and activation statistics are measured on the stored checkpoint over the whole dataset.
      </p>
    </div>
  );
}
