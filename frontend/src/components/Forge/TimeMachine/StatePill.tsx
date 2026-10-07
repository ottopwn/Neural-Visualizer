import { FlaskConical, History, Radio } from 'lucide-react';
import { useForgeStore } from '../../../forge/store';

/**
 * Says which of the three states the numbers on screen come from:
 * LIVE (latest weights), HISTORICAL (a stored, immutable checkpoint), and
 * whether a temporary WHAT-IF overlay is applied on top.
 */
export function StatePill({ epoch, historical, whatIf, size = 'sm' }: {
  epoch: number;
  historical: boolean;
  whatIf?: number;
  size?: 'sm' | 'md';
}) {
  const md = size === 'md';
  const base = `inline-flex items-center gap-1 rounded-md border font-semibold tracking-wide ${md ? 'text-[11px] px-2 py-1' : 'text-[10px] px-1.5 py-0.5'}`;
  return (
    <span className="inline-flex items-center gap-1" data-testid="state-pill">
      {historical ? (
        <span className={base} title="Stored checkpoint from the training history. Checkpoints are immutable."
          style={{ color: 'var(--tm-hist-text)', borderColor: 'var(--tm-hist)', background: 'color-mix(in srgb, var(--tm-hist) 12%, transparent)' }}>
          <History size={md ? 12 : 10} /> HISTORICAL · EPOCH {epoch}
        </span>
      ) : (
        <span className={base} title="The latest weights of the model."
          style={{ color: 'var(--tm-live-text)', borderColor: 'var(--tm-live)', background: 'color-mix(in srgb, var(--tm-live) 12%, transparent)' }}>
          <Radio size={md ? 12 : 10} /> LIVE · EPOCH {epoch}
        </span>
      )}
      {!!whatIf && (
        <span className={base} title="Temporary what-if overlay: applied to the forward pass only, never stored."
          style={{ color: 'var(--tm-whatif-text)', borderColor: 'var(--tm-whatif)', background: 'color-mix(in srgb, var(--tm-whatif) 12%, transparent)' }}>
          <FlaskConical size={md ? 12 : 10} /> WHAT-IF ×{whatIf}
        </span>
      )}
    </span>
  );
}

/** State pill for the Forge experiment the Microscope is currently showing. */
export function ForgeStatePill({ size }: { size?: 'sm' | 'md' }) {
  const session = useForgeStore((s) => s.session);
  const checkpointEpoch = useForgeStore((s) => s.checkpointEpoch);
  const n = useForgeStore((s) => s.interventions.length);
  if (!session) return null;
  return <StatePill epoch={checkpointEpoch ?? session.epoch} historical={checkpointEpoch !== null} whatIf={n} size={size} />;
}
