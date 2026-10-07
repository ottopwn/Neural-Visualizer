import { Loader2, Sparkles, Zap } from 'lucide-react';
import type { ReactNode } from 'react';
import { FORGE_MODELS, useExperiment } from '../../app/experiment';
import { useWorkspace } from '../../app/workspace';
import { useNetworkStore } from '../../store/networkStore';

/**
 * What a Forge instrument shows when there is no real model to read from:
 * either nothing is built yet, or the selected model type is illustrative.
 */
export function NeedsModel({ icon, title, body }: { icon: ReactNode; title: string; body: ReactNode }) {
  const modelType = useNetworkStore((s) => s.networkConfig.model_type);
  const setModel = useNetworkStore((s) => s.setNetworkConfig);
  const building = useExperiment((s) => s.buildStatus.type === 'loading');
  const build = useExperiment((s) => s.build);
  const openWelcome = useWorkspace((s) => s.setWelcomeOpen);
  const real = FORGE_MODELS.has(modelType);

  return (
    <div className="h-full flex items-center justify-center p-6">
      <div className="max-w-md text-center">
        <div className="mx-auto mb-4 w-11 h-11 rounded-lg flex items-center justify-center border"
          style={{ borderColor: 'var(--border-soft)', color: 'var(--accent)', background: 'var(--bg-card)' }} aria-hidden="true">
          {icon}
        </div>
        <h2 className="text-[17px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>{title}</h2>
        <div className="mt-2 text-[13px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>{body}</div>
        {!real && (
          <p className="mt-3 text-[12.5px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            The selected model type, <b>{modelType}</b>, is an illustrative diagram: there is no real model behind it to inspect.
          </p>
        )}
        <div className="mt-5 flex items-center justify-center gap-2">
          {real ? (
            <button type="button" className="btn-primary" disabled={building} onClick={() => void build()}>
              {building ? <Loader2 size={14} className="animate-spin" /> : <Zap size={14} />}Build a real network
            </button>
          ) : (
            <button type="button" className="btn-primary" onClick={() => setModel({ model_type: 'ANN' })}>Switch to ANN</button>
          )}
          <button type="button" className="btn-secondary" onClick={() => openWelcome(true)}><Sparkles size={14} />Guided demos</button>
        </div>
      </div>
    </div>
  );
}
