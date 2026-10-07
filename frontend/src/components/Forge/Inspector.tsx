import { Loader2, Microscope, X } from 'lucide-react';
import { useForgeStore } from '../../forge/store';
import type { Inspection, ModelStructure } from '../../forge/types';
import { ConnectionPanel } from './ConnectionPanel';
import { LayerPanel } from './LayerPanel';
import { NeuronPanel } from './NeuronPanel';
import { ProvenanceBadge } from './shared';
import { ForgeStatePill } from './TimeMachine/StatePill';
import { ThroughTime } from './TimeMachine/ThroughTime';

function title(i: Inspection): { name: string; sub: string } {
  if (i.kind === 'neuron') {
    return {
      name: i.name,
      sub: `${i.role} neuron · ${i.layer_label}${i.activation_fn ? ` · ${i.activation_fn}` : ''}`,
    };
  }
  if (i.kind === 'layer') {
    return { name: i.label, sub: `${i.role} layer · ${i.size} neurons${i.activation_fn ? ` · ${i.activation_fn}` : ''}` };
  }
  return { name: `${i.source_name} → ${i.target_name}`, sub: `connection · W${i.layer}[${i.target}, ${i.source}]` };
}

export function Inspector({ structure }: { structure: ModelStructure }) {
  const inspection = useForgeStore((s) => s.inspection);
  const selection = useForgeStore((s) => s.selection);
  const inspecting = useForgeStore((s) => s.inspecting);
  const select = useForgeStore((s) => s.select);
  const session = useForgeStore((s) => s.session);

  if (!selection) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center gap-3 px-6" style={{ color: 'var(--text-muted)' }}>
        <div className="w-12 h-12 rounded-xl flex items-center justify-center" style={{ background: 'rgba(96,165,250,0.1)' }}>
          <Microscope size={22} style={{ color: '#60a5fa' }} />
        </div>
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Click any neuron or layer header</p>
        <p className="text-xs leading-relaxed">
          The microscope shows what that component computes for the selected input: its real weights, bias,
          weighted sum, activation, gradients, and how it behaves across the whole dataset.
        </p>
      </div>
    );
  }

  const stale = !inspection || inspection.kind !== selection.kind;
  const t = inspection && !stale ? title(inspection) : null;

  return (
    <div className="h-full flex flex-col min-h-0">
      <div className="flex-shrink-0 px-3 py-2.5 border-b" style={{ borderColor: 'var(--border)' }}>
        <div className="flex items-center gap-2">
          <span className="text-base font-semibold font-mono" style={{ color: 'var(--text-primary)' }}>{t?.name ?? '…'}</span>
          {inspecting && <Loader2 size={13} className="animate-spin" style={{ color: 'var(--text-faint)' }} />}
          <button type="button" aria-label="Close inspector" className="ml-auto p-1 rounded hover:bg-white/5" onClick={() => select(null)}>
            <X size={14} style={{ color: 'var(--text-muted)' }} />
          </button>
        </div>
        {t && <div className="text-[11px] mt-0.5" style={{ color: 'var(--text-muted)' }}>{t.sub}</div>}
        {inspection && !stale && (
          <div className="mt-1.5 flex flex-wrap items-center gap-1">
            <ProvenanceBadge provenance={inspection.provenance} />
            <ForgeStatePill />
          </div>
        )}
      </div>
      <div className="flex-1 min-h-0 overflow-y-auto p-3" style={{ opacity: inspecting ? 0.6 : 1, transition: 'opacity 0.15s' }}>
        {inspection && !stale && (
          inspection.kind === 'neuron' ? <NeuronPanel n={inspection} structure={structure} />
            : inspection.kind === 'layer' ? <LayerPanel l={inspection} />
              : <ConnectionPanel c={inspection} structure={structure} />
        )}
        {inspection && !stale && session && session.epoch > 0 && <div className="mt-2.5"><ThroughTime compact /></div>}
      </div>
    </div>
  );
}
