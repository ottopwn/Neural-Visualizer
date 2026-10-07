import { ChevronDown, Crosshair, FlaskConical, Microscope } from 'lucide-react';
import { useState, type ReactNode } from 'react';
import { useForgeStore } from '../../forge/store';
import { Inspector } from '../Forge/Inspector';
import { ProbePicker } from '../Forge/ProbePicker';
import { WhatIfPanel } from '../Forge/WhatIfPanel';

function Block({ title, icon, children, defaultOpen = true, badge, testId }: {
  title: string; icon: ReactNode; children: ReactNode; defaultOpen?: boolean; badge?: ReactNode; testId?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-b" style={{ borderColor: 'var(--border)' }} data-testid={testId}>
      <button type="button" className="w-full flex items-center gap-2 px-3 py-2.5 text-left" aria-expanded={open} onClick={() => setOpen(!open)}>
        <span style={{ color: 'var(--text-faint)' }} aria-hidden="true">{icon}</span>
        <span className="eyebrow">{title}</span>
        {badge}
        <ChevronDown size={14} className="ml-auto" style={{ color: 'var(--text-faint)', transform: open ? 'none' : 'rotate(-90deg)', transition: 'transform 0.15s' }} />
      </button>
      {open && <div className="px-3 pb-3">{children}</div>}
    </section>
  );
}

/**
 * The global inspector column: the experiment's inputs and outputs (probe,
 * prediction, what-if before/after) and the Neural Microscope for the
 * selected component.  Every workspace shares it, so a selection or an edit
 * made anywhere is visible everywhere.
 */
export function InspectorPanel() {
  const session = useForgeStore((s) => s.session);
  const nIv = useForgeStore((s) => s.interventions.length);

  if (!session) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-center gap-2 px-6" style={{ color: 'var(--text-muted)' }}>
        <Microscope size={22} style={{ color: 'var(--text-faint)' }} aria-hidden="true" />
        <p className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Neural Microscope</p>
        <p className="text-xs leading-relaxed">Build a real ANN to inspect probe inputs, predictions, what-if edits and any neuron, layer or connection.</p>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto" aria-label="Inspector">
      <Block title="Probe input" icon={<Crosshair size={13} />} testId="probe-block">
        <ProbePicker />
      </Block>
      <Block title={nIv ? 'What-if' : 'Prediction'} icon={<FlaskConical size={13} />} testId="prediction-block"
        badge={nIv ? <span className="badge-orange">{nIv} edit{nIv > 1 ? 's' : ''}</span> : undefined}>
        <WhatIfPanel />
      </Block>
      <div className="px-3 pt-2.5 pb-1 flex items-center gap-2">
        <Microscope size={13} style={{ color: 'var(--text-faint)' }} aria-hidden="true" />
        <span className="eyebrow">Neural Microscope</span>
      </div>
      <Inspector structure={session.structure} embedded />
    </div>
  );
}
