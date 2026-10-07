import { useState } from 'react';
import type { Provenance } from '../../forge/types';

export function ProvenanceBadge({ provenance }: { provenance: Provenance }) {
  const parts = [
    provenance.is_latest ? `live weights · epoch ${provenance.checkpoint_epoch}` : `checkpoint · epoch ${provenance.checkpoint_epoch}`,
  ];
  if (provenance.interventions_applied) parts.push(`${provenance.interventions_applied} intervention${provenance.interventions_applied > 1 ? 's' : ''}`);
  return (
    <span
      className="inline-flex items-center gap-1 text-[10px] px-1.5 py-0.5 rounded border"
      style={{ borderColor: 'rgba(16,185,129,0.4)', color: '#6ee7b7', background: 'rgba(16,185,129,0.08)' }}
      title={provenance.note}
    >
      <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#10b981' }} />
      REAL · {parts.join(' · ')}
    </span>
  );
}

/** Number input + quick actions to apply a what-if value. */
export function ValueEditor({ label, current, original, onApply, onRestore }: {
  label: string;
  current: number;
  original: number | null; // non-null when currently edited
  onApply: (v: number) => void;
  onRestore?: () => void;
}) {
  const [draft, setDraft] = useState(current.toFixed(3));
  const parsed = Number(draft);
  const valid = draft.trim() !== '' && Number.isFinite(parsed);
  return (
    <div className="space-y-1.5">
      <div className="flex items-center gap-1.5">
        <span className="text-[11px] w-14" style={{ color: 'var(--text-muted)' }}>{label}</span>
        <input
          aria-label={`New ${label}`}
          className="input-base !py-1 !px-2 !text-xs font-mono flex-1 min-w-0"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && valid) onApply(parsed); }}
        />
        <button type="button" className="btn-primary !py-1 !px-2 text-xs" disabled={!valid} onClick={() => onApply(parsed)}>Apply</button>
      </div>
      <div className="flex flex-wrap gap-1">
        {[
          ['0', 0],
          ['× 2', current * 2],
          ['− (flip)', -current],
        ].map(([t, v]) => (
          <button key={t as string} type="button" className="btn-secondary !py-0.5 !px-2 text-[11px]"
            onClick={() => { setDraft((v as number).toFixed(3)); onApply(v as number); }}>
            {t}
          </button>
        ))}
        {original !== null && onRestore && (
          <button type="button" className="btn-secondary !py-0.5 !px-2 text-[11px] ml-auto" onClick={onRestore}>
            Restore {original.toFixed(3)}
          </button>
        )}
      </div>
    </div>
  );
}
