import { ChevronLeft, ChevronRight, Loader2, X } from 'lucide-react';
import { useEffect } from 'react';
import { JOURNEY, usePresentation } from '../../app/demos';
import { useForgeStore } from '../../forge/store';
import { useTimeMachine } from '../../forge/timeMachine';
import { useI18n, useT } from '../../i18n';

/**
 * Presentation mode: a curated journey through the real application.  Each
 * step performs real actions; the narration is computed from the resulting
 * state (re-rendered whenever the model state changes).
 */
export function PresentationBar() {
  const { active, index, busy, error, next, back, exit, goTo } = usePresentation();
  // Subscribe to the state the narration reads, so it updates when values arrive.
  useForgeStore((s) => s.session);
  useForgeStore((s) => s.trace);
  useForgeStore((s) => s.comparison);
  useForgeStore((s) => s.selection);
  useTimeMachine((s) => s.cursor);
  useI18n((s) => s.lang);
  const t = useT();

  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') exit();
      if (e.key === 'PageDown' || (e.key === 'ArrowRight' && e.shiftKey)) { e.preventDefault(); void next(); }
      if (e.key === 'PageUp' || (e.key === 'ArrowLeft' && e.shiftKey)) { e.preventDefault(); void back(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active, next, back, exit]);

  if (!active) return null;
  const step = JOURNEY[index];
  const last = index === JOURNEY.length - 1;

  return (
    <div role="region" aria-label={t.presentation.region} className="flex-shrink-0 border-t px-4 py-3 flex items-center gap-4"
      style={{ borderColor: 'var(--accent)', background: 'var(--bg-card)', boxShadow: '0 -8px 24px rgba(0,0,0,0.18)' }}>
      <div className="flex items-center gap-1 flex-shrink-0" aria-label={t.presentation.steps}>
        {JOURNEY.map((s, i) => (
          <button key={i} type="button" onClick={() => void goTo(i)} disabled={busy} aria-label={t.presentation.step(i + 1, s.title())}
            aria-current={i === index ? 'step' : undefined}
            className="h-1.5 rounded-full transition-all" style={{ width: i === index ? 22 : 10, background: i <= index ? 'var(--accent)' : 'var(--border-soft)' }} />
        ))}
      </div>
      <div className="flex-1 min-w-0" aria-live="polite">
        <div className="text-[11px] font-semibold uppercase tracking-[0.08em]" style={{ color: 'var(--accent-text)' }}>
          {t.presentation.stepOf(index + 1, JOURNEY.length, step.title())}
        </div>
        <p className="text-[14px] leading-snug mt-0.5 m-0" style={{ color: 'var(--text-primary)' }}>
          {busy ? <span className="inline-flex items-center gap-1.5" style={{ color: 'var(--text-muted)' }}><Loader2 size={14} className="animate-spin" />{t.presentation.working}</span>
            : error ? <span style={{ color: 'var(--text-neg)' }}>{error}</span> : step.text()}
        </p>
      </div>
      <div className="flex items-center gap-2 flex-shrink-0">
        <button type="button" className="btn-secondary" onClick={() => void back()} disabled={busy || index === 0}><ChevronLeft size={15} />{t.common.back}</button>
        {last
          ? <button type="button" className="btn-primary" onClick={exit}>{t.common.finish}</button>
          : <button type="button" className="btn-primary" onClick={() => void next()} disabled={busy}>{t.common.next}<ChevronRight size={15} /></button>}
        <button type="button" className="btn-ghost p-1.5" aria-label={t.presentation.exit} onClick={exit}><X size={16} /></button>
      </div>
    </div>
  );
}
