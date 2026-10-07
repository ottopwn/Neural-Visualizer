import { ArrowRight, Loader2, MonitorPlay, X } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { DEMOS, JOURNEY, useGuide, usePresentation } from '../../app/demos';
import { useWorkspace } from '../../app/workspace';
import { BrandMark } from './TopBar';

export const WELCOME_KEY = 'nf-welcome-seen';

/** First-run welcome and the guided-demo launcher (also opened from the top bar). */
export function Welcome() {
  const open = useWorkspace((s) => s.welcomeOpen);
  const setOpen = useWorkspace((s) => s.setWelcomeOpen);
  const run = useGuide((s) => s.run);
  const running = useGuide((s) => s.running);
  const start = usePresentation((s) => s.start);
  const firstBtn = useRef<HTMLButtonElement>(null);

  const close = () => {
    setOpen(false);
    try { localStorage.setItem(WELCOME_KEY, '1'); } catch { /* storage unavailable */ }
  };

  useEffect(() => {
    if (!open) return;
    firstBtn.current?.focus();
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') close(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-6" style={{ background: 'rgba(5,8,12,0.55)' }}
      onMouseDown={(e) => { if (e.target === e.currentTarget) close(); }}>
      <div role="dialog" aria-modal="true" aria-labelledby="welcome-title"
        className="w-full max-w-[880px] max-h-[calc(100vh-48px)] overflow-y-auto rounded-xl border"
        style={{ background: 'var(--bg-card)', borderColor: 'var(--border-soft)', boxShadow: 'var(--shadow-pop)' }}>
        <div className="flex items-start gap-4 px-7 pt-7 pb-5 border-b" style={{ borderColor: 'var(--border)' }}>
          <BrandMark size={40} />
          <div className="flex-1">
            <h1 id="welcome-title" className="text-[22px] font-semibold tracking-tight m-0" style={{ color: 'var(--text-primary)' }}>
              Step inside a neural network.
            </h1>
            <p className="mt-1.5 text-[14px] leading-relaxed max-w-[620px]" style={{ color: 'var(--text-muted)' }}>
              Neural Forge trains a real PyTorch network on your machine, lets you watch it learn, inspect what every neuron computes,
              and change it to see what happens. Every number on screen is computed by the model — nothing is simulated.
            </p>
          </div>
          <button type="button" className="btn-ghost p-1.5" aria-label="Close" onClick={close}><X size={18} /></button>
        </div>

        <div className="px-7 py-5">
          <button ref={firstBtn} type="button" disabled={!!running}
            onClick={() => { close(); void start(); }}
            className="w-full text-left rounded-lg border p-4 flex items-center gap-4 transition-colors hover:bg-[var(--bg-hover)]"
            style={{ borderColor: 'var(--accent)', background: 'var(--bg-active)' }}>
            <MonitorPlay size={24} style={{ color: 'var(--accent)' }} aria-hidden="true" />
            <span className="flex-1">
              <span className="block text-[15px] font-semibold" style={{ color: 'var(--text-primary)' }}>Take the {JOURNEY.length}-step tour</span>
              <span className="block text-[13px] mt-0.5" style={{ color: 'var(--text-muted)' }}>
                About 3 minutes: build → train → rewind → inspect a neuron → disable it → follow the forward and backward pass → 3D.
              </span>
            </span>
            <ArrowRight size={18} style={{ color: 'var(--accent)' }} aria-hidden="true" />
          </button>

          <div className="eyebrow mt-6 mb-2.5">Or jump straight to one experiment</div>
          <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))' }}>
            {DEMOS.map((d) => (
              <button key={d.id} type="button" disabled={!!running}
                onClick={() => { close(); void run(d); }}
                className="text-left rounded-lg border p-3.5 transition-colors hover:bg-[var(--bg-hover)] disabled:opacity-50"
                style={{ borderColor: 'var(--border-soft)' }}>
                <span className="block text-[14px] font-semibold" style={{ color: 'var(--text-primary)' }}>{d.title}</span>
                <span className="block text-[12.5px] mt-1 leading-relaxed" style={{ color: 'var(--text-muted)' }}>{d.body}</span>
              </button>
            ))}
          </div>
          <div className="mt-5 flex items-center justify-between gap-3 flex-wrap">
            <p className="text-[12px] m-0" style={{ color: 'var(--text-faint)' }}>
              Real model internals are available for ANN (MLP) models. Other architectures are labelled as illustrative diagrams.
            </p>
            <button type="button" className="btn-secondary" onClick={close}>Explore on my own</button>
          </div>
        </div>
      </div>
    </div>
  );
}

/** Status of a running one-click demo. */
export function GuideToast() {
  const running = useGuide((s) => s.running);
  const error = useGuide((s) => s.error);
  const clear = useGuide((s) => s.clearError);
  if (!running && !error) return null;
  return (
    <div role="status" className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[55] flex items-center gap-2 px-3.5 py-2 rounded-lg border text-[13px]"
      style={{ background: 'var(--bg-card)', borderColor: error ? 'var(--neg)' : 'var(--border-soft)', boxShadow: 'var(--shadow-pop)', color: error ? 'var(--text-neg)' : 'var(--text-primary)' }}>
      {running ? <><Loader2 size={14} className="animate-spin" />Running demo: {running}…</> : <>{error}<button type="button" className="btn-ghost p-1" aria-label="Dismiss" onClick={clear}><X size={14} /></button></>}
    </div>
  );
}
