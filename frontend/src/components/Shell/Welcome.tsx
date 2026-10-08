import { ArrowRight, Compass, FlaskConical, Languages, Loader2, MonitorPlay, X } from 'lucide-react';
import { useEffect, useRef } from 'react';
import { DEMOS, useGuide, usePresentation } from '../../app/demos';
import { useWorkspace } from '../../app/workspace';
import { BrandMark } from './TopBar';
import { useI18n, useT } from '../../i18n';
import { LANGS, type Lang } from '../../i18n/lang';
import { LiveNetwork } from './LiveNetwork';

export const WELCOME_KEY = 'nf-welcome-seen';

/** Home: first-run landing (Explore / Learn / Laboratory) and the demo launcher (also opened from the top bar). */
export function Welcome() {
  const open = useWorkspace((s) => s.welcomeOpen);
  const setOpen = useWorkspace((s) => s.setWelcomeOpen);
  const run = useGuide((s) => s.run);
  const running = useGuide((s) => s.running);
  const start = usePresentation((s) => s.start);
  const setExperience = useWorkspace((s) => s.setExperience);
  const t = useT();
  const lang = useI18n((s) => s.lang);
  const setLang = useI18n((s) => s.setLang);
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

  const choose = (e: 'explore' | 'lab') => { close(); setExperience(e); };
  const choices = [
    { key: 'explore', icon: <Compass size={22} />, text: t.home.explore, onClick: () => choose('explore'), primary: true },
    { key: 'learn', icon: <MonitorPlay size={22} />, text: t.home.learn, onClick: () => { close(); void start(); }, primary: false },
    { key: 'lab', icon: <FlaskConical size={22} />, text: t.home.lab, onClick: () => choose('lab'), primary: false },
  ];

  return (
    <div className="fixed inset-0 z-[60] overflow-y-auto home-bg">
      <div role="dialog" aria-modal="true" aria-labelledby="welcome-title" className="min-h-full max-w-[1240px] mx-auto px-6 py-6 flex flex-col">
        <div className="flex items-center gap-2.5">
          <BrandMark size={30} />
          <span className="text-[15px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>{t.home.eyebrow}</span>
          <div className="ml-auto flex items-center gap-2">
            <label className="flex items-center gap-1 text-[12px]" style={{ color: 'var(--text-muted)' }}>
              <Languages size={14} aria-hidden="true" />
              <select className="select-base !py-1 !pl-1.5 !pr-6 !text-xs !w-auto" aria-label={t.lang.label} value={lang}
                onChange={(e) => setLang(e.target.value as Lang)}>
                {LANGS.map((l) => <option key={l} value={l}>{t.lang[l]}</option>)}
              </select>
            </label>
            <button type="button" className="btn-ghost p-1.5" aria-label={t.common.close} onClick={close}><X size={18} /></button>
          </div>
        </div>

        <div className="flex-1 grid gap-8 lg:grid-cols-[1.05fr_1fr] items-center py-8">
          <div>
            <h1 id="welcome-title" className="home-title m-0">{t.welcome.title}</h1>
            <p className="mt-4 text-[15.5px] leading-relaxed max-w-[560px]" style={{ color: 'var(--text-muted)' }}>{t.welcome.intro}</p>
            <div className="mt-7 grid gap-3">
              {choices.map((c, i) => (
                <button key={c.key} ref={i === 0 ? firstBtn : undefined} type="button" disabled={!!running} onClick={c.onClick}
                  className="home-choice" data-primary={c.primary || undefined}>
                  <span className="home-choice-icon" aria-hidden="true">{c.icon}</span>
                  <span className="flex-1 min-w-0">
                    <span className="block text-[10.5px] font-semibold tracking-[0.14em]" style={{ color: 'var(--accent-text)' }}>{c.text.tag}</span>
                    <span className="block text-[16px] font-semibold" style={{ color: 'var(--text-primary)' }}>{c.text.title}</span>
                    <span className="block text-[13px] mt-0.5 leading-snug" style={{ color: 'var(--text-muted)' }}>{c.text.body}</span>
                  </span>
                  <ArrowRight size={18} style={{ color: 'var(--accent)' }} aria-hidden="true" />
                </button>
              ))}
            </div>
          </div>
          <div className="home-stage">
            <LiveNetwork />
          </div>
        </div>

        <div>
          <div className="eyebrow mb-2.5">{t.welcome.jump}</div>
          <div className="grid gap-2.5" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))' }}>
            {DEMOS.map((d) => (
              <button key={d.id} type="button" disabled={!!running}
                onClick={() => { close(); void run(d); }}
                className="text-left flex flex-col justify-start rounded-xl border p-3.5 transition-colors hover:bg-[var(--bg-hover)] disabled:opacity-50"
                style={{ borderColor: 'var(--border-soft)', background: 'color-mix(in srgb, var(--bg-card) 70%, transparent)' }}>
                <span className="block text-[14px] font-semibold" style={{ color: 'var(--text-primary)' }}>{t.demos[d.id].title}</span>
                <span className="block text-[12.5px] mt-1 leading-relaxed" style={{ color: 'var(--text-muted)' }}>{t.demos[d.id].body}</span>
              </button>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between gap-3 flex-wrap">
            <p className="text-[12px] m-0" style={{ color: 'var(--text-faint)' }}>{t.welcome.realNote}</p>
            <button type="button" className="btn-secondary" onClick={close}>{t.welcome.ownWay}</button>
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
  const t = useT();
  if (!running && !error) return null;
  return (
    <div role="status" className="fixed bottom-4 left-1/2 -translate-x-1/2 z-[55] flex items-center gap-2 px-3.5 py-2 rounded-lg border text-[13px]"
      style={{ background: 'var(--bg-card)', borderColor: error ? 'var(--neg)' : 'var(--border-soft)', boxShadow: 'var(--shadow-pop)', color: error ? 'var(--text-neg)' : 'var(--text-primary)' }}>
      {running ? <><Loader2 size={14} className="animate-spin" />{t.welcome.running(t.demos[running as keyof typeof t.demos]?.title ?? running)}</> : <>{error}<button type="button" className="btn-ghost p-1" aria-label={t.common.dismiss} onClick={clear}><X size={14} /></button></>}
    </div>
  );
}
