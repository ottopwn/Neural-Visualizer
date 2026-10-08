import { Compass, FlaskConical, FlaskRound, GraduationCap, Languages, Moon, MonitorPlay, PanelLeft, PanelRight, Sparkles, Sun } from 'lucide-react';
import { MODES, useWorkspace, type WorkspaceMode } from '../../app/workspace';
import { useTheme } from '../../contexts/theme';
import { useForgeStore } from '../../forge/store';
import { useI18n, useT } from '../../i18n';
import { LANGS, type Lang } from '../../i18n/lang';
import { ForgeStatePill } from '../Forge/TimeMachine/StatePill';

/** Small mark: three layers of nodes. Static, no decoration beyond the logo. */
export function BrandMark({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <rect x="0.5" y="0.5" width="23" height="23" rx="6" fill="var(--accent)" />
      <g stroke="#fff" strokeOpacity="0.55" strokeWidth="1">
        <path d="M6 8 L12 6 M6 8 L12 12 M6 8 L12 18 M6 16 L12 6 M6 16 L12 12 M6 16 L12 18 M12 6 L18 12 M12 12 L18 12 M12 18 L18 12" />
      </g>
      <g fill="#fff">
        <circle cx="6" cy="8" r="1.8" /><circle cx="6" cy="16" r="1.8" />
        <circle cx="12" cy="6" r="1.6" /><circle cx="12" cy="12" r="1.6" /><circle cx="12" cy="18" r="1.6" />
        <circle cx="18" cy="12" r="2" />
      </g>
    </svg>
  );
}

interface Props {
  onDemo: () => void;
  onPresent: () => void;
  availableModes: Set<WorkspaceMode>;
}

export function TopBar({ onDemo, onPresent, availableModes }: Props) {
  const { theme, setTheme } = useTheme();
  const mode = useWorkspace((s) => s.mode);
  const setMode = useWorkspace((s) => s.setMode);
  const leftOpen = useWorkspace((s) => s.leftOpen);
  const rightOpen = useWorkspace((s) => s.rightOpen);
  const setLeftOpen = useWorkspace((s) => s.setLeftOpen);
  const setRightOpen = useWorkspace((s) => s.setRightOpen);
  const expMode = useForgeStore((s) => s.mode);
  const setExpMode = useForgeStore((s) => s.setMode);
  const experience = useWorkspace((s) => s.experience);
  const setExperience = useWorkspace((s) => s.setExperience);
  const lang = useI18n((s) => s.lang);
  const setLang = useI18n((s) => s.setLang);
  const t = useT();
  const lab = experience === 'lab';

  return (
    <header className="app-topbar flex items-center gap-3 px-3 h-12 flex-shrink-0 border-b"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
      {lab && (
        <button type="button" className="btn-ghost p-1.5" aria-label={leftOpen ? t.topbar.hideExperiment : t.topbar.showExperiment}
          aria-pressed={leftOpen} title={t.topbar.experimentPanel} onClick={() => setLeftOpen(!leftOpen)}>
          <PanelLeft size={16} />
        </button>
      )}

      <div className="flex items-center gap-2 flex-shrink-0 pr-2">
        <BrandMark />
        <div className="leading-none">
          <div className="text-[14px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>Neural Forge</div>
          <div className="text-[10.5px] mt-0.5 hidden 2xl:block" style={{ color: 'var(--text-faint)' }}>{t.topbar.tagline}</div>
        </div>
      </div>

      <div className="seg flex-shrink-0" role="group" aria-label={t.experience.label}>
        <button type="button" aria-pressed={!lab} onClick={() => setExperience('explore')} title={t.experience.exploreHint}>
          <span className="inline-flex items-center gap-1"><Compass size={13} />{t.experience.explore}</span>
        </button>
        <button type="button" aria-pressed={lab} onClick={() => setExperience('lab')} title={t.experience.labHint}>
          <span className="inline-flex items-center gap-1"><FlaskConical size={13} />{t.experience.lab}</span>
        </button>
      </div>

      {lab && <nav className="flex items-center gap-0.5 min-w-0 overflow-x-auto" role="tablist" aria-label={t.topbar.workspace}>
        {MODES.filter((id) => availableModes.has(id)).map((id) => {
          const active = mode === id;
          const m = t.modes[id];
          return (
            <button key={id} type="button" role="tab" aria-selected={active} title={m.hint}
              onClick={() => setMode(id)}
              className="relative px-2.5 h-12 text-[13px] font-medium whitespace-nowrap transition-colors"
              style={{ color: active ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              <span className="hidden 2xl:inline">{m.label}</span>
              <span className="2xl:hidden">{m.short}</span>
              {active && <span className="absolute left-2 right-2 bottom-0 h-[2px] rounded-full" style={{ background: 'var(--accent)' }} />}
            </button>
          );
        })}
      </nav>}

      <div className="ml-auto flex items-center gap-2 flex-shrink-0">
        {lab && <ForgeStatePill />}
        {lab && (
          <div className="seg" role="group" aria-label={t.topbar.explanationLevel}>
            <button type="button" aria-pressed={expMode === 'learn'} onClick={() => setExpMode('learn')} title={t.topbar.learnHint}>
              <span className="inline-flex items-center gap-1"><GraduationCap size={13} />{t.topbar.learn}</span>
            </button>
            <button type="button" aria-pressed={expMode === 'lab'} onClick={() => setExpMode('lab')} title={t.topbar.labLevelHint}>
              <span className="inline-flex items-center gap-1"><FlaskRound size={13} />{t.topbar.labLevel}</span>
            </button>
          </div>
        )}
        <button type="button" className="btn-secondary !py-1.5 !px-2.5 !text-xs" onClick={onDemo} title={t.topbar.demosHint}>
          <Sparkles size={13} /><span className="hidden lg:inline">{t.topbar.demos}</span>
        </button>
        <button type="button" className="btn-secondary !py-1.5 !px-2.5 !text-xs" onClick={onPresent} title={t.topbar.presentHint}>
          <MonitorPlay size={13} /><span className="hidden lg:inline">{t.topbar.present}</span>
        </button>
        <label className="flex items-center gap-1 text-[12px]" style={{ color: 'var(--text-muted)' }} title={t.lang.label}>
          <Languages size={14} aria-hidden="true" />
          <select className="select-base !py-1 !pl-1.5 !pr-6 !text-xs !w-auto" aria-label={t.lang.label} value={lang}
            onChange={(e) => setLang(e.target.value as Lang)}>
            {LANGS.map((l) => <option key={l} value={l}>{t.lang[l]}</option>)}
          </select>
        </label>
        <button type="button" className="btn-ghost p-1.5" onClick={() => setTheme(theme === 'paper' ? 'dark' : 'paper')}
          aria-label={theme === 'paper' ? t.topbar.toDark : t.topbar.toPaper} title={t.topbar.theme}>
          {theme === 'paper' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
        {lab && (
          <button type="button" className="btn-ghost p-1.5" aria-label={rightOpen ? t.topbar.hideInspector : t.topbar.showInspector}
            aria-pressed={rightOpen} title={t.topbar.inspector} onClick={() => setRightOpen(!rightOpen)}>
            <PanelRight size={16} />
          </button>
        )}
      </div>
    </header>
  );
}
