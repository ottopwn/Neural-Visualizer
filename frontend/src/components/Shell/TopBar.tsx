import { FlaskRound, GraduationCap, Moon, MonitorPlay, PanelLeft, PanelRight, Sparkles, Sun } from 'lucide-react';
import { MODES, useWorkspace, type WorkspaceMode } from '../../app/workspace';
import { useTheme } from '../../contexts/theme';
import { useForgeStore } from '../../forge/store';
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

  return (
    <header className="flex items-center gap-3 px-3 h-12 flex-shrink-0 border-b"
      style={{ borderColor: 'var(--border)', background: 'var(--bg-sidebar)' }}>
      <button type="button" className="btn-ghost p-1.5" aria-label={leftOpen ? 'Hide experiment panel' : 'Show experiment panel'}
        aria-pressed={leftOpen} title="Experiment panel" onClick={() => setLeftOpen(!leftOpen)}>
        <PanelLeft size={16} />
      </button>

      <div className="flex items-center gap-2 flex-shrink-0 pr-2">
        <BrandMark />
        <div className="leading-none">
          <div className="text-[14px] font-semibold tracking-tight" style={{ color: 'var(--text-primary)' }}>Neural Forge</div>
          <div className="text-[10.5px] mt-0.5 hidden 2xl:block" style={{ color: 'var(--text-faint)' }}>Inspect. Intervene. Understand.</div>
        </div>
      </div>

      <nav className="flex items-center gap-0.5 min-w-0 overflow-x-auto" role="tablist" aria-label="Workspace">
        {MODES.filter((m) => availableModes.has(m.id)).map((m) => {
          const active = mode === m.id;
          return (
            <button key={m.id} type="button" role="tab" aria-selected={active} title={m.hint}
              onClick={() => setMode(m.id)}
              className="relative px-2.5 h-12 text-[13px] font-medium whitespace-nowrap transition-colors"
              style={{ color: active ? 'var(--text-primary)' : 'var(--text-muted)' }}>
              <span className="hidden xl:inline">{m.label}</span>
              <span className="xl:hidden">{m.short}</span>
              {active && <span className="absolute left-2 right-2 bottom-0 h-[2px] rounded-full" style={{ background: 'var(--accent)' }} />}
            </button>
          );
        })}
      </nav>

      <div className="ml-auto flex items-center gap-2 flex-shrink-0">
        <ForgeStatePill />
        <div className="seg" role="group" aria-label="Explanation level">
          <button type="button" aria-pressed={expMode === 'learn'} onClick={() => setExpMode('learn')}
            title="Learn: plain-language explanations generated from the real values">
            <span className="inline-flex items-center gap-1"><GraduationCap size={13} />Learn</span>
          </button>
          <button type="button" aria-pressed={expMode === 'lab'} onClick={() => setExpMode('lab')}
            title="Lab: equations, tensor shapes, gradients and raw numbers">
            <span className="inline-flex items-center gap-1"><FlaskRound size={13} />Lab</span>
          </button>
        </div>
        <button type="button" className="btn-secondary !py-1.5 !px-2.5 !text-xs" onClick={onDemo} title="Guided demos that run the real application">
          <Sparkles size={13} /><span className="hidden lg:inline">Demos</span>
        </button>
        <button type="button" className="btn-secondary !py-1.5 !px-2.5 !text-xs" onClick={onPresent} title="Presentation mode: a curated 10-step journey">
          <MonitorPlay size={13} /><span className="hidden lg:inline">Present</span>
        </button>
        <button type="button" className="btn-ghost p-1.5" onClick={() => setTheme(theme === 'paper' ? 'dark' : 'paper')}
          aria-label={theme === 'paper' ? 'Switch to dark theme' : 'Switch to Paper (light) theme'} title="Theme">
          {theme === 'paper' ? <Moon size={16} /> : <Sun size={16} />}
        </button>
        <button type="button" className="btn-ghost p-1.5" aria-label={rightOpen ? 'Hide inspector' : 'Show inspector'}
          aria-pressed={rightOpen} title="Inspector" onClick={() => setRightOpen(!rightOpen)}>
          <PanelRight size={16} />
        </button>
      </div>
    </header>
  );
}
