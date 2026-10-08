import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle, RotateCcw } from 'lucide-react';
import { readStoredLang } from '../i18n/lang';

const TEXT = {
  en: {
    title: 'This view stopped working',
    app: 'Neural Forge stopped working',
    body: 'An unexpected error interrupted the interface. Your model is still on the backend: retry this view or reload the page and press Build again if needed.',
    translate: 'If you are using your browser\'s page translator, turn it off: Neural Forge has its own Italian and English versions (language selector in the top bar).',
    retry: 'Retry',
    reload: 'Reload page',
    details: 'Technical details',
  },
  it: {
    title: 'Questa vista si è interrotta',
    app: 'Neural Forge si è interrotto',
    body: "Un errore imprevisto ha interrotto l'interfaccia. Il tuo modello è ancora sul backend: riprova questa vista o ricarica la pagina e, se serve, premi di nuovo Costruisci.",
    translate: 'Se stai usando il traduttore del browser, disattivalo: Neural Forge ha già la sua versione italiana e inglese (selettore della lingua nella barra in alto).',
    retry: 'Riprova',
    reload: 'Ricarica pagina',
    details: 'Dettagli tecnici',
  },
};

interface Props { children: ReactNode; scope: 'app' | 'view'; resetKey?: unknown }
interface State { error: Error | null; prevKey?: unknown }

/**
 * Catches render errors so one broken view (or a browser extension that
 * rewrites the DOM under React, such as page translators) never leaves a
 * blank screen.  A view-level boundary resets when `resetKey` changes.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null, prevKey: this.props.resetKey };

  static getDerivedStateFromError(error: Error): Partial<State> {
    return { error };
  }

  static getDerivedStateFromProps(props: Props, state: State): Partial<State> | null {
    if (props.resetKey !== state.prevKey) return { error: null, prevKey: props.resetKey };
    return null;
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.warn('[Neural Forge] view error caught by boundary:', error.message, info.componentStack?.split('\n')[1]?.trim());
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    const t = TEXT[readStoredLang()];
    const domError = /removeChild|insertBefore|not a child of this node/i.test(error.message);
    return (
      <div role="alert" className="h-full w-full flex items-center justify-center p-8" style={{ background: 'var(--bg-base)', color: 'var(--text-primary)', minHeight: this.props.scope === 'app' ? '100vh' : undefined }}>
        <div className="max-w-md text-center space-y-3">
          <AlertTriangle size={26} className="mx-auto" style={{ color: 'var(--warn)' }} aria-hidden="true" />
          <h2 className="text-lg font-semibold m-0">{this.props.scope === 'app' ? t.app : t.title}</h2>
          <p className="text-sm leading-relaxed m-0" style={{ color: 'var(--text-muted)' }}>{t.body}</p>
          {domError && <p className="text-sm leading-relaxed m-0" style={{ color: 'var(--text-warn)' }}>{t.translate}</p>}
          <div className="flex justify-center gap-2 pt-1">
            {this.props.scope === 'view' && (
              <button type="button" className="btn-secondary" onClick={() => this.setState({ error: null })}><RotateCcw size={14} />{t.retry}</button>
            )}
            <button type="button" className="btn-primary" onClick={() => window.location.reload()}>{t.reload}</button>
          </div>
          <details className="text-left text-xs" style={{ color: 'var(--text-faint)' }}>
            <summary className="cursor-pointer">{t.details}</summary>
            <pre className="whitespace-pre-wrap mt-1">{error.message}</pre>
          </details>
        </div>
      </div>
    );
  }
}
