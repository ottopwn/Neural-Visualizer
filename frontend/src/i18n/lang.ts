// Language selection, kept dependency-free so the error boundary can use it
// even when the React tree is broken.

export type Lang = 'en' | 'it';
export const LANGS: Lang[] = ['en', 'it'];
export const LANG_KEY = 'nf-lang';

/** Stored choice, else the browser language (Italian if it starts with "it"), else English. */
export function readStoredLang(): Lang {
  try {
    const v = localStorage.getItem(LANG_KEY);
    if (v === 'en' || v === 'it') return v;
  } catch { /* storage unavailable */ }
  const nav = typeof navigator !== 'undefined' ? navigator.language || '' : '';
  return nav.toLowerCase().startsWith('it') ? 'it' : 'en';
}
