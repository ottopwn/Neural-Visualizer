// Native interface translation (English / Italian).
//
// The UI text lives in typed dictionaries (en.ts is the reference shape; it.ts
// must match it).  Components read the active dictionary with useT(); plain
// modules (demo narration, status messages) use tr().  The choice is
// persisted under `nf-lang` and mirrored to <html lang>.

import { create } from 'zustand';
import { en, type Dict } from './en';
import { it } from './it';
import { LANG_KEY, readStoredLang, type Lang } from './lang';

export type { Lang } from './lang';
export type { Dict } from './en';

const DICTS: Record<Lang, Dict> = { en, it };

interface I18nState {
  lang: Lang;
  setLang: (l: Lang) => void;
}

function applyLang(l: Lang) {
  if (typeof document !== 'undefined') document.documentElement.lang = l;
}

const initial = readStoredLang();
applyLang(initial);

export const useI18n = create<I18nState>((set) => ({
  lang: initial,
  setLang: (lang) => {
    try { localStorage.setItem(LANG_KEY, lang); } catch { /* storage unavailable */ }
    applyLang(lang);
    set({ lang });
  },
}));

/** The active dictionary (re-renders on language change). */
export function useT(): Dict {
  return DICTS[useI18n((s) => s.lang)];
}

/** The active dictionary, outside React. */
export function tr(): Dict {
  return DICTS[useI18n.getState().lang];
}
