import { createContext, useContext } from 'react';

/** Two maintained themes: dark (default) and Paper (light). */
export type Theme = 'dark' | 'paper';

export interface ThemeCtx {
  theme: Theme;
  setTheme: (t: Theme) => void;
}

export const ThemeContext = createContext<ThemeCtx>({ theme: 'dark', setTheme: () => {} });

export const useTheme = () => useContext(ThemeContext);

/** Older builds offered 'cyberpunk' / 'matrix'; anything unknown falls back to dark. */
export function readStoredTheme(raw: string | null): Theme {
  return raw === 'paper' ? 'paper' : 'dark';
}
