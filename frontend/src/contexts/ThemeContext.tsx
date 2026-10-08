import { useEffect, useState, type ReactNode } from 'react';
import { ThemeContext, readStoredTheme, type Theme } from './theme';

function stored(): Theme {
  try {
    return readStoredTheme(localStorage.getItem('nv-theme'));
  } catch {
    return 'dark';
  }
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(stored);

  const setTheme = (t: Theme) => {
    setThemeState(t);
    try { localStorage.setItem('nv-theme', t); } catch { /* storage unavailable */ }
  };

  useEffect(() => {
    if (theme === 'dark') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
  }, [theme]);

  return <ThemeContext.Provider value={{ theme, setTheme }}>{children}</ThemeContext.Provider>;
}
