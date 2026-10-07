import { expect, type Page } from '@playwright/test';

/** Open the app with a theme, skipping the first-run welcome. */
export async function open(page: Page, theme: 'dark' | 'paper' = 'dark') {
  await page.addInitScript((t) => {
    localStorage.setItem('nv-theme', t);
    localStorage.setItem('nf-welcome-seen', '1');
  }, theme);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: 'Step inside a real neural network' })).toBeVisible();
}

/** Build the default real ANN and train it once (20 epochs). */
export async function buildAndTrain(page: Page) {
  await page.getByRole('button', { name: /^(Build|Rebuild)$/ }).click();
  await expect(page.getByText(/Real model · \d+ parameters/)).toBeVisible();
  await page.getByRole('button', { name: /^Train \d+$/ }).click();
  await expect(page.getByText(/Epoch 20 · accuracy/)).toBeVisible();
}

export function workspace(page: Page, name: RegExp) {
  return page.getByRole('tablist', { name: 'Workspace' }).getByRole('tab', { name });
}

export function inspector(page: Page) {
  return page.getByRole('complementary', { name: 'Inspector' });
}

/** Fail the test on uncaught page errors and console errors. */
export function watchConsole(page: Page): string[] {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
  return errors;
}
