import { expect, test } from '@playwright/test';
import { watchConsole } from './helpers';

test('H · Explore (Italian): build → train → rewind → most important neuron, all on the real model', async ({ page }) => {
  const errors = watchConsole(page);
  await page.addInitScript(() => {
    localStorage.setItem('nf-welcome-seen', '1');
    localStorage.setItem('nf-lang', 'it');
    localStorage.setItem('nf-experience', 'explore');
  });
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('lang', 'it');
  const explore = page.getByRole('region', { name: 'Esplora' });
  await expect(explore.getByRole('heading', { name: 'Costruisci una vera rete neurale' })).toBeVisible();

  await explore.getByRole('button', { name: /^Piccola/ }).click();
  await explore.getByRole('button', { name: 'Crea la rete' }).click();
  await expect(explore.getByText(/Creata una vera rete PyTorch: 2 → 4 → 2/)).toBeVisible();

  await explore.getByRole('button', { name: /^Addestra 30 epoche$/ }).click();
  await expect(explore.getByText(/In 30 epoche l’accuratezza è passata da/)).toBeVisible({ timeout: 60_000 });
  await expect(explore.getByRole('img', { name: /Regioni di decisione della rete all'epoca 30/ })).toBeVisible();

  // Rewind to the first stored checkpoint: the map follows the real checkpoint.
  const slider = explore.getByRole('slider', { name: 'Epoca di addestramento' });
  await slider.focus();
  await page.keyboard.press('Home');
  await expect(explore.getByRole('img', { name: /all'epoca 0$/ })).toBeVisible();
  await page.keyboard.press('End');

  await explore.getByRole('button', { name: 'Trova il neurone più importante' }).click();
  await expect(explore.getByTestId('explore-key-result')).toContainText(/Su 4 neuroni nascosti|Nessun neurone/, { timeout: 60_000 });

  // Leaving for the Laboratory keeps the same real model.
  await explore.getByRole('button', { name: 'Apri nel Laboratorio' }).click();
  await expect(page.getByRole('tablist', { name: 'Area di lavoro' }).getByRole('tab', { name: /Rete/ })).toHaveAttribute('aria-selected', 'true');
  await expect(page.getByText(/2 → 4 → 2|2→4→2/).first()).toBeVisible();
  expect(errors).toEqual([]);
});

test('H · language selector switches the whole shell and persists', async ({ page }) => {
  const errors = watchConsole(page);
  await page.addInitScript(() => {
    localStorage.setItem('nf-welcome-seen', '1');
    if (!localStorage.getItem('nf-lang')) localStorage.setItem('nf-lang', 'en');
    if (!localStorage.getItem('nf-experience')) localStorage.setItem('nf-experience', 'lab');
  });
  await page.goto('/');
  await expect(page.getByRole('tablist', { name: 'Workspace' }).getByRole('tab', { name: /Time/ })).toBeVisible();
  await page.getByRole('combobox', { name: 'Language' }).selectOption('it');
  await expect(page.locator('html')).toHaveAttribute('lang', 'it');
  await expect(page.getByRole('tablist', { name: 'Area di lavoro' }).getByRole('tab', { name: /Tempo|Macchina del tempo/ })).toBeVisible();
  await expect(page.getByRole('button', { name: /^Costruisci$/ })).toBeVisible();
  await page.reload();
  await expect(page.getByRole('combobox', { name: 'Lingua' })).toHaveValue('it');
  await page.getByRole('combobox', { name: 'Lingua' }).selectOption('en');
  await expect(page.getByRole('button', { name: /^Build$/ })).toBeVisible();
  expect(errors).toEqual([]);
});
