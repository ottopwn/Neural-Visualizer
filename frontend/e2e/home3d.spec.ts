import { expect, test } from '@playwright/test';
import { buildAndTrain, open, watchConsole, workspace } from './helpers';

test('I · home: three paths and a live real network', async ({ page }) => {
  const errors = watchConsole(page);
  await page.addInitScript(() => { if (!localStorage.getItem('nf-lang')) localStorage.setItem('nf-lang', 'en'); });
  await page.goto('/');
  const home = page.getByRole('dialog', { name: 'Step inside a neural network.' });
  await expect(home).toBeVisible();
  await expect(home.getByRole('button', { name: /EXPLORE/ })).toBeVisible();
  await expect(home.getByRole('button', { name: /LEARN/ })).toBeVisible();
  await expect(home.getByText(/Live: a real PyTorch network just initialised on this machine \(\d+ parameters\)/)).toBeVisible({ timeout: 30_000 });
  await home.getByRole('button', { name: /LABORATORY/ }).click();
  await expect(home).toBeHidden();
  await expect(workspace(page, /^Network$/)).toBeVisible();
  expect(errors).toEqual([]);
});

test('I · 3D: quality levels, aesthetic layer toggle, fly-through and the real neuron card', async ({ page }) => {
  const errors = watchConsole(page);
  await open(page);
  await buildAndTrain(page);
  await page.getByRole('button', { name: 'Inspect neuron L1N2' }).click();
  await workspace(page, /^3D$/).click();
  const view = page.getByTestId('forge-3d');
  await expect(view.locator('canvas')).toBeVisible();
  const card = page.getByTestId('neuron-card');
  await expect(card).toContainText('L1N2');
  await expect(card).toContainText('z = Σ w·a + b');
  await page.getByRole('group', { name: 'Quality' }).getByRole('button', { name: 'High' }).click();
  await expect(view.locator('canvas')).toBeVisible();
  await page.getByLabel('Glow & depth').uncheck();
  await page.getByLabel('Glow & depth').check();
  await page.getByRole('button', { name: 'Fly through' }).click();
  await expect(page.getByRole('button', { name: 'Stop flight' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Fly through' })).toBeVisible({ timeout: 15_000 });
  expect(errors).toEqual([]);
});
