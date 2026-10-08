import { expect, test } from '@playwright/test';
import { inspector, watchConsole } from './helpers';

test('F · first-run welcome → "Break the network" demo runs the real what-if', async ({ page }) => {
  const errors = watchConsole(page);
  await page.goto('/');
  const dialog = page.getByRole('dialog', { name: 'Step inside a neural network.' });
  await expect(dialog).toBeVisible();
  await dialog.getByRole('button', { name: /Break the network/ }).click();
  await expect(page.getByTestId('whatif-result')).toBeVisible({ timeout: 60_000 });
  await expect(inspector(page).getByText(/Disable L\dN\d/).first()).toBeVisible();
  // The welcome is not shown again.
  await page.reload();
  await expect(page.getByRole('dialog')).toBeHidden();
  expect(errors).toEqual([]);
});

test('F · presentation mode walks the first steps on the real model', async ({ page }) => {
  const errors = watchConsole(page);
  await page.addInitScript(() => localStorage.setItem('nf-welcome-seen', '1'));
  await page.goto('/');
  await page.getByRole('button', { name: /Present/ }).click();
  const bar = page.getByRole('region', { name: 'Presentation' });
  await expect(bar).toContainText(/A real PyTorch MLP with \d+ parameters/);
  await bar.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(bar).toContainText(/epochs of real Adam training\. Accuracy went from/, { timeout: 60_000 });
  await bar.getByRole('button', { name: 'Next', exact: true }).click();
  await expect(bar).toContainText(/Back at epoch \d+/);
  await bar.getByRole('button', { name: 'Exit presentation' }).click();
  await expect(bar).toBeHidden();
  expect(errors).toEqual([]);
});
