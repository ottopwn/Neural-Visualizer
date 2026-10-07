import { expect, test } from '@playwright/test';
import { open, watchConsole, workspace } from './helpers';

test('G · Transformer Lab: real attention, causal mask and head ablation', async ({ page }) => {
  test.setTimeout(150_000); // the lab model trains once per backend process (a few seconds)
  const errors = watchConsole(page);
  await open(page);
  await workspace(page, /Transformer/).click();
  await expect(page.getByTestId('attention-matrix')).toBeVisible({ timeout: 120_000 });
  await expect(page.getByText(/Not GPT, ChatGPT or Claude/)).toBeVisible();

  await page.getByRole('button', { name: 'the cat sat on', exact: true }).click();
  await expect(page.getByTestId('token-strip').getByText('<bos>', { exact: true })).toBeVisible();
  await expect(page.getByTestId('token-strip').getByText('sat', { exact: true })).toBeVisible();
  await expect(page.getByText(/score = \(q·k\) × 1\/√16/)).toBeVisible();

  await page.getByRole('tab', { name: 'Next token' }).click();
  await page.getByRole('button', { name: /L0 · H0 on/ }).click();
  await expect(page.getByText('Orange tick = probability before the ablation.')).toBeVisible();
  await page.getByRole('button', { name: 'Restore all heads' }).click();
  await expect(page.getByText('Orange tick = probability before the ablation.')).toBeHidden();
  expect(errors).toEqual([]);
});
