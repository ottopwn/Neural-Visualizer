import { expect, test } from '@playwright/test';
import { buildAndTrain, inspector, open, watchConsole, workspace } from './helpers';

for (const theme of ['dark', 'paper'] as const) {
  test(`A · build → train → Time Machine → historical Microscope (${theme})`, async ({ page }) => {
    const errors = watchConsole(page);
    await open(page, theme);
    await buildAndTrain(page);

    await workspace(page, /Time/).click();
    await expect(page.getByText(/21 stored checkpoints/)).toBeVisible();
    await page.getByRole('button', { name: 'First checkpoint' }).first().click();
    await expect(page.getByTestId('state-pill').first()).toContainText('HISTORICAL · EPOCH 0');

    // Select a neuron in the Time Machine's network pane: the Microscope shows epoch 0.
    await page.getByRole('button', { name: 'Inspect neuron L1N3' }).click();
    const insp = inspector(page);
    await expect(insp.getByText('L1N3', { exact: true })).toBeVisible();
    await expect(insp.getByText(/REAL · checkpoint · epoch 0/)).toBeVisible();

    // Back to live: the same neuron, now at the latest epoch.
    await page.getByRole('button', { name: 'Return to the live model' }).first().click();
    await expect(insp.getByText(/REAL · live weights · epoch 20/)).toBeVisible();
    expect(errors).toEqual([]);
  });
}

test('B · What-if: disable a neuron → before/after → undo', async ({ page }) => {
  const errors = watchConsole(page);
  await open(page);
  await buildAndTrain(page);
  await page.getByRole('button', { name: 'Inspect neuron L1N2' }).click();
  const insp = inspector(page);
  await insp.getByRole('button', { name: 'Disable L1N2' }).click();
  const result = page.getByTestId('whatif-result');
  await expect(result).toBeVisible();
  await expect(result.getByText('Dataset accuracy')).toBeVisible();
  await expect(page.getByTestId('state-pill').first()).toContainText('WHAT-IF ×1');
  await result.getByRole('button', { name: 'Undo' }).click();
  await expect(result).toBeHidden();
  expect(errors).toEqual([]);
});

test('C · Forward explorer shows the real weighted sum', async ({ page }) => {
  const errors = watchConsole(page);
  await open(page);
  await buildAndTrain(page);
  await workspace(page, /Forward|Pass/).click();
  await page.getByRole('button', { name: 'Next step' }).click();
  await expect(page.getByRole('heading', { name: /Dense 1: linear transform/ })).toBeVisible();
  await expect(page.getByText(/Σ w·a = .*z = /)).toBeVisible();
  await page.getByRole('button', { name: 'Next step' }).click();
  await expect(page.getByRole('heading', { name: /Dense 1: activation/ })).toBeVisible();
  await page.getByRole('button', { name: 'Last step' }).click();
  await expect(page.getByRole('heading', { name: 'Prediction' })).toBeVisible();
  expect(errors).toEqual([]);
});

test('D · Backward explorer shows real gradients and an SGD preview', async ({ page }) => {
  const errors = watchConsole(page);
  await open(page);
  await buildAndTrain(page);
  await workspace(page, /Forward|Pass/).click();
  await page.getByRole('button', { name: /Backward pass/ }).click();
  await expect(page.getByRole('heading', { name: /Loss L = −log p\(target\)/ })).toBeVisible();
  await page.getByRole('button', { name: 'Output · dW' }).click();
  await expect(page.getByRole('heading', { name: /Output: parameter gradients/ })).toBeVisible();
  await expect(page.getByText('One real SGD step on this sample (preview)')).toBeVisible();

  // Same step at another epoch: the comparison columns appear.
  await page.getByLabel('Compare with epoch').selectOption({ label: 'epoch 0' });
  await expect(page.getByText(/Comparing with the same probe at epoch 0/)).toBeVisible();
  expect(errors).toEqual([]);
});

test('E · 3D view renders the real network and discloses edge filtering', async ({ page }) => {
  const errors = watchConsole(page);
  await open(page);
  await buildAndTrain(page);
  await workspace(page, /^3D$/).click();
  const view = page.getByTestId('forge-3d');
  await expect(view.locator('canvas')).toBeVisible();
  await expect(page.getByTestId('edge-disclosure')).toContainText(/All 96 connections shown/);
  await page.getByLabel('Maximum connections drawn').selectOption('100');
  await expect(page.getByTestId('edge-disclosure')).toContainText(/Showing|All 96/);
  await page.getByRole('button', { name: /Select and focus Dense 1/ }).click();
  await expect(inspector(page).getByText('Dense 1', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Backward pass' }).click();
  await page.getByRole('button', { name: 'Next pass step' }).click();
  await expect(page.getByText('Gradient at the logits: dL/dz = p − y')).toBeVisible();
  expect(errors).toEqual([]);
});
