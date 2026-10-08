import { expect, test } from '@playwright/test';
import { open, workspace } from './helpers';

test('M0 · page opts out of browser translation', async ({ page }) => {
  await open(page);
  await expect(page.locator('html')).toHaveAttribute('translate', 'no');
  await expect(page.locator('meta[name="google"]')).toHaveAttribute('content', 'notranslate');
});

test('M0 · DOM rewritten like a page translator never leaves a blank screen', async ({ page }) => {
  await open(page);
  await page.getByRole('button', { name: /^(Build|Costruisci)$/ }).first().click();
  await page.waitForTimeout(1500);
  // What Chrome Translate does: replace every text node with <font><font>translated</font></font>.
  await page.evaluate(() => {
    const walker = document.createTreeWalker(document.getElementById('root')!, NodeFilter.SHOW_TEXT);
    const nodes: Text[] = [];
    while (walker.nextNode()) nodes.push(walker.currentNode as Text);
    for (const n of nodes) {
      if (!n.textContent?.trim() || !n.parentNode) continue;
      const outer = document.createElement('font');
      const inner = document.createElement('font');
      inner.textContent = n.textContent.toUpperCase();
      outer.appendChild(inner);
      n.parentNode.replaceChild(outer, n);
    }
  });
  // Drive React updates over the rewritten DOM.
  for (const name of [/time/i, /forward|avanti/i, /analysis|analisi/i, /^network|^rete/i]) {
    await workspace(page, name).click({ timeout: 3000 }).catch(() => undefined);
    await page.waitForTimeout(300);
  }
  // Either the app still renders, or an error boundary explains what happened: never an empty page.
  const visibleText = await page.evaluate(() => document.body.innerText.trim().length);
  expect(visibleText).toBeGreaterThan(40);
});
