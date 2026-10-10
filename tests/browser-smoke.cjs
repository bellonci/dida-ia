const { chromium } = require('playwright');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce', colorScheme: 'dark' });
  const page = await context.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  const base = process.env.DIDA_TEST_URL || 'http://127.0.0.1:8766';
  try {
  await page.goto(base);
  await page.waitForFunction(() => document.getElementById('connectionText').textContent === 'Pronta para conversar');
  if (await page.locator('#profileDialog').evaluate(el => el.open)) await page.locator('#profileLater').click();
  await page.locator('#welcome').waitFor({ state: 'visible' });
  await page.screenshot({ path: 'artifacts/dida-desktop-dark.png', fullPage: true });
  await page.locator('.suggestion').first().click();
  assert.match(await page.locator('#input').inputValue(), /organizar uma ideia/);
  await page.locator('#input').fill('oi');
  await page.locator('#input').press('Enter');
  await page.getByText('Não consegui gerar uma resposta agora.', { exact: false }).waitFor();
  assert.equal(await page.locator('#welcome').isVisible(), false);
  await page.locator('#memoryOpen').click();
  await page.getByText('Seu histórico está vazio.', { exact: true }).waitFor();
  await page.locator('#memoryEnabled').uncheck();
  await page.getByText('Memória desativada. Seu histórico foi mantido.', { exact: true }).waitFor();
  await page.locator('#memoryEnabled').check();
  await page.getByText('Memória ativada.', { exact: true }).waitFor();
  await page.locator('#memoryClear').click();
  assert.equal(await page.locator('#memoryConfirm').isVisible(), true);
  await page.locator('#memoryKeep').click();
  assert.equal(await page.locator('#memoryConfirm').isVisible(), false);
  await page.locator('#memoryClose').click();
  await page.locator('#voiceOpen').click();
  await page.getByText('A amostra não altera suas preferências.', { exact: false }).waitFor();
  await page.locator('#voiceClose').click();
  await page.locator('#filesOpen').click();
  await page.locator('#filesDialog').waitFor({ state: 'visible' });
  await page.locator('#filesClose').click();
  await page.locator('#themeToggle').click();
  assert.equal(await page.locator('html').getAttribute('data-theme'), 'light');
  await page.screenshot({ path: 'artifacts/dida-desktop-light.png', fullPage: true });
  for (const width of [390, 720, 1024]) {
    await page.setViewportSize({ width, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, `horizontal overflow at ${width}`);
    assert.equal(await page.locator('#input').isVisible(), true);
    await page.locator('#memoryOpen').click();
    await page.locator('#memoryDialog').waitFor({ state: 'visible' });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.locator('#memoryClose').click();
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: 'artifacts/dida-mobile.png', fullPage: true });
  assert.deepEqual(errors, []);
  console.log('PASS: desktop/mobile, themes, suggestions, send, dialogs, memory toggles and deletion confirmation; no JS errors');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
