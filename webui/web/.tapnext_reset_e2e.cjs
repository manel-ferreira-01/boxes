/** Live check: tapnext console Reset button must reset the tracks viewer.
 *
 *  1) upload dog.jpg, Call twice -> tracks player accumulates steps (2)
 *  2) click Reset -> header chip "done", player shows "no tracked steps yet",
 *     history shrinks to the reset entry only
 *  3) Call again -> fresh session, step 1 / 1
 *  4) no page/console errors
 */
const { chromium } = require('playwright-core');
const path = require('path');
const os = require('os');

const log = (k, v) => console.log(`  ${k}: ${v}`);
let failures = 0;
const check = (cond, msg) => {
  console.log(`  ${cond ? 'PASS' : 'FAIL'} — ${msg}`);
  if (!cond) failures += 1;
};

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

  try {
    await page.goto('http://127.0.0.1:8090/index.html#/box/tapnext', { waitUntil: 'networkidle' });

    await page.locator('input[type="file"]').first()
      .setInputFiles('/home/manuelf/boxes/images/clip/test/dog.jpg');

    // ---- two track calls accumulate steps in the viewer ----
    console.log('track call 1/2 …');
    await page.locator('button', { hasText: /^Call$/ }).click();
    await page.waitForSelector('text=/step 1 \\/ 1/', { timeout: 120000 });

    console.log('track call 2/2 …');
    await page.locator('button', { hasText: /^Call$/ }).click();
    await page.waitForSelector('text=/step 1 \\/ 2/', { timeout: 120000 });
    const stepText2 = await page.locator('text=/step \\d+ \\/ \\d+/').first().textContent();
    const histEntries = await page.locator('.hist .item').count();
    log('viewer', stepText2.trim());
    check(histEntries >= 2, `history has >= 2 entries (got ${histEntries})`);
    check(stepText2.includes('/ 2'), 'viewer accumulated 2 steps before reset');

    // ---- Reset: box session clears AND the viewer must follow ----
    console.log('clicking Reset …');
    await page.locator('button', { hasText: /^Reset$/ }).click();
    await page.waitForSelector('.resulthead .chip.ok:has-text("done")', { timeout: 30000 });
    check(true, 'reset answered with status "done"');

    await page.waitForSelector('text=/no tracked steps yet/', { timeout: 30000 });
    check(true, 'tracks viewer shows "no tracked steps yet" after reset');
    const histAfter = await page.locator('.hist .item').count();
    log('history entries after reset', histAfter);
    check(histAfter === 1, `history holds only the reset entry (got ${histAfter})`);

    // ---- a fresh sequence starts from step 1 ----
    console.log('track call after reset …');
    await page.locator('button', { hasText: /^Call$/ }).click();
    await page.waitForSelector('text=/step 1 \\/ 1/', { timeout: 120000 });
    const stepText3 = await page.locator('text=/step \\d+ \\/ \\d+/').first().textContent();
    check(stepText3.includes('/ 1'), `fresh session starts at step 1/1 (got ${stepText3.trim()})`);
    check(errors.length === 0, 'no page/console errors' + (errors.length ? `: ${errors.join(' | ')}` : ''));
  } finally {
    await browser.close();
  }

  console.log(failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`);
  if (failures) process.exitCode = 1;
})().catch((e) => { console.error('E2E ERROR:', e.message); process.exitCode = 1; });
