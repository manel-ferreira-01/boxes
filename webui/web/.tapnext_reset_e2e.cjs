/** Live check: tapnext Reset AND page-reload must not mix a dead box
 *  session with a fresh viewer.
 *
 *  Wire-level signal: the tracks tensor's F dimension.  A contaminated
 *  session returns F > the frames sent in this call (dead frames + new
 *  frames); a clean one returns F === frames sent.
 *
 *  1) track x2 in one session   -> F=1 then F=2 (accumulation is intended)
 *  2) Reset                     -> done chip, viewer cleared, history=1
 *  3) track                     -> F=1 (fresh box session)
 *  4) page reload               -> NEW session id (old code kept the old one)
 *  5) track                     -> F=1  (old code: F=2, dead session reused)
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
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

  // capture the tracks F dimension from every /api/call response
  let lastF = null;
  page.on('response', (resp) => {
    if (!resp.url().includes('/api/call') || resp.status() !== 200) return;
    resp.json().then((j) => {
      const t = j.fields && j.fields.tracks;
      lastF = t && Array.isArray(t.shape) ? t.shape[0] : null;
    }).catch(() => {});
  });

  const upload = async () =>
    page.locator('input[type="file"]').first()
      .setInputFiles('/home/manuelf/boxes/images/clip/test/dog.jpg');
  const call = async () => page.locator('button', { hasText: /^Call$/ }).click();

  try {
    await page.goto('http://127.0.0.1:8090/index.html#/box/tapnext', { waitUntil: 'networkidle' });
    await upload();

    // ---- 1) one session accumulates (intended box behaviour) ----
    await call();
    await page.waitForSelector('text=/step 1 \\/ 1/', { timeout: 120000 });
    check(lastF === 1, `first track: F=1 (got ${lastF})`);

    await call();
    await page.waitForSelector('text=/step 1 \\/ 2/', { timeout: 120000 });
    check(lastF === 2, `second track, same session: F=2 (got ${lastF})`);

    // ---- 2) Reset: box session + viewer both clear ----
    await page.locator('button', { hasText: /^Reset$/ }).click();
    await page.waitForSelector('.resulthead .chip.ok:has-text("done")', { timeout: 30000 });
    await page.waitForSelector('text=/no tracked steps yet/', { timeout: 30000 });
    check((await page.locator('.hist .item').count()) === 1, 'reset: history holds only the reset entry');

    // ---- 3) track again: fresh box session ----
    await call();
    await page.waitForSelector('text=/step 1 \\/ 1/', { timeout: 120000 });
    check(lastF === 1, `post-reset track: F=1 (got ${lastF})`);

    // ---- 4) reload: viewer is fresh -> box session must be fresh too ----
    const sidBefore = await page.locator('input.mono').first().inputValue();
    check(sidBefore.startsWith('ses_'), 'session id set before reload');
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForSelector('button:has-text("Call")', { timeout: 30000 });
    const sidAfter = await page.locator('input.mono').first().inputValue();
    log('session id', `${sidBefore} -> ${sidAfter}`);
    check(sidAfter.startsWith('ses_') && sidAfter !== sidBefore,
      'reload issues a NEW session id (old code kept the old one)');

    // ---- 5) tracking after reload must not span the dead session ----
    await upload();
    await call();
    await page.waitForSelector('text=/step 1 \\/ 1/', { timeout: 120000 });
    await sleep(300);   // let the response-capture promise settle
    check(lastF === 1, `post-reload track: F=1 (got ${lastF}; old code: 2 — dead session frames + new)`);
    check(errors.length === 0, 'no page/console errors' + (errors.length ? `: ${errors.join(' | ')}` : ''));
  } finally {
    await browser.close();
  }

  console.log(failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`);
  if (failures) process.exitCode = 1;
})().catch((e) => { console.error('E2E ERROR:', e.message); process.exitCode = 1; });
