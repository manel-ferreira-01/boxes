/** Live check of the yolo console (def-driven, generic UI):
 *  - console shows a generated session id
 *  - every detection response carries track_ids aligned with boxes
 *  - ids are STABLE across calls within one session (per-call capture!)
 *  - reset (session action) is scoped to the session; next call gets fresh ids
 *  - list (session action) returns the sessions array including this session
 */
const { chromium } = require('playwright-core');
const path = require('path');
const os = require('os');

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

  await page.addInitScript(() => { window.__calls = []; });
  page.on('response', (resp) => {
    if (!resp.url().includes('/api/call') || resp.status() !== 200) return;
    resp.json().then((j) => {
      page.evaluate((entry) => window.__calls.push(entry), {
        action: (j.config_extra && j.config_extra.action) || null,
        session: (j.config_extra && j.config_extra.session) || null,
        sessions: (j.config_extra && j.config_extra.sessions) || null,
        boxes: j.fields && j.fields.detections && j.fields.detections[0] && (j.fields.detections[0].boxes || []).length,
        track_id: j.fields && j.fields.detections && j.fields.detections[0] && j.fields.detections[0].track_id,
      }).catch(() => {});
    }).catch(() => {});
  });
  const nCalls = () => page.evaluate(() => window.__calls.length);
  const calls = () => page.evaluate(() => window.__calls);
  const waitFor = async (cond, what, timeoutMs) => {
    const t0 = Date.now();
    while (Date.now() - t0 < timeoutMs) {
      if (await cond()) return true;
      await sleep(250);
    }
    console.log(`  (timeout waiting: ${what})`);
    return false;
  };

  try {
    await page.goto('http://127.0.0.1:8090/index.html#/box/yolo', { waitUntil: 'networkidle' });

    const sid0 = await page.locator('input.mono').first().inputValue();
    check(sid0.startsWith('ses_'), `console shows a generated session id (${sid0})`);

    await page.locator('input[type="file"]').first()
      .setInputFiles('/home/manuelf/boxes/images/clip/test/dog.jpg');

    // ---- call 1 --------------------------------------------------------
    await page.locator('button', { hasText: /^Call$/ }).click();
    let ok = await waitFor(async () => (await nCalls()) >= 1, 'call1 response', 180000);
    let cs = await calls();
    check(ok && cs.length === 1, `call1 answered (${cs.length} call response(s))`);
    const c1 = cs[0] || {};
    check(c1.session === sid0, `call1 echoes console session (${c1.session})`);
    check(Array.isArray(c1.track_id) && c1.track_id.length === c1.boxes && c1.boxes > 0,
      `call1: ${c1.boxes} box(es), track_ids=${JSON.stringify(c1.track_id)}`);
    const id1 = c1.track_id && c1.track_id[0];

    // ---- call 2 (same session) ----------------------------------------
    await page.locator('button', { hasText: /^Call$/ }).click();
    ok = await waitFor(async () => (await nCalls()) >= 2, 'call2 response', 120000);
    cs = await calls();
    const c2 = cs[1] || {};
    check(ok, `call2 answered`);
    check(c2.track_id && c2.track_id[0] === id1,
      `ids stable across calls in one session (${id1} -> ${c2.track_id && c2.track_id[0]})`);

    // ---- reset (session action) -----------------------------------------
    await page.locator('button', { hasText: /^reset$/ }).first().click();
    ok = await waitFor(async () => {
      const list = await calls();
      return list.some((c) => c.action === 'reset');
    }, 'reset response', 30000);
    const r0 = (await calls()).find((c) => c.action === 'reset');
    check(ok && r0 && r0.session === sid0, `reset scoped to session (action=${r0 && r0.action}, session=${r0 && r0.session})`);

    // ---- call 3 (same session, post-reset) -> fresh id -------------------
    await page.locator('button', { hasText: /^Call$/ }).click();
    ok = await waitFor(async () => (await nCalls()) >= 4, 'call3 response', 120000);
    cs = await calls();
    const resetIdx = cs.findIndex((c) => c.action === 'reset');
    const c3 = (cs.slice(resetIdx + 1).find((c) => c.action === null) || cs[cs.length - 1]) || {};
    check(ok, `call3 answered`);
    check(c3.track_id && typeof c3.track_id[0] === 'number',
      `post-reset call has a fresh track id (${c3.track_id && c3.track_id[0]})`);
    check(c3.track_id && c3.track_id[0] !== id1, `post-reset id differs from pre-reset (${id1} -> ${c3.track_id && c3.track_id[0]})`);

    // ---- list (session action) ------------------------------------------
    await page.locator('button', { hasText: /^list$/ }).first().click();
    ok = await waitFor(async () => {
      const list = await calls();
      return list.some((c) => c.action === 'list');
    }, 'list response', 30000);
    const l0 = (await calls()).find((c) => c.action === 'list');
    check(ok && l0, `list answered (action=${l0 && l0.action})`);
    check(Array.isArray(l0 && l0.sessions) && l0.sessions.some((s) => s.session === sid0),
      `list includes this session (${JSON.stringify(l0 && l0.sessions)})`);

    check(errors.length === 0, 'no page/console errors' + (errors.length ? `: ${errors.join(' | ')}` : ''));
  } finally {
    await browser.close();
  }

  console.log(failures === 0 ? 'ALL CHECKS PASSED' : `${failures} CHECK(S) FAILED`);
  if (failures) process.exitCode = 1;
})().catch((e) => { console.error('E2E ERROR:', e.message); process.exitCode = 1; });
