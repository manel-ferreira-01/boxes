/** Live check of the moge `points` visualizer (real webui + real moge box).
 *
 *  1) upload dog.jpg through the UI, Call the box
 *  2) wait for the point-cloud block's note ("… reprojected from depth …")
 *     — that only appears after depth+intrinsics+image were assembled
 *  3) assert no page/console errors, count canvases, screenshot for
 *     the eyeball pass (the webui guide's remaining step). */
const { chromium } = require('playwright-core');
const path = require('path');
const os = require('os');

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });
  const page = await browser.newPage({ viewport: { width: 1280, height: 1400 } });
  const errors = [];
  page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error') errors.push('CONSOLE: ' + m.text()); });

  await page.goto('http://127.0.0.1:8090/index.html#/box/moge', { waitUntil: 'networkidle' });
  await page.waitForSelector('input[type="file"]', { state: 'attached', timeout: 30000 });
  await page.setInputFiles('input[type="file"]', '/home/manuelf/boxes/images/clip/test/dog.jpg');
  await page.click('button:has-text("Call")');
  await page.waitForSelector('text=/reprojected from depth/', { timeout: 300000 });
  const note = await page.locator('text=/reprojected from depth/').first().textContent();
  await page.waitForTimeout(2500);   // let a few RAF frames of points render
  const canvases = page.locator('.glbview canvas');
  await canvases.first().waitFor({ timeout: 15000 });
  console.log('cloud note:', note.trim());
  console.log('canvases:', await canvases.count());
  await canvases.first().screenshot({ path: '/tmp/moge_points.png' });
  console.log('errors:', JSON.stringify(errors));
  if (errors.length) { console.error('FAIL: page errors'); process.exit(1); }
  console.log('OK — screenshot at /tmp/moge_points.png');
  await browser.close();
})().catch((e) => { console.error('FATAL', e.message); process.exit(1); });
