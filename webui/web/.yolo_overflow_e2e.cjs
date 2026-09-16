/** Reproduce the results-panel horizontal overflow: real webui + real yolo
 *  box, a few images in, then measure what sticks out of the viewport. */
const { chromium } = require('playwright-core');
const path = require('path');
const os = require('os');

const VIEW = parseInt(process.env.VIEW_W || '1280', 10);

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });
  const page = await browser.newPage({ viewport: { width: VIEW, height: 1000 } });
  await page.goto('http://127.0.0.1:8090/index.html#/box/yolo', { waitUntil: 'networkidle' });

  const imgs = [
    '/home/manuelf/boxes/images/yolo/test/dog.jpg',
    '/home/manuelf/boxes/images/yolo/test/car.jpg',
    '/home/manuelf/boxes/images/yolo/test/dog.jpg',
    '/home/manuelf/boxes/images/yolo/test/car.jpg',
  ];
  await page.setInputFiles('label.fld:has-text("images") input[type="file"]', imgs);
  await page.click('button:has-text("Call")');
  await page.waitForSelector('.tablewrap', { timeout: 180000 });
  await page.waitForTimeout(800);   // let images/lazy bits settle

  const m = await page.evaluate(() => {
    const vw = document.documentElement.clientWidth;
    const sw = document.documentElement.scrollWidth;
    const out = [];
    for (const el of document.querySelectorAll('*')) {
      const r = el.getBoundingClientRect();
      if (r.width > 0 && r.right > vw + 1) {
        out.push({
          tag: el.tagName.toLowerCase(),
          cls: String(el.className || '').slice(0, 70),
          w: Math.round(r.width),
          right: Math.round(r.right),
        });
      }
    }
    out.sort((a, b) => b.right - a.right);
    // the table's own width for good measure
    const tbl = document.querySelector('.tablewrap table');
    return { vw, sw, table: tbl ? Math.round(tbl.getBoundingClientRect().width) : null, offenders: out.slice(0, 15) };
  });
  console.log(JSON.stringify(m, null, 1));
  await page.screenshot({ path: '/tmp/yolo_overflow.png' });
  await browser.close();
})().catch((e) => { console.error('FATAL', e.message); process.exit(1); });
