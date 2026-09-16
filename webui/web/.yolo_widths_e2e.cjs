/** Parametric overflow measurement: real webui + real yolo box, video in
 *  (32 frames, like the user's case), measure which container actually
 *  scrolls horizontally, at several viewport widths. */
const { chromium } = require('playwright-core');
const path = require('path');
const os = require('os');

const WIDTHS = (process.env.WIDTHS || '900,1024,1280,1440').split(',').map((x) => parseInt(x, 10));

(async () => {
  const browser = await chromium.launch({
    executablePath: path.join(os.homedir(), '.cache/ms-playwright/chromium_headless_shell-1155/chrome-linux/headless_shell'),
  });

  for (const W of WIDTHS) {
    const page = await browser.newPage({ viewport: { width: W, height: 900 } });
    await page.goto('http://127.0.0.1:8090/index.html#/box/yolo', { waitUntil: 'networkidle' });
    const vids = ['images/video_frames', 'input'];
    // upload the kitchen video through the images widget (browser extracts frames)
    await page.setInputFiles('label.fld:has-text("images") input[type="file"]', '/home/manuelf/boxes/cozinha.mp4');
    // wait until frame extraction + upload finish (spinner gone, chips present)
    await page.waitForFunction(() => {
      const chips = document.querySelectorAll('label.fld .filechip').length;
      const spin = [...document.querySelectorAll('.note')].some((n) => /extracting|uploading/.test(n.textContent || ""));
      return chips >= 20 && !spin;
    }, null, { timeout: 300000 });
    await page.click('button:has-text("Call")');
    await page.waitForSelector('.tablewrap', { timeout: 240000 });
    await page.waitForTimeout(1200);

    const m = await page.evaluate(() => {
      const de = document.documentElement;
      const main = document.querySelector('main');
      const tw = document.querySelector('.tablewrap');
      const tbl = tw && tw.querySelector('table');
      const ig = document.querySelector('.imggrid');
      const pick = (el) => el && {
        cw: el.clientWidth, sw: el.scrollWidth,
        h: Math.round(el.getBoundingClientRect().height),
      };
      return {
        innerW: window.innerWidth,
        doc: { cw: de.clientWidth, sw: de.scrollWidth },
        main: pick(main),
        tablewrap: pick(tw),
        table: tbl ? { sw: tbl.scrollWidth, w: Math.round(tbl.getBoundingClientRect().width) } : null,
        imggrid: pick(ig),
        pageRect: (document.querySelector('.page') || {}).getBoundingClientRect && (() => { const r = document.querySelector('.page').getBoundingClientRect(); return { l: Math.round(r.left), w: Math.round(r.width) }; })(),
      };
    });
    console.log(`width=${W}`, JSON.stringify(m));
    await page.close();
  }
  await browser.close();
})().catch((e) => { console.error('FATAL', e.message); process.exit(1); });
