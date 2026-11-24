// Render the campaign cards (docs/media/campaign/card-*.svg) to PNG.
// Usage (from the repository root): node docs/media/campaign/render.js
const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const dir = __dirname;
  const browser = await chromium.launch();
  for (const file of fs.readdirSync(dir).filter((f) => /^card-.*\.svg$/.test(f))) {
    const svg = fs.readFileSync(path.join(dir, file), "utf8");
    const [, w, h] = svg.match(/width="(\d+)" height="(\d+)"/);
    const page = await browser.newPage({
      viewport: { width: Number(w), height: Number(h) },
      deviceScaleFactor: 1,
    });
    await page.goto("file://" + path.join(dir, file));
    await page.waitForTimeout(300);
    const png = path.join(dir, file.replace(/\.svg$/, ".png"));
    await page.screenshot({ path: png });
    await page.close();
    console.log("wrote", path.basename(png));
  }
  await browser.close();
})();
