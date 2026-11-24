// Render docs/media/blade-platform.svg to PNG with Chromium (Playwright).
// Usage (from the repository root): node docs/media/figure/render.js [scale]
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const scale = Number(process.argv[2] || 2);
  const root = path.resolve(__dirname, "..", "..", "..");
  const svg = path.join(root, "docs", "media", "blade-platform.svg");
  const png = path.join(root, "docs", "media", "blade-platform.png");
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 2400, height: 1000 },
    deviceScaleFactor: scale,
  });
  await page.goto("file://" + svg);
  await page.waitForTimeout(500);
  await page.screenshot({ path: png, fullPage: false });
  await browser.close();
  console.log("wrote", path.relative(root, png));
})();
