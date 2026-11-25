// Render docs/media/blade-platform(-dark).svg to PNG with Chromium (Playwright).
// Usage (from the repository root): node docs/media/figure/render.js [scale]
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const scale = Number(process.argv[2] || 2);
  const root = path.resolve(__dirname, "..", "..", "..");
  const browser = await chromium.launch();
  const page = await browser.newPage({
    viewport: { width: 2400, height: 1000 },
    deviceScaleFactor: scale,
  });
  for (const name of ["blade-platform", "blade-platform-dark"]) {
    const svg = path.join(root, "docs", "media", name + ".svg");
    const png = path.join(root, "docs", "media", name + ".png");
    await page.goto("file://" + svg);
    await page.waitForTimeout(500);
    await page.screenshot({ path: png, fullPage: false });
    console.log("wrote", path.relative(root, png));
  }
  await browser.close();
})();
